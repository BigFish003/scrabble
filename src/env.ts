import '../upstream/bot-starter/sim/strategies.ts';
import { createAgent } from '../upstream/bot-starter/src/strategy.ts';
import { Agent } from '../upstream/bot-starter/src/strategy/agent.ts';
import type { Params } from '../upstream/bot-starter/src/strategy/params.ts';
import { sharedIndex } from '../upstream/bot-starter/src/game/dictionary.ts';
import type { AuctionType } from '../upstream/bot-starter/src/game/letters.ts';
import type { Decision, Policy, WordPlayed } from '../upstream/bot-starter/src/game/types.ts';
import { makeSeat } from '../upstream/bot-starter/sim/registry.ts';
import type { MatchResult } from '../upstream/bot-starter/sim/engine.ts';
import { sampleWorld } from './belief.ts';
import { cloneGraph } from './clone.ts';
import { planDecision, type DecisionPoint } from './planning.ts';
import { advance, auctionView, createGame, matchStart, observe, result, type GameState, type Observation, type PublicEvent } from './game.ts';

export type Objective = 'score' | 'win' | 'rank' | 'margin';
export interface EnvConfig {
  opponents: string[]; auctionType?: AuctionType; learnerSeat?: number; rackSize?: number;
  objective?: Objective; plannerParams?: Partial<Params>;
}
export type { DecisionPoint } from './planning.ts';
export interface StepResult {
  observation: DecisionPoint | null; reward: number; terminated: boolean;
  info: { round: number; scoreDelta: number; outcome: Observation['history'][number]['outcome']; match: MatchResult | null };
}
export interface Replay { version: 1; config: EnvConfig; seed: number; actions: number[] }

export function utility(scores: readonly number[], seat: number, objective: Objective): number {
  const mine = scores[seat]!, rivals = scores.filter((_, i) => i !== seat);
  if (objective === 'score') return mine / 100;
  if (objective === 'margin') return (mine - Math.max(...rivals)) / 100;
  if (objective === 'rank') return 1 - rivals.filter(s => s > mine).length / rivals.length;
  const best = Math.max(...scores);
  return mine === best ? 1 / scores.filter(s => s === best).length : 0;
}

/** Single learner, one whole simultaneous auction per step. The word/cash
 * planner stays fixed. Learning and search control only its bid adjustment. */
export class ScrabbleEnv {
  readonly config: EnvConfig;
  private game!: GameState;
  private policies: (Policy | null)[] = [];
  private planner!: Agent;
  private pending: DecisionPoint | null = null;
  private seed = 0;
  private actionLog: number[] = [];
  private sampled = false;

  constructor(config: EnvConfig) {
    this.config = cloneGraph(config);
    const seat = config.learnerSeat ?? 0;
    if (!Number.isInteger(seat) || seat < 0 || seat > config.opponents.length) throw new Error('Invalid learner seat.');
    if (config.objective && !['score', 'win', 'rank', 'margin'].includes(config.objective)) throw new Error('Invalid reward objective.');
  }
  get seat(): number { return this.config.learnerSeat ?? 0; }
  get done(): boolean { return this.game?.done ?? false; }
  get objective(): Objective { return this.config.objective ?? 'score'; }

  reset(seed: number): DecisionPoint {
    this.seed = seed; this.actionLog = []; this.sampled = false; this.pending = null;
    const specs = this.config.opponents.map(name => makeSeat(name));
    specs.splice(this.seat, 0, { name: 'learner' });
    this.game = createGame({ seed, auctionType: this.config.auctionType ?? 'Vickrey', rackSize: this.config.rackSize,
      players: specs.map(s => ({ name: s.name, sim: s.sim ?? null, offline: !!s.offline })) });
    this.planner = createAgent('learner', sharedIndex(), this.config.plannerParams);
    this.policies = specs.map((s, i) => i === this.seat ? this.planner : s.make?.() ?? null);
    this.policies.forEach((p, i) => p?.onMatchStart(matchStart(this.game, i)));
    return this.decision();
  }

  /** Idempotent: planning occurs once, even when the caller inspects a node
   * repeatedly. Returned copies cannot mutate the live environment. */
  decision(): DecisionPoint {
    if (!this.game) throw new Error('Call reset(seed) first.');
    if (this.done) throw new Error('Terminal state has no decision.');
    if (!this.pending) {
      this.pending = planDecision(this.planner, observe(this.game, this.seat));
    }
    return cloneGraph(this.pending);
  }

  step(actionId: number): StepResult {
    const point = this.decision(), action = point.actions[actionId];
    if (!Number.isInteger(actionId) || !action?.legal) throw new Error(`Illegal or duplicate bid action: ${actionId}`);
    const before = this.game.players[this.seat]!.score;
    const decisions: (Decision | null)[] = this.policies.map((p, i) => i === this.seat
      ? { words: point.words, bid: action.bid } : p?.onAuction(auctionView(this.game, i)) ?? null);
    // Price learning must observe the submitted bid, including when RL skips.
    this.planner.noteSubmittedBid(action.bid);
    const events = advance(this.game, decisions);
    this.deliver(events);
    this.pending = null; this.actionLog.push(actionId);
    const scoreDelta = this.game.players[this.seat]!.score - before;
    const reward = this.objective === 'score' ? scoreDelta / 100 : this.done
      ? utility(this.game.players.map(p => p.score), this.seat, this.objective) : 0;
    const h = this.game.history.at(-1)!;
    return { observation: this.done ? null : this.decision(), reward, terminated: this.done,
      info: { round: h.round, scoreDelta, outcome: { ...h.outcome }, match: this.done ? this.result() : null } };
  }
  private deliver(events: PublicEvent[]): void {
    for (const event of events) for (const p of this.policies) {
      if (event.kind === 'word') p?.onWordPlayed(cloneGraph(event.value));
      else p?.onAuctionResult(cloneGraph(event.value));
    }
  }

  result(): MatchResult { return result(this.game); }
  /** Exact in-memory clone: includes policy memories and every RNG stream.
   * This contains simulator truth. Use searchRoot() for fair hidden-state search. */
  clone(): ScrabbleEnv { return cloneGraph(this); }
  snapshot(): ScrabbleEnv { return this.clone(); }
  restore(snapshot: ScrabbleEnv): void { Object.assign(this, snapshot.clone()); }
  /** Explicit diagnostic access, excluded from policy/training inputs. */
  debugState(): GameState { return structuredClone(this.game); }
  exportReplay(): Replay {
    if (this.sampled) throw new Error('Belief branches are in-memory search states, not seed-based episode replays.');
    return { version: 1, config: cloneGraph(this.config), seed: this.seed, actions: this.actionLog.slice() };
  }
  static fromReplay(replay: Replay): ScrabbleEnv {
    if (replay.version !== 1) throw new Error('Unsupported replay version.');
    const env = new ScrabbleEnv(replay.config); env.reset(replay.seed);
    for (const action of replay.actions) env.step(action);
    return env;
  }

  /** Only learner-visible observation + its own planner memory cross this
   * boundary. Real opponent sessions, random streams and hidden tiles do not.
   * Models are assumptions supplied by the caller, not the actual opponents.
   */
  searchRoot(seed: number, opponentModels?: string[]): ScrabbleEnv {
    const point = this.decision();
    return ScrabbleEnv.fromObservation(point.observation, seed, opponentModels, this.objective,
      this.planner, point, this.config.plannerParams);
  }

  static fromObservation(o: Observation, seed: number, opponentModels?: string[], objective: Objective = 'score',
    ownPlanner?: Agent, ownPoint?: DecisionPoint, plannerParams?: Partial<Params>): ScrabbleEnv {
    if (o.done) throw new Error('No search decision at a terminal observation.');
    const models = opponentModels ?? o.players.filter((_, i) => i !== o.seat).map(() => 'final');
    if (models.length !== o.players.length - 1) throw new Error('Supply one model for each opponent.');
    const env = new ScrabbleEnv({ opponents: models, learnerSeat: o.seat, auctionType: o.auctionType, rackSize: o.rackSize, objective, plannerParams });
    env.game = sampleWorld(o, seed); env.seed = seed; env.sampled = true;
    env.planner = createAgent('learner', sharedIndex(), plannerParams);
    let j = 0;
    env.policies = o.players.map((p, i) => {
      if (i === o.seat) return env.planner;
      const model = models[j++]!;
      if (p.sim) return null;
      const spec = makeSeat(model);
      if (!spec.make) throw new Error(`External opponent needs an external policy model: ${model}`);
      return spec.make();
    });
    env.replayModelHistory(o, !!(ownPlanner && ownPoint));
    if (ownPlanner && ownPoint) {
      env.planner = cloneGraph(ownPlanner); env.policies[o.seat] = env.planner;
      env.pending = cloneGraph(ownPoint);
    }
    return env;
  }

  /** Reconstruct model memories using sampled private initial racks and the
   * public transcript. Historical rival sealed bids are unobserved: their
   * modeled bids are an approximation, not a reconstruction of true memory. */
  private replayModelHistory(o: Observation, skipOwn = false): void {
    const racks = this.game.players.map(p => p.initialRack.slice());
    if (skipOwn) this.policies[this.seat] = null;
    this.policies.forEach((p, i) => p?.onMatchStart(matchStart(this.game, i)));
    const word = (w: WordPlayed) => { for (const ch of w.word) racks[w.player]![ch.charCodeAt(0) - 65]!--; };
    for (let k = 0; k < o.history.length; k++) {
      const h = o.history[k]!;
      this.policies.forEach((p, i) => p?.onAuction({ round: h.round, letter: h.letter, bagTotal: h.bagTotal,
        rack: racks[i]!.slice(), balances: h.balances.slice(), scores: h.scores.slice() }));
      if (!skipOwn) this.planner.noteSubmittedBid(o.ownBids[k] ?? 0);
      for (const w of h.words) { word(w); this.deliver([{ kind: 'word', value: w }]); }
      if (h.outcome.winner !== null) racks[h.outcome.winner]![h.letter.charCodeAt(0) - 65]!++;
      this.deliver([{ kind: 'result', value: h.outcome }]);
      for (const w of h.simWords) { word(w); this.deliver([{ kind: 'word', value: w }]); }
    }
    if (racks.some((r, p) => r.some((v, c) => v !== this.game.players[p]!.rack[c]))) throw new Error('Model-history replay failed.');
  }
}
