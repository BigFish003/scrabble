/** Adapter for the upstream headless Policy callback interface. It constructs
 * the same observation/features as the training env from private own-rack
 * views and public callbacks, and can run in upstream worker-thread matches.
 * Live server connection/reducer timing is intentionally handled upstream.
 */
import { createAgent } from '../upstream/bot-starter/src/strategy.ts';
import type { Agent } from '../upstream/bot-starter/src/strategy/agent.ts';
import type { AuctionOutcome, AuctionView, Decision, MatchStart, Policy, WordPlayed } from '../upstream/bot-starter/src/game/types.ts';
import type { Observation, RoundRecord } from './game.ts';
import { planDecision, type DecisionPoint } from './planning.ts';
import { ActorCritic } from './network.ts';
import { Rng } from './rng.ts';

export class LearnedBidPolicy implements Policy {
  readonly name: string;
  private network: ActorCritic;
  private deterministic: boolean;
  private planner!: Agent;
  private start!: MatchStart;
  private history: RoundRecord[] = [];
  private ownBids: number[] = [];
  private current: RoundRecord | null = null;
  private resolved = false;
  private rng = new Rng(0);
  constructor(network: ActorCritic, options: { name?: string; stochastic?: boolean } = {}) {
    this.network = network; this.name = options.name ?? 'learned'; this.deterministic = !options.stochastic;
  }
  onMatchStart(m: MatchStart): void {
    this.start = structuredClone(m); this.history = []; this.ownBids = []; this.current = null; this.resolved = false;
    this.rng = new Rng(m.seed ^ 0x61c88647);
    this.planner = createAgent(this.name); this.planner.onMatchStart(m);
  }
  onAuction(v: AuctionView): Decision {
    if (this.current) {
      if (!this.resolved) throw new Error('Auction opened before previous result.');
      this.history.push(this.current);
    }
    this.current = { round: v.round, letter: v.letter, bagTotal: v.bagTotal, balances: v.balances.slice(), scores: v.scores.slice(),
      words: [], outcome: { round: v.round, letter: v.letter, winner: null, paid: 0 }, simWords: [] };
    this.resolved = false;
    const m = this.start;
    const observation: Observation = { ...structuredClone(v), seat: m.seat, auctionType: m.auctionType, rackSize: m.rack.reduce((a, b) => a + b, 0),
      ownInitialRack: m.rack.slice(), players: m.names.map((name, i) => ({ name, sim: m.simStrategies[i]!, offline: !!m.offline?.[i] })),
      history: structuredClone(this.history), ownBids: this.ownBids.slice(), done: false };
    const point = planDecision(this.planner, observation);
    const bid = this.selectBid(point);
    this.planner.noteSubmittedBid(bid); this.ownBids.push(bid);
    return { words: point.words, bid };
  }
  protected selectBid(point: DecisionPoint): number {
    const action = this.network.act(point.features, point.actions.map(a => a.legal), this.rng, this.deterministic);
    return point.actions[action]!.bid;
  }
  onAuctionResult(r: AuctionOutcome): void {
    if (!this.current || this.current.round !== r.round) throw new Error('Unexpected auction result.');
    this.current.outcome = { ...r }; this.resolved = true;
    this.planner.onAuctionResult(r);
  }
  onWordPlayed(w: WordPlayed): void {
    if (!this.current) throw new Error('Word arrived before auction open.');
    (this.resolved ? this.current.simWords : this.current.words).push({ ...w });
    this.planner.onWordPlayed(w);
  }
}
