import { sharedIndex } from '../../upstream/bot-starter/src/game/dictionary.ts';
import { LETTER_VALUES, type Counts } from '../../upstream/bot-starter/src/game/letters.ts';
import type { Decision } from '../../upstream/bot-starter/src/game/types.ts';
import { bidActions } from '../actions.ts';
import { publicEvidence, type Evidence } from '../belief.ts';
import { encode } from '../features.ts';
import { advance, observe, type GameState, type Observation } from '../game.ts';
import type { DecisionPoint } from '../planning.ts';
import { bidSupport, type Style } from './market.ts';

export type Position = Omit<DecisionPoint, 'informationSet'>;
interface CashPlan { words: string[]; rack: Counts; cash: number }
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Cache word queries within a single search. Repeated counterfactuals often
 * have identical racks. Every returned plan is treated as immutable. */
export class WordCache {
  private values = new Map<string, number>();
  private plans = new Map<string, CashPlan>();
  value(rack: Counts): number {
    const key = String.fromCharCode(...rack);
    let value = this.values.get(key);
    if (value === undefined) {
      value = sharedIndex().greedyValue(rack, 2);
      this.values.set(key, value);
    }
    return value;
  }
  gain(rack: Counts, letter: number): number {
    const added = rack.slice();
    added[letter]!++;
    return this.value(added) - this.value(rack);
  }
  cash(rack: Counts, minLength: number): CashPlan {
    const key = String.fromCharCode(minLength, ...rack);
    const cached = this.plans.get(key);
    if (cached) return cached;
    const left = rack.slice(), words: string[] = [];
    const index = sharedIndex();
    let cash = 0;
    for (let sig = index.bestRewardSig(left, minLength); sig >= 0; sig = index.bestRewardSig(left, minLength)) {
      words.push(index.sigWord[sig]!);
      cash += index.sigReward[sig]!;
      index.subtract(left, sig);
    }
    const plan = { words, cash, rack: left };
    this.plans.set(key, plan);
    return plan;
  }
}

function cashLength(style: Style, remaining: number, funds: number): number {
  if (style === 'idle') return 99;
  if (style === 'face' || style === 'random' || style === 'premium') return 3;
  if (style === 'patient') return 7;
  if (remaining === 0) return 2;
  if (funds < 9 || remaining < 3) return 3;
  return 7;
}

/** The fast continuation sees this player's own rack and public inputs.
 * Its random quantile is fixed per particle, seat and auction across bids. */
function modeledDecision(s: GameState, seat: number, style: Style, u: number, cache: WordCache): Decision {
  if (style === 'idle') return { words: [], bid: 0 };
  const p = s.players[seat]!;
  const plan = cache.cash(p.rack, cashLength(style, s.bagTotal, p.balance));
  const gain = style === 'completion' ? Math.max(0, cache.gain(plan.rack, s.letter)) : 0;
  const support = bidSupport(style, { mode: s.auctionType, letter: s.letter, balance: p.balance + plan.cash, remaining: s.bagTotal, gain });
  return { words: plan.words, bid: support[Math.floor(u * support.length)]! };
}

export function innerPosition(s: GameState, seat: number, cache: WordCache): { point: Position; evidence: Evidence } {
  // Closed round records are immutable in advance(). Internal lookahead can
  // share them while keeping each branch's history array independent.
  const o = observe(s, seat, false), evidence = publicEvidence(o);
  const p = s.players[seat]!;
  const plan = cache.cash(p.rack, cashLength('completion', s.bagTotal, p.balance));
  const decision = modeledDecision(s, seat, 'completion', 0.5, cache);
  return { evidence, point: {
    observation: o, words: plan.words, cashPoints: plan.cash, postCashRack: plan.rack,
    baselineBid: decision.bid, actions: bidActions(decision.bid, p.balance + plan.cash, o.bagTotal === 0, o.players.some(p => p.sim)),
    features: encode(o, plan.rack, decision.bid, plan.cash, evidence),
  } };
}

export interface Particle { state: GameState; styles: Style[]; noise: number[][] }
export function copyParticleState(s: GameState): GameState {
  // advance replaces RNG snapshots and only appends to history. It never
  // edits an initial rack or a previously completed auction.
  return { ...s, bag: s.bag.slice(), history: s.history.slice(), players: s.players.map(p => ({
    ...p, rack: p.rack.slice(), bids: p.bids.slice(),
  })) };
}
export function simulateAuction(s: GameState, seat: number, mine: Decision, particle: Particle, depth: number, cache: WordCache): void {
  const joint = s.players.map((p, i) => {
    if (i === seat) return mine;
    if (p.sim) return null;
    return modeledDecision(s, i, particle.styles[i]!, particle.noise[depth]![i]!, cache);
  });
  advance(s, joint);
}
export function continuation(s: GameState, seat: number, cache: WordCache): Decision {
  const decision = modeledDecision(s, seat, 'completion', 0.5, cache);
  const hasModule = s.players.some(p => p.sim);
  if (s.bagTotal === 0 || (s.bagTotal === 1 && !hasModule)) {
    const p = s.players[seat]!;
    const budget = p.balance + cache.cash(p.rack, cashLength('completion', s.bagTotal, p.balance)).cash;
    decision.bid = s.bagTotal === 0 && !hasModule ? 0 : budget;
  }
  return decision;
}

/** Margins help distinguish decisions in early rollouts. Split wins and
 * shared rank receive most of the weight in completed games. */
export function terminalValue(scores: readonly number[], seat: number): number {
  const mine = scores[seat]!, rivals = scores.filter((_, i) => i !== seat);
  const top = Math.max(...scores);
  const win = mine === top ? 1 / scores.filter(s => s === top).length : 0;
  const rank = 1 - rivals.filter(s => s > mine).length / rivals.length;
  return 0.6 * win + 0.2 * rank + 0.2 * sigmoid((mine - Math.max(...rivals)) / 45);
}

/** Project standings from visible racks, public finances and auctions left.
 * Coins only influence the share of future scoring opportunities. */
export function publicValue(o: Observation, evidence: Evidence, cache: WordCache): number {
  const time = Math.max(0, o.bagTotal);
  const potential = o.players.map((_, i) => {
    const rack = i === o.seat ? o.rack : evidence.minimumRack[i]!;
    const hidden = i === o.seat ? 0 : evidence.hiddenDealSizes[i]!;
    const playable = cache.value(rack) + 2.6 * hidden;
    const face = rack.reduce((sum, count, j) => sum + count * LETTER_VALUES[j]!, 0) + 1.9 * hidden;
    return playable + Math.min(0.65, time / 20) * Math.max(0, 2.35 * face - playable);
  });
  const power = potential.map((v, i) => Math.pow(3 + o.balances[i]! + 0.55 * v, 0.8));
  const totalPower = power.reduce((a, b) => a + b, 0);
  const scores = o.scores.map((s, i) => s + potential[i]! + 4.3 * time * power[i]! / totalPower);
  const mine = scores[o.seat]!, rivals = scores.filter((_, i) => i !== o.seat);
  const temp = 5 + 0.78 * time;
  const top = Math.max(...scores);
  const win = Math.exp((mine - top) / temp) / scores.reduce((sum, s) => sum + Math.exp((s - top) / temp), 0);
  const rank = rivals.reduce((sum, s) => sum + sigmoid((mine - s) / temp), 0) / rivals.length;
  return 0.6 * win + 0.2 * rank + 0.2 * sigmoid((mine - Math.max(...rivals)) / 45);
}

/** The parent edge already identifies our preceding decision and history. */
export function observationBranch(s: GameState, seat: number): string {
  const h = s.history.at(-1)!;
  return JSON.stringify([h.words, h.outcome.winner, h.outcome.paid, h.simWords, s.letter, s.players[seat]!.rack]);
}
