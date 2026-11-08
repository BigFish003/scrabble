import { LETTER_VALUES, VOWELS, type AuctionType } from '../../upstream/bot-starter/src/game/letters.ts';
import { simBid } from '../../upstream/bot-starter/sim/engine.ts';
import type { Observation, PlayerMeta, RoundRecord } from '../game.ts';
import type { Rng } from '../rng.ts';

export const STYLES = ['face', 'premium', 'patient', 'random', 'capital', 'completion', 'idle'] as const;
export type Style = typeof STYLES[number];
const PRIOR = [0.17, 0.14, 0.12, 0.12, 0.17, 0.26, 0.02];
export interface PriceContext { letter: number; balance: number; remaining: number; mode: AuctionType; gain?: number }

/** Finite distributions make censored auction evidence and market quantiles
 * inexpensive to integrate. Each entry in a support has equal mass. */
export function bidSupport(style: Style, c: PriceContext): number[] {
  const face = LETTER_VALUES[c.letter]!;
  let bids: number[];
  switch (style) {
    case 'face': bids = [face]; break;
    case 'premium': bids = [2 * face + 4]; break;
    case 'patient': bids = [face + 2]; break;
    case 'random': bids = Array.from({ length: 13 }, (_, i) => i); break;
    case 'idle': bids = [0]; break;
    case 'capital': {
      const share = 0.12 + 0.022 * face + (VOWELS[c.letter] ? 0.045 : 0);
      const center = c.balance * share * (c.remaining < 8 ? 1.4 : 1);
      bids = [0.65, 0.85, 1, 1.2, 1.5].map(r => r * center);
      break;
    }
    case 'completion': {
      const center = 2.8 + 1.8 * face + 0.95 * (c.gain ?? 4);
      const scale = c.mode === 'FirstPrice' ? 0.82 : 1;
      bids = [0.35, 0.65, 0.9, 1.3, 1.95].map(r => r * center * scale);
      break;
    }
  }
  return bids.map(b => Math.max(0, Math.min(c.balance, Math.round(b))));
}

type Rates = { less: number; equal: number; greater: number };
interface Mass { amount: number; p: number }
function rates(support: Mass[], price: number): Rates {
  const r = { less: 0, equal: 0, greater: 0 };
  for (const item of support) {
    if (item.amount < price) r.less += item.p;
    else if (item.amount === price) r.equal += item.p;
    else r.greater += item.p;
  }
  return r;
}
function uniform(amounts: number[]): Mass[] {
  return amounts.map(amount => ({ amount, p: 1 / amounts.length }));
}

/** P(observed winner and payment) under independent modeled bids. Integrate
 * the number of tied rivals to reproduce uniform tie breaking. At reserve 1,
 * a sole bidder is also compatible with the Vickrey observation. */
export function outcomeLikelihood(all: Rates[], mode: AuctionType, winner: number | null, paid: number): number {
  if (winner === null) return paid === 0 ? all.reduce((p, r) => p * r.equal, 1) : 0;
  if (paid <= 0 || !all[winner]) return 0;
  let allLess = 1, allAtMost = 1;
  let tieCounts = [1];
  all.forEach((r, i) => {
    if (i === winner) return;
    allLess *= r.less;
    allAtMost *= r.less + r.equal;
    const next = new Array<number>(tieCounts.length + 1).fill(0);
    tieCounts.forEach((p, k) => { next[k]! += p * r.less; next[k + 1]! += p * r.equal; });
    tieCounts = next;
  });
  const share = tieCounts.reduce((sum, p, k) => sum + p / (k + 1), 0);
  const w = all[winner]!;
  if (mode === 'FirstPrice') return w.equal * share;
  const excludeLower = paid > 1 ? allLess : 0;
  return Math.max(0, w.greater * (allAtMost - excludeLower) + w.equal * (share - excludeLower));
}

/** Infer external styles from public outcomes and own bids. Approximate
 * factorized posteriors allow a different rival style to be sampled per seat.
 * Names and registry strategy labels are never used as external evidence. */
export class MarketBelief {
  private weights: number[][];
  private processed = 0;
  constructor(players: readonly PlayerMeta[]) {
    this.weights = players.map(p => p.offline ? [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.93] : [...PRIOR]);
  }
  probabilities(seat: number): readonly number[] { return this.weights[seat]!; }
  sample(seat: number, rng: Rng): Style {
    let u = rng.next();
    for (let k = 0; k < STYLES.length; k++) {
      u -= this.weights[seat]![k]!;
      if (u <= 0) return STYLES[k]!;
    }
    return STYLES.at(-1)!;
  }
  private distribution(meta: PlayerMeta, i: number, c: PriceContext): Mass[] {
    if (meta.sim) return uniform([-2, -1, 0, 1, 2].map(j => simBid(meta.sim!, c.letter, c.balance, j)));
    const mass = new Map<number, number>();
    STYLES.forEach((style, k) => {
      const support = bidSupport(style, c);
      const weight = this.weights[i]![k]! / support.length;
      for (const amount of support) mass.set(amount, (mass.get(amount) ?? 0) + weight);
    });
    return [...mass].map(([amount, p]) => ({ amount, p }));
  }
  sync(o: Observation): void {
    while (this.processed < o.history.length) {
      this.update(o, o.history[this.processed]!, o.ownBids[this.processed]!);
      this.processed++;
    }
  }
  private update(o: Observation, h: RoundRecord, myBid: number): void {
    const { winner, paid } = h.outcome;
    const balances = [...h.balances];
    for (const w of h.words) balances[w.player]! += w.reward;
    const contexts = balances.map((balance, i) => ({ balance: o.players[i]!.sim ? h.balances[i]! : balance,
      letter: h.letter.charCodeAt(0) - 65, mode: o.auctionType, remaining: h.bagTotal }));
    const all = o.players.map((p, i) => rates(i === o.seat ? uniform([myBid]) : this.distribution(p, i, contexts[i]!), paid));
    const updated = this.weights.map(w => w.slice());
    o.players.forEach((p, i) => {
      if (i === o.seat || p.sim) return;
      const saved = all[i]!;
      const likelihoods = STYLES.map(style => {
        all[i] = rates(uniform(bidSupport(style, contexts[i]!)), paid);
        return outcomeLikelihood(all, o.auctionType, winner, paid);
      });
      all[i] = saved;
      const best = Math.max(...likelihoods);
      const words = h.words.filter(w => w.player === i);
      const weights = this.weights[i]!.map((weight, k) => {
        let evidence = best > 0 ? 0.08 + 0.92 * likelihoods[k]! / best : 1;
        if (words.length && STYLES[k] === 'idle') evidence *= 0.02;
        if (words.some(w => w.word.length < 7) && STYLES[k] === 'patient' && h.bagTotal > 1) evidence *= 0.4;
        return (0.98 * weight + 0.02 * PRIOR[k]!) * Math.pow(evidence, 0.65);
      });
      const sum = weights.reduce((a, b) => a + b, 0);
      updated[i] = weights.map(x => x / sum);
    });
    this.weights = updated;
  }
  /** Distribution of the maximum opposing bid, using only public inputs. */
  clearingPrices(o: Observation): Mass[] {
    const rivals = o.players.flatMap((p, i) => i === o.seat ? [] : [this.distribution(p, i, {
      letter: o.letter.charCodeAt(0) - 65, balance: o.balances[i]!, remaining: o.bagTotal, mode: o.auctionType,
    })]);
    const levels = [...new Set(rivals.flatMap(s => s.map(x => x.amount)))].sort((a, b) => a - b);
    let previous = 0;
    return levels.map(amount => {
      const cdf = rivals.reduce((prod, bids) => prod * bids.reduce((sum, b) => sum + (b.amount <= amount ? b.p : 0), 0), 1);
      const p = Math.max(0, cdf - previous);
      previous = cdf;
      return { amount, p };
    });
  }
}

export function quantile(prices: readonly Mass[], q: number): number {
  let sum = 0;
  for (const x of prices) { sum += x.p; if (sum + 1e-10 >= q) return x.amount; }
  return prices.at(-1)?.amount ?? 0;
}
