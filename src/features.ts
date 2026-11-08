import { sharedIndex } from '../upstream/bot-starter/src/game/dictionary.ts';
import { BAG_DISTRIBUTION, LETTER_VALUES, VOWELS, type Counts } from '../upstream/bot-starter/src/game/letters.ts';
import { publicEvidence } from './belief.ts';
import type { Observation } from './game.ts';

export const MAX_PLAYERS = 12;
const letters = [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];
const globalNames = ['balance-after-cash', 'score', 'leader-gap', 'players', 'remaining', 'progress', 'baseline-bid', 'cash-points',
  'rack-size', 'vowel-fraction', 'rack-greedy-value', 'completion-gain', 'letter-face-value', 'mean-price', 'letter-price',
  'recent-win-share', 'recent-spend', 'first-price', 'baseline-budget-share', 'unseen-size'];
const rivalNames = ['present', 'balance', 'score', 'known-rack-size', 'hidden-initial-tiles', 'recent-wins', 'mean-price', 'sim-style'];
export const FEATURE_NAMES = [
  ...letters.map(c => `rack-${c}`), ...letters.map(c => `letter-${c}`), ...letters.map(c => `unseen-${c}`), ...globalNames,
  ...Array.from({ length: MAX_PLAYERS - 1 }, (_, i) => rivalNames.map(k => `rival-${i + 1}-${k}`)).flat(),
];
export const FEATURE_VERSION = 1;

/** Encode only the observation and deterministic planner outputs. Values are
 * scaled and clipped; missing opponent slots have an explicit presence bit. */
export function encode(o: Observation, postCashRack: Counts, baseline: number, cashPoints: number, e = publicEvidence(o)): number[] {
  const me = o.seat, n = o.players.length;
  const idx = sharedIndex(), c = o.letter.charCodeAt(0) - 65;
  const size = postCashRack.reduce((a, b) => a + b, 0);
  const value = idx.greedyValue(postCashRack, 2);
  const withLetter = postCashRack.slice(); withLetter[c]!++;
  const gain = idx.greedyValue(withLetter, 2) - value;
  const prices = o.history.filter(h => h.outcome.winner !== null);
  const mean = (a: number[]) => a.length ? a.reduce((a, b) => a + b, 0) / a.length : 0;
  const recent = o.history.slice(-10), balance = o.balances[me]! + cashPoints;
  const total = 98 - o.rackSize * n;
  const bestRival = Math.max(...o.scores.filter((_, p) => p !== me));
  const x = [
    ...postCashRack.map(v => v / 5), ...letters.map((_, i) => Number(i === c)),
    ...e.unseen.map((v, i) => v / BAG_DISTRIBUTION[i]!),
    balance / 200, o.scores[me]! / 200, (o.scores[me]! + cashPoints - bestRival) / 100, n / MAX_PLAYERS,
    o.bagTotal / 98, o.round / total, baseline / 50, cashPoints / 50, size / 20,
    postCashRack.reduce((sum, v, i) => sum + (VOWELS[i] ? v : 0), 0) / Math.max(1, size),
    value / 100, gain / 50, LETTER_VALUES[c]! / 10,
    mean(prices.map(h => h.outcome.paid)) / 20,
    mean(prices.filter(h => h.letter === o.letter).map(h => h.outcome.paid)) / 20,
    recent.filter(h => h.outcome.winner === me).length / Math.max(1, recent.length),
    recent.reduce((sum, h) => sum + (h.outcome.winner === me ? h.outcome.paid : 0), 0) / 100,
    Number(o.auctionType === 'FirstPrice'), baseline / Math.max(1, balance), e.unseen.reduce((a, b) => a + b, 0) / 98,
  ];
  for (let offset = 1; offset < MAX_PLAYERS; offset++) {
    if (offset >= n) { x.push(...new Array(rivalNames.length).fill(0)); continue; }
    const p = (me + offset) % n, wins = prices.filter(h => h.outcome.winner === p);
    const sim = o.players[p]!.sim;
    x.push(1, o.balances[p]! / 200, o.scores[p]! / 200,
      e.minimumRack[p]!.reduce((a, b) => a + b, 0) / 20, e.hiddenDealSizes[p]! / Math.max(1, o.rackSize),
      recent.filter(h => h.outcome.winner === p).length / Math.max(1, recent.length),
      mean(wins.map(h => h.outcome.paid)) / 20, sim === 'Cheapskate' ? 1 / 3 : sim === 'ValueBidder' ? 2 / 3 : sim === 'Aggressive' ? 1 : 0);
  }
  if (x.length !== FEATURE_NAMES.length || x.some(v => !Number.isFinite(v))) throw new Error('Invalid feature vector.');
  return x.map(v => Math.max(-5, Math.min(5, v)));
}
