import { ScrabbleEnv, type Objective } from './env.ts';
import { baselineAction } from './actions.ts';
import { Rng } from './rng.ts';
import { ActorCritic } from './network.ts';
import type { AuctionType } from '../upstream/bot-starter/src/game/letters.ts';
import type { MatchResult } from '../upstream/bot-starter/sim/engine.ts';

export const FIELDS: Record<string, string[]> = {
  base: ['starter', 'greedy', 'overbidder', 'hoarder', 'random'],
  module: ['brutus-sim', 'brutus-sim:brutus2', 'valor-sim', 'valor-sim:valor2', 'cheapo-sim'],
  mixed: ['final', 'botlike', 'brutus-sim', 'greedy', 'random'],
  self: ['final', 'final:f2', 'final:f3', 'final:f4', 'final:f5'],
  strong: ['shadow', 'late', 'outbid', 'jitter', 'botlike-hi'],
  heldout: ['sniper', 'allin', 'botlike-hi', 'denier', 'snowball'],
  duel: ['final'],
  practice: ['botlike', 'overbidder', 'hoarder', 'greedy', 'late'],
  league: ['final', 'shadow', 'sniper', 'late', 'denier'],
  'mixed-rivals': ['shadow', 'sniper', 'valor-sim', 'late', 'hoarder'],
  'mixed-modules': ['final', 'brutus-sim', 'cheapo-sim', 'denier', 'botlike-hi'],
};
export type Mode = 'vickrey' | 'first' | 'both';
export interface MatchCase { field: string; auctionType: AuctionType; seat: number; seed: number; block: number }
export interface EpisodeMetrics {
  score: number; rank: number; winCredit: number; topFinish: number; margin: number; spent: number;
  decisions: number; rejectedBids: number; rejectedWords: number; ms: number;
}
export type PairedRow = MatchCase & { candidate: EpisodeMetrics; baseline: EpisodeMetrics };

export function modes(mode: Mode): AuctionType[] {
  if (!['both', 'vickrey', 'first'].includes(mode)) throw new Error('Mode must be vickrey, first or both.');
  return mode === 'both' ? ['Vickrey', 'FirstPrice'] : [mode === 'first' ? 'FirstPrice' : 'Vickrey'];
}
export function checkFields(fields: string[]): void {
  if (!fields.length || fields.some(f => !FIELDS[f])) throw new Error(`Fields: ${Object.keys(FIELDS).join(', ')}`);
}
export function trainingCase(index: number, seed: number, fields: string[], mode: Mode): MatchCase {
  checkFields(fields);
  const field = fields[index % fields.length]!, types = modes(mode), opponents = FIELDS[field]!;
  // Cycle every field/mode/seat combination rather than coupling seat to field.
  const cycle = Math.floor(index / (fields.length * types.length));
  return { field, auctionType: types[Math.floor(index / fields.length) % types.length]!, seat: cycle % (opponents.length + 1), seed: (seed + index) >>> 0, block: index };
}
export function evaluationPlan(seed: number, seeds: number, fields: string[], mode: Mode): MatchCase[] {
  checkFields(fields);
  const plan: MatchCase[] = [];
  for (let block = 0; block < seeds; block++) for (const field of fields) for (const auctionType of modes(mode)) {
    for (let seat = 0; seat <= FIELDS[field]!.length; seat++) plan.push({ field, auctionType, seat, seed: (seed + block) >>> 0, block });
  }
  return plan;
}
export function makeEnv(c: MatchCase, objective: Objective = 'score'): ScrabbleEnv {
  return new ScrabbleEnv({ opponents: FIELDS[c.field]!, auctionType: c.auctionType, learnerSeat: c.seat, objective });
}
export function metrics(env: ScrabbleEnv, ms: number): EpisodeMetrics {
  return matchMetrics(env.result(), env.seat, ms);
}
export function matchMetrics(match: MatchResult, seat: number, ms: number): EpisodeMetrics {
  const mine = match.seats[seat]!, best = Math.max(...match.seats.map(s => s.score));
  return { score: mine.score, rank: mine.rank, winCredit: mine.score === best ? 1 / match.seats.filter(s => s.score === best).length : 0,
    topFinish: Number(mine.rank === 1), margin: mine.score - Math.max(...match.seats.filter((_, i) => i !== seat).map(s => s.score)),
    spent: mine.spent, decisions: match.auctions, rejectedBids: mine.rejectedBids, rejectedWords: mine.rejectedWords, ms };
}
export function runEpisode(c: MatchCase, network?: ActorCritic, stochastic = false, policySeed = 123): EpisodeMetrics {
  const env = makeEnv(c), start = performance.now(), rng = new Rng(policySeed);
  let point = env.reset(c.seed);
  while (!env.done) {
    const action = network ? network.act(point.features, point.actions.map(a => a.legal), rng, !stochastic) : baselineAction(point.actions);
    point = env.step(action).observation!;
  }
  return metrics(env, performance.now() - start);
}
export function evaluateCases(plan: MatchCase[], network: ActorCritic, progress?: (completed: number, total: number) => void,
  baselineCache?: EpisodeMetrics[], stochastic = false): PairedRow[] {
  return plan.map((c, i) => {
    const baseline = baselineCache?.[i] ?? runEpisode(c);
    const candidate = runEpisode(c, network, stochastic, (c.seed ^ (c.seat + 1) * 7919) >>> 0);
    progress?.(i + 1, plan.length);
    return { ...c, candidate, baseline };
  });
}
const mean = (a: number[]) => a.reduce((a, b) => a + b, 0) / Math.max(1, a.length);
function summarizeSide(rows: PairedRow[], side: 'candidate' | 'baseline') {
  return { games: rows.length, winCredit: mean(rows.map(r => r[side].winCredit)), topFinish: mean(rows.map(r => r[side].topFinish)),
    meanRank: mean(rows.map(r => r[side].rank)), meanScore: mean(rows.map(r => r[side].score)), meanMargin: mean(rows.map(r => r[side].margin)),
    rejectedBids: rows.reduce((sum, r) => sum + r[side].rejectedBids, 0), rejectedWords: rows.reduce((sum, r) => sum + r[side].rejectedWords, 0),
    decisionsPerSecond: rows.reduce((sum, r) => sum + r[side].decisions, 0) / Math.max(0.001, rows.reduce((sum, r) => sum + r[side].ms, 0) / 1000) };
}
function interval(rows: PairedRow[], metric: 'winCredit' | 'score'): { seedBlocks: number; mean: number; interval95: [number, number] | null } {
  const groups = new Map<number, number[]>();
  for (const row of rows) {
    const values = groups.get(row.block) ?? [];
    values.push(row.candidate[metric] - row.baseline[metric]); groups.set(row.block, values);
  }
  const blocks = [...groups.values()].map(mean), estimate = mean(blocks);
  // Seats/modes/fields sharing a seed are correlated. Bootstrap whole seed
  // blocks, NOT individual games. Fewer than five blocks is only a smoke run.
  if (blocks.length < 5) return { seedBlocks: blocks.length, mean: estimate, interval95: null };
  const rng = new Rng(20261107), samples: number[] = [];
  for (let b = 0; b < 2000; b++) {
    let sum = 0;
    for (let k = 0; k < blocks.length; k++) sum += blocks[rng.int(blocks.length)]!;
    samples.push(sum / blocks.length);
  }
  samples.sort((a, b) => a - b);
  return { seedBlocks: blocks.length, mean: estimate, interval95: [samples[50]!, samples[1949]!] };
}
export function summarize(rows: PairedRow[]) {
  const groups: Record<string, { candidate: ReturnType<typeof summarizeSide>; baseline: ReturnType<typeof summarizeSide> }> = {};
  for (const row of rows) {
    const key = `${row.field}/${row.auctionType}`;
    if (groups[key]) continue;
    const matches = rows.filter(r => `${r.field}/${r.auctionType}` === key);
    groups[key] = { candidate: summarizeSide(matches, 'candidate'), baseline: summarizeSide(matches, 'baseline') };
  }
  return { candidate: summarizeSide(rows, 'candidate'), baseline: summarizeSide(rows, 'baseline'),
    pairedWinCredit: interval(rows, 'winCredit'), pairedScore: interval(rows, 'score'), groups };
}
