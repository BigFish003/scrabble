import { resolve } from 'node:path';
import { UPSTREAM_COMMIT, writeJSON } from '../checkpoint.ts';
import { summarize } from '../experiments.ts';
import { operational, pairedPlan, running, type EvalOptions } from './experiment.ts';
import { checkpointHash, loadGate, type Gate, type Scope } from './gate.ts';
import { MatchPool } from './pool.ts';
import { loadTree } from './storage.ts';
import type { JobResult } from './worker.ts';

export interface Credit { block: number; context: string; mode: 'endgame' | 'full'; winDelta: number; utilityDelta: number }
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
function estimate(rows: Credit[]) {
  const blocks = Map.groupBy(rows, r => r.block);
  const utility = [...blocks.values()].map(b => mean(b.map(r => r.utilityDelta)));
  const win = [...blocks.values()].map(b => mean(b.map(r => r.winDelta)));
  const n = utility.length, gain = mean(utility);
  const standardError = n > 1 ? Math.sqrt(utility.reduce((s, g) => s + (g - gain) ** 2, 0) / (n * (n - 1))) : 0;
  return { cases: rows.length, seedBlocks: n, winGain: mean(win), utilityGain: gain, standardError, discountedGain: gain - standardError };
}
/** Match-level, paired calibration. Favor the expert when evidence is sparse.
 * The one-standard-error discount is a selection rule, not a confidence interval. */
export function chooseScopes(credits: Credit[]) {
  const choices: Record<string, Scope> = {};
  const evidence: Record<string, Record<'full' | 'endgame', ReturnType<typeof estimate>>> = {};
  for (const [context, rows] of Map.groupBy(credits, r => r.context)) {
    const endgame = estimate(rows.filter(r => r.mode === 'endgame')), full = estimate(rows.filter(r => r.mode === 'full'));
    evidence[context] = { endgame, full };
    let best = 0.002;
    choices[context] = 'expert';
    for (const scope of ['endgame', 'full'] as const) {
      const e = evidence[context]![scope];
      if (e.seedBlocks >= 5 && e.winGain >= 0 && e.discountedGain > best) {
        choices[context] = scope;
        best = e.discountedGain;
      }
    }
  }
  return { choices, evidence };
}

function pairs(rows: JobResult[], side: 'candidate' | 'alternative', reference: 'alternative' | 'expert') {
  return rows.map(r => ({ ...r.match, candidate: r[side]!.metrics, baseline: r[reference]!.metrics }));
}
function calibrateFromMatches(rows: JobResult[]) {
  const credits: Credit[] = [];
  for (const r of rows) for (const [mode, bidder] of [['full', r.candidate], ['endgame', r.alternative!]] as const) {
    credits.push({ block: r.match.block, context: bidder.context, mode,
      winDelta: bidder.metrics.winCredit - r.expert!.metrics.winCredit, utilityDelta: bidder.utility - r.expert!.utility });
  }
  return chooseScopes(credits);
}
export async function runCalibration(options: EvalOptions, gatePath: string, confirm = false) {
  const { saved, net } = loadTree(options.model);
  const gate = confirm ? loadGate(gatePath, options.model) : undefined;
  const plan = pairedPlan(options, saved, gate ? [[gate.seed, gate.seedBlocks]] : []);
  const tree = { ...saved.search, timeMs: options.timeMs ?? saved.search.timeMs };
  const kind = confirm ? 'confirm' : 'calibrate';
  const pool = new MatchPool(options.workers), start = performance.now();
  console.log(JSON.stringify({ event: kind + '-start', pairedCases: plan.length, matches: plan.length * 3, seed: options.seed }));
  try {
    const rows = await pool.run(plan, { kind, network: net.toJSON(), tree, trainedGames: saved.progress.games,
      initialSeed: saved.settings.seed, gate }, running(kind, plan.length));
    const summary = { candidateVsExpert: summarize(pairs(rows, 'candidate', 'expert')),
      alternativeVsExpert: summarize(pairs(rows, 'alternative', 'expert')), candidateVsAlternative: summarize(pairs(rows, 'candidate', 'alternative')) };
    const machine = { candidate: operational(rows.map(r => r.candidate)), alternative: operational(rows.map(r => r.alternative!)),
      expert: operational(rows.map(r => r.expert!)) };
    const selection = confirm ? undefined : calibrateFromMatches(rows);
    const deployment: Gate = gate ?? { format: 'scrabble-search-gate-v1', modelHash: checkpointHash(options.model),
      seed: options.seed, seedBlocks: options.seeds, fields: options.fields, choices: selection!.choices };
    const cases = rows.map(r => ({ ...r.match, context: r.candidate.context, scope: r.candidate.scope,
      candidate: r.candidate.metrics, alternative: r.alternative!.metrics, expert: r.expert!.metrics,
      utility: [r.candidate.utility, r.alternative!.utility, r.expert!.utility] }));
    const report = { format: 'scrabble-search-' + kind + '-v1', upstream: UPSTREAM_COMMIT, model: resolve(options.model),
      checkpointGames: saved.progress.games, gate: deployment, tree, seed: options.seed, seedBlocks: options.seeds,
      rotations: options.rotations, fields: options.fields, mode: options.mode, workers: options.workers,
      pairedCases: rows.length, matches: 3 * rows.length, elapsedMinutes: (performance.now() - start) / 60000,
      candidate: confirm ? 'calibrated' : 'full', alternative: confirm ? 'full' : 'endgame', summary, operational: machine, selection, cases };
    writeJSON(options.out, report);
    if (!confirm) writeJSON(gatePath, deployment);
    console.log(JSON.stringify({ event: kind + '-finished', pairedCases: rows.length, minutes: report.elapsedMinutes, choices: deployment.choices,
      candidate: summary.candidateVsExpert.candidate, alternative: summary.alternativeVsExpert.candidate,
      expert: summary.candidateVsExpert.baseline, winVsExpert: summary.candidateVsExpert.pairedWinCredit,
      scoreVsExpert: summary.candidateVsExpert.pairedScore, winVsAlternative: summary.candidateVsAlternative.pairedWinCredit, out: options.out }));
    return report;
  } finally { await pool.close(); }
}
