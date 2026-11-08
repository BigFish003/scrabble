import { appendFileSync, existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { UPSTREAM_COMMIT, writeJSON } from '../checkpoint.ts';
import { FIELDS, evaluationPlan, summarize, trainingCase, type Mode, type PairedRow } from '../experiments.ts';
import { Rng } from '../rng.ts';
import { MatchPool } from './pool.ts';
import { initialTreeNetwork, type TreeConfig } from './tree.ts';
import { loadTree, saveTree, type TrainingProgress, type TrainingSettings, type TreeCheckpoint } from './storage.ts';
import type { Played } from './matches.ts';

export function running(label: string, count: number) {
  let last = performance.now();
  return (n: number) => {
    if (performance.now() - last >= 15000) {
      console.log(JSON.stringify({ event: label, completed: n, count }));
      last = performance.now();
    }
  };
}
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
export function operational(rows: Played[]) {
  const times = rows.flatMap(r => r.latency).sort((a, b) => a - b);
  const at = (p: number) => times[Math.max(0, Math.ceil(p * times.length) - 1)] ?? 0;
  return { decisions: times.length, p50Ms: at(0.5), p95Ms: at(0.95), p99Ms: at(0.99), maxMs: at(1),
    over250Ms: times.filter(t => t > 250).length, searched: rows.reduce((s, r) => s + r.counters.searched, 0),
    simulations: rows.reduce((s, r) => s + r.counters.simulations, 0), overrides: rows.reduce((s, r) => s + r.counters.overrides, 0),
    searchOverrides: rows.reduce((s, r) => s + r.counters.searchOverrides, 0), endgames: rows.reduce((s, r) => s + r.counters.endgames, 0),
    timeStops: rows.reduce((s, r) => s + r.counters.timeStops, 0), deepSelections: rows.reduce((s, r) => s + r.counters.deepSelections, 0),
    maxTreeDepth: Math.max(0, ...rows.map(r => r.counters.maxDepth)) };
}

export async function trainTree(options: { out: string; games: number; workers: number; minutes: number; resume?: string }, requested: TrainingSettings) {
  mkdirSync(options.out, { recursive: true });
  const latest = resolve(options.out, 'latest.json');
  if (!options.resume && existsSync(latest)) throw new Error('This run has a checkpoint. Resume it or choose a new output folder.');
  const loaded = options.resume ? loadTree(options.resume) : undefined;
  const settings = loaded?.saved.settings ?? requested;
  const net = loaded?.net ?? initialTreeNetwork(settings.seed);
  const state: TrainingProgress = loaded?.saved.progress ?? { games: 0, positions: 0, updates: 0, simulations: 0,
    deepSelections: 0, shuffle: new Rng(settings.seed ^ 0xa163ce19).snapshot(), bestValidation: null };
  if (options.games < state.games) throw new Error('The requested total precedes the saved game count.');
  const rng = Rng.restore(state.shuffle);
  const pool = new MatchPool(options.workers), start = performance.now();
  const log = (entry: object) => {
    const line = JSON.stringify(entry);
    appendFileSync(resolve(options.out, 'training.jsonl'), line + '\n'); console.log(line);
  };
  let interrupted = false;
  const stop = () => { interrupted = true; console.log('Finishing the current batch and saving the checkpoint.'); };
  process.on('SIGINT', stop);
  const expired = () => performance.now() - start >= options.minutes * 60000;
  try {
    if (!loaded) saveTree(resolve(options.out, 'initial.json'), net, settings, state);
    log({ event: 'train-start', completed: state.games, target: options.games, workers: options.workers, settings });
    let nextValidation = settings.validationEvery ? (Math.floor(state.games / settings.validationEvery) + 1) * settings.validationEvery : Infinity;
    while (!interrupted && !expired() && state.games < options.games) {
      const batchStart = performance.now();
      const n = Math.min(settings.batchGames, options.games - state.games);
      const cases = Array.from({ length: n }, (_, i) => trainingCase(state.games + i, settings.seed, settings.fields, settings.mode));
      const outcomes = await pool.run(cases, { kind: 'train', network: net.toJSON(), tree: settings.tree,
        trainedGames: state.games, initialSeed: settings.seed }, running('collect', n));
      const episodes = outcomes.map(r => r.candidate);
      const examples = episodes.flatMap(e => e.examples);
      if (!examples.length) throw new Error('No search targets were generated. Increase the simulation or time budget.');
      const losses: number[] = [];
      for (let epoch = 0; epoch < settings.epochs; epoch++) {
        rng.shuffle(examples);
        for (let offset = 0; offset < examples.length; offset += 64) {
          const update = net.learn(examples.slice(offset, offset + 64), { learningRate: settings.learningRate,
            entropy: 0.002, valueWeight: 0.6, maxGradientNorm: 1 });
          losses.push(update.loss); state.updates++;
        }
      }
      const ops = operational(episodes);
      state.games += n; state.positions += examples.length; state.shuffle = rng.snapshot();
      state.simulations += ops.simulations; state.deepSelections += ops.deepSelections;
      saveTree(latest, net, settings, state);
      log({ event: 'train-update', games: state.games, totalPositions: state.positions, updates: state.updates,
        totalSimulations: state.simulations, batchLoss: mean(losses), meanScore: mean(episodes.map(e => e.metrics.score)),
        winCredit: mean(episodes.map(e => e.metrics.winCredit)), batchSeconds: (performance.now() - batchStart) / 1000, operational: ops });
      if (!interrupted && !expired() && settings.validationGames && (state.games >= nextValidation || (state.games === options.games && settings.validationEvery))) {
        const plan = Array.from({ length: settings.validationGames }, (_, i) => {
          const c = trainingCase(i, settings.seed + 1000000, settings.fields, settings.mode);
          c.seat = i % (FIELDS[c.field]!.length + 1);
          return c;
        });
        const validation = await pool.run(plan, { kind: 'validation', network: net.toJSON(), tree: settings.tree,
          trainedGames: state.games, initialSeed: settings.seed }, running('validate', plan.length));
        const utility = mean(validation.map(r => r.candidate.utility));
        const accepted = state.bestValidation === null || utility > state.bestValidation;
        if (accepted) {
          state.bestValidation = utility;
          saveTree(resolve(options.out, 'best.json'), net, settings, state);
        }
        saveTree(latest, net, settings, state);
        log({ event: 'validation', games: state.games, utility, selected: accepted, cases: plan.length,
          winCredit: mean(validation.map(r => r.candidate.metrics.winCredit)), meanScore: mean(validation.map(r => r.candidate.metrics.score)) });
        nextValidation = (Math.floor(state.games / settings.validationEvery) + 1) * settings.validationEvery;
      }
    }
    saveTree(latest, net, settings, state);
    log({ event: 'finished', games: state.games, positions: state.positions, simulations: state.simulations, updates: state.updates,
      minutes: (performance.now() - start) / 60000, checkpoint: latest, reason: interrupted ? 'interrupt' : state.games === options.games ? 'game-limit' : 'time-limit' });
  } finally {
    process.off('SIGINT', stop);
    await pool.close();
  }
}

export interface EvalOptions { model: string; out: string; workers: number; seed: number; seeds: number;
  rotations: number; fields: string[]; mode: Mode; timeMs?: number }
export function pairedPlan(options: EvalOptions, saved: TreeCheckpoint, reserved: [number, number][] = []) {
  const plan = evaluationPlan(options.seed, options.seeds, options.fields, options.mode).filter(c => {
    const n = FIELDS[c.field]!.length + 1;
    return !options.rotations || (c.seat - c.block % n + n) % n < options.rotations;
  });
  const overlaps = (seed: number, start: number, count: number) => ((seed - start) >>> 0) < count;
  const ranges: [number, number][] = [[saved.settings.seed, saved.progress.games],
    [saved.settings.seed + 1000000, saved.settings.validationGames], ...reserved];
  if (plan.some(c => ranges.some(([seed, length]) => overlaps(c.seed, seed, length)))) throw new Error('Requested seeds overlap training or selection data.');
  return plan;
}
export async function evaluateTree(options: EvalOptions) {
  const { saved, net } = loadTree(options.model);
  const tree: TreeConfig = { ...saved.search, timeMs: options.timeMs ?? saved.search.timeMs };
  const plan = pairedPlan(options, saved);
  const pool = new MatchPool(options.workers);
  const start = performance.now();
  try {
    console.log(JSON.stringify({ event: 'evaluation-start', pairedCases: plan.length, matches: plan.length * 3, seed: options.seed, workers: options.workers, tree }));
    const outcomes = await pool.run(plan, { kind: 'evaluate', network: net.toJSON(), tree, trainedGames: saved.progress.games,
      initialSeed: saved.settings.seed }, running('evaluate', plan.length));
    const pairs = (a: 'candidate' | 'cold', b: 'cold' | 'expert'): PairedRow[] => outcomes.map(r => ({
      ...r.match, candidate: r[a]!.metrics, baseline: r[b]!.metrics,
    }));
    const trainedVsExpert = summarize(pairs('candidate', 'expert')), coldVsExpert = summarize(pairs('cold', 'expert'));
    const trainedVsCold = summarize(pairs('candidate', 'cold'));
    const ops = { trained: operational(outcomes.map(r => r.candidate)), cold: operational(outcomes.map(r => r.cold!)),
      expert: operational(outcomes.map(r => r.expert!)) };
    const report = { format: 'scrabble-ismcts-evaluation-v1', upstream: UPSTREAM_COMMIT, model: resolve(options.model),
      checkpointGames: saved.progress.games, tree, seed: options.seed, seedBlocks: options.seeds, rotations: options.rotations,
      fields: options.fields, mode: options.mode, workers: options.workers, pairedCases: outcomes.length, matches: outcomes.length * 3,
      elapsedMinutes: (performance.now() - start) / 60000, trainedVsExpert, coldVsExpert, trainedVsCold, operational: ops, cases: outcomes };
    writeJSON(options.out, report);
    console.log(JSON.stringify({ event: 'evaluation-finished', pairedCases: outcomes.length, elapsedMinutes: report.elapsedMinutes,
      trained: trainedVsExpert.candidate, cold: coldVsExpert.candidate, expert: trainedVsExpert.baseline,
      pairedWinVsExpert: trainedVsExpert.pairedWinCredit, pairedWinVsCold: trainedVsCold.pairedWinCredit,
      pairedScoreVsExpert: trainedVsExpert.pairedScore, operational: ops, report: options.out }));
    return report;
  } finally { await pool.close(); }
}
