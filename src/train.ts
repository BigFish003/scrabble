import { appendFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { setImmediate } from 'node:timers/promises';
import { ActorCritic, type Example } from './network.ts';
import { Rng } from './rng.ts';
import { saveCheckpoint, readCheckpoint, type TrainingState } from './checkpoint.ts';
import { checkFields, evaluateCases, makeEnv, metrics, runEpisode, summarize, trainingCase, type Mode } from './experiments.ts';
import type { Objective } from './env.ts';

export interface TrainOptions {
  out: string; games: number; batchSize: number; seed: number; fields: string[]; mode: Mode; objective: Objective;
  learningRate: number; entropy: number; evalEvery: number; evalGames: number; maxMinutes: number;
  resume?: string; init?: string;
}
/** One policy update per batch of complete games; no replay of stale policy
 * gradients and no clipping-ratio claim. Gamma=1 preserves the score/rank
 * objective. A checkpoint is written after every complete update. */
export async function train(options: TrainOptions): Promise<void> {
  checkFields(options.fields);
  mkdirSync(options.out, { recursive: true });
  let network = new ActorCritic(options.seed);
  let state: TrainingState = { games: 0, seed: options.seed, fields: options.fields, mode: options.mode, objective: options.objective,
    policyRng: new Rng(options.seed ^ 0x61c88647).snapshot(), batchSize: options.batchSize, learningRate: options.learningRate,
    entropy: options.entropy, bestWinCredit: null, validationGames: options.evalGames };
  if (options.resume) {
    const loaded = readCheckpoint(options.resume); network = loaded.network; state = loaded.checkpoint.training;
    // Resume uses the stored objective, mixture and optimizer settings. The
    // caller may change the total game/time budget and validation cadence.
    Object.assign(options, { seed: state.seed, fields: state.fields, mode: state.mode, objective: state.objective,
      batchSize: state.batchSize, learningRate: state.learningRate, entropy: state.entropy, evalGames: state.validationGames });
  } else if (options.init) {
    const data = readCheckpoint(options.init).checkpoint.network;
    data.adam.step = 0;
    for (const values of [...Object.values(data.adam.m), ...Object.values(data.adam.v)]) values.fill(0);
    network = ActorCritic.fromJSON(data);
  }
  const rng = Rng.restore(state.policyRng), started = performance.now();
  let lastUpdate = performance.now(), lastProgress = performance.now();
  let stopRequested = false;
  const stop = () => { stopRequested = true; console.log('Finishing the current game, updating the collected batch, then saving.'); };
  process.on('SIGINT', stop);
  const validationPlan = Array.from({ length: options.evalGames }, (_, i) => trainingCase(i, (options.seed + 1_000_000) >>> 0, options.fields, options.mode));
  let validationBaseline: ReturnType<typeof runEpisode>[] | undefined;
  let nextEval = options.evalEvery > 0 ? (Math.floor(state.games / options.evalEvery) + 1) * options.evalEvery : Infinity;
  const timedOut = () => (performance.now() - started) / 60000 >= options.maxMinutes;
  if (!options.resume) saveCheckpoint(resolve(options.out, 'initial.json'), network, state);
  console.log(JSON.stringify({ event: 'start', algorithm: 'episodic-policy-gradient-with-value-baseline',
    gamesAlreadyCompleted: state.games, targetGames: options.games, fields: options.fields, mode: options.mode, objective: options.objective, out: options.out }));
  try {
    while (state.games < options.games && !stopRequested && !timedOut()) {
      const samples: Example[] = [], episodeMetrics = [];
      const batchEnd = Math.min(options.games, state.games + options.batchSize);
      while (state.games < batchEnd && !stopRequested && !timedOut()) {
        const c = trainingCase(state.games, options.seed, options.fields, options.mode), env = makeEnv(c, options.objective), t0 = performance.now();
        let point = env.reset(c.seed);
        const trajectory: { features: number[]; mask: boolean[]; action: number; value: number; reward: number }[] = [];
        while (!env.done) {
          const mask = point.actions.map(a => a.legal), prediction = network.predict(point.features, mask);
          const action = network.act(point.features, mask, rng), step = env.step(action);
          trajectory.push({ features: point.features, mask, action, value: prediction.value, reward: step.reward });
          point = step.observation!;
        }
        let target = 0;
        for (let i = trajectory.length - 1; i >= 0; i--) {
          const t = trajectory[i]!; target += t.reward;
          samples.push({ features: t.features, mask: t.mask, action: t.action, target, advantage: target - t.value });
        }
        state.games++; episodeMetrics.push(metrics(env, performance.now() - t0));
        await setImmediate(); // Let Ctrl+C request a graceful save between games.
        if (performance.now() - lastProgress > 20000) {
          console.log(JSON.stringify({ event: 'progress', games: state.games, elapsedMinutes: (performance.now() - started) / 60000 }));
          lastProgress = performance.now();
        }
      }
      if (!samples.length) break;
      const meanAdv = samples.reduce((sum, e) => sum + e.advantage, 0) / samples.length;
      const stdAdv = Math.sqrt(samples.reduce((sum, e) => sum + (e.advantage - meanAdv) ** 2, 0) / samples.length);
      for (const e of samples) e.advantage = (e.advantage - meanAdv) / Math.max(1e-6, stdAdv);
      const update = network.learn(samples, { learningRate: options.learningRate, entropy: options.entropy, valueWeight: 0.5, maxGradientNorm: 0.5 });
      state.policyRng = rng.snapshot();
      const log = { event: 'update', games: state.games, decisions: samples.length, ...update,
        meanScore: episodeMetrics.reduce((sum, e) => sum + e.score, 0) / episodeMetrics.length,
        winCredit: episodeMetrics.reduce((sum, e) => sum + e.winCredit, 0) / episodeMetrics.length,
        decisionsPerSecond: samples.length / ((performance.now() - lastUpdate) / 1000),
        elapsedMinutes: (performance.now() - started) / 60000 };
      appendFileSync(resolve(options.out, 'training.jsonl'), JSON.stringify(log) + '\n'); console.log(JSON.stringify(log));
      saveCheckpoint(resolve(options.out, 'latest.json'), network, state);
      if (state.games >= nextEval && !stopRequested && !timedOut() && validationPlan.length) {
        console.log(JSON.stringify({ event: 'validation-start', games: state.games, matches: validationPlan.length }));
        if (!validationBaseline) {
          validationBaseline = [];
          for (const c of validationPlan) {
            if (stopRequested || timedOut()) break;
            validationBaseline.push(runEpisode(c)); await setImmediate();
          }
        }
        const rows = [];
        if (validationBaseline.length === validationPlan.length) for (let i = 0; i < validationPlan.length; i++) {
          if (stopRequested || timedOut()) break;
          rows.push(...evaluateCases([validationPlan[i]!], network, undefined, [validationBaseline[i]!]));
          await setImmediate();
          if (performance.now() - lastProgress > 20000) { console.log(JSON.stringify({ event: 'validation-progress', done: i + 1, total: validationPlan.length })); lastProgress = performance.now(); }
        }
        if (rows.length !== validationPlan.length) { console.log(JSON.stringify({ event: 'validation-skipped', reason: 'budget-or-interrupt', completed: rows.length })); break; }
        const report = summarize(rows);
        const win = report.candidate.winCredit;
        const accepted = state.bestWinCredit === null || win > state.bestWinCredit;
        if (accepted) { state.bestWinCredit = win; saveCheckpoint(resolve(options.out, 'best.json'), network, state, report); }
        saveCheckpoint(resolve(options.out, 'latest.json'), network, state, report);
        const entry = { event: 'validation', games: state.games, accepted, report };
        appendFileSync(resolve(options.out, 'training.jsonl'), JSON.stringify(entry) + '\n'); console.log(JSON.stringify(entry));
        nextEval = (Math.floor(state.games / options.evalEvery) + 1) * options.evalEvery;
      }
      lastUpdate = performance.now();
    }
    // Even a zero-minute run produces a loadable model.
    saveCheckpoint(resolve(options.out, 'latest.json'), network, state);
    console.log(JSON.stringify({ event: 'finished', games: state.games, elapsedMinutes: (performance.now() - started) / 60000,
      reason: stopRequested ? 'interrupted-after-game' : timedOut() ? 'time-budget' : 'game-budget', checkpoint: resolve(options.out, 'latest.json') }));
  } finally { process.off('SIGINT', stop); }
}
