import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';
import { test } from 'node:test';
import { evaluateTree, trainTree } from '../src/mcts/experiment.ts';
import { loadTree, type TrainingSettings } from '../src/mcts/storage.ts';
import { DEFAULT_TREE } from '../src/mcts/tree.ts';

test('resuming between frozen batches reproduces training and rejects an overlapping evaluation deal', async t => {
  const runs = fileURLToPath(new URL('../runs/', import.meta.url));
  mkdirSync(runs, { recursive: true });
  const root = mkdtempSync(join(runs, 'resume-test-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const settings: TrainingSettings = {
    seed: 815983, fields: ['module'], mode: 'first', batchGames: 1, epochs: 1, learningRate: 0.001,
    validationEvery: 0, validationGames: 0,
    tree: { ...DEFAULT_TREE, simulations: 12, particles: 2, width: 3, depth: 2, horizon: 2, every: 9, timeMs: 0 },
  };
  const one = join(root, 'one'), two = join(root, 'two');
  await trainTree({ out: one, games: 2, workers: 1, minutes: 3 }, settings);
  await trainTree({ out: two, games: 1, workers: 1, minutes: 3 }, settings);
  const model = join(two, 'latest.json');
  await trainTree({ out: two, games: 2, workers: 1, minutes: 3, resume: model }, settings);
  const reference = loadTree(join(one, 'latest.json')).saved, resumed = loadTree(model).saved;
  assert.deepEqual(resumed.progress, reference.progress);
  assert.deepEqual(resumed.network, reference.network);
  assert.ok(resumed.progress.positions > 0 && resumed.progress.deepSelections > 0);
  await assert.rejects(evaluateTree({ model, out: join(root, 'overlap.json'), workers: 1, seed: settings.seed + 1,
    seeds: 1, rotations: 1, fields: ['module'], mode: 'first' }), /overlap/);
});
