import { readFileSync } from 'node:fs';
import { MCTS_ACTION_NAMES } from '../actions.ts';
import { UPSTREAM_COMMIT, writeJSON } from '../checkpoint.ts';
import { ActorCritic, type NetworkData } from '../network.ts';
import type { Mode } from '../experiments.ts';
import type { RngState } from '../rng.ts';
import { checkTree, type TreeConfig } from './tree.ts';

export interface TrainingSettings {
  seed: number; fields: string[]; mode: Mode; batchGames: number; epochs: number; learningRate: number;
  validationEvery: number; validationGames: number; tree: TreeConfig;
}
export interface TrainingProgress {
  games: number; positions: number; updates: number; simulations: number; deepSelections: number; shuffle: RngState;
  bestValidation: number | null;
}
export interface TreeCheckpoint {
  format: 'scrabble-ismcts-v1'; upstream: string; objective: 'standing-v1'; search: TreeConfig;
  settings: TrainingSettings; progress: TrainingProgress; network: NetworkData;
}
export function saveTree(path: string, net: ActorCritic, settings: TrainingSettings, progress: TrainingProgress): void {
  const search = { ...settings.tree, timeMs: settings.tree.timeMs || 150 };
  writeJSON(path, { format: 'scrabble-ismcts-v1', upstream: UPSTREAM_COMMIT, objective: 'standing-v1',
    search, settings, progress, network: net.toJSON() } satisfies TreeCheckpoint);
}
export function loadTree(path: string): { saved: TreeCheckpoint; net: ActorCritic } {
  const saved = JSON.parse(readFileSync(path, 'utf8')) as TreeCheckpoint;
  if (saved.format !== 'scrabble-ismcts-v1' || saved.upstream !== UPSTREAM_COMMIT || saved.objective !== 'standing-v1' ||
    JSON.stringify(saved.network.actionNames) !== JSON.stringify(MCTS_ACTION_NAMES)) throw new Error('Incompatible MCTS checkpoint.');
  checkTree(saved.search);
  checkTree(saved.settings.tree);
  return { saved, net: ActorCritic.fromJSON(saved.network) };
}
