import { mkdirSync, readFileSync, writeFileSync, renameSync } from 'node:fs';
import { dirname } from 'node:path';
import { ActorCritic, type NetworkData } from './network.ts';
import type { Objective } from './env.ts';
import type { Mode } from './experiments.ts';
import type { RngState } from './rng.ts';

export const UPSTREAM_COMMIT = '6d61af7daf2f32b4048740de60d75ab2eb5c9d81';
export interface TrainingState {
  games: number; seed: number; fields: string[]; mode: Mode; objective: Objective; policyRng: RngState;
  batchSize: number; learningRate: number; entropy: number; bestWinCredit: number | null; validationGames: number;
}
export interface Checkpoint {
  format: 'scrabble-rl-v1'; upstreamCommit: string; createdAt: string; network: NetworkData;
  training: TrainingState; validation?: unknown;
}
export function writeJSON(path: string, value: unknown): void {
  mkdirSync(dirname(path), { recursive: true });
  const temp = `${path}.tmp`;
  writeFileSync(temp, JSON.stringify(value, null, 2) + '\n');
  renameSync(temp, path);
}
export function saveCheckpoint(path: string, network: ActorCritic, training: TrainingState, validation?: unknown): void {
  writeJSON(path, { format: 'scrabble-rl-v1', upstreamCommit: UPSTREAM_COMMIT, createdAt: new Date().toISOString(),
    network: network.toJSON(), training, ...(validation ? { validation } : {}) } satisfies Checkpoint);
}
export function readCheckpoint(path: string): { checkpoint: Checkpoint; network: ActorCritic } {
  const checkpoint = JSON.parse(readFileSync(path, 'utf8')) as Checkpoint;
  if (checkpoint.format !== 'scrabble-rl-v1' || checkpoint.upstreamCommit !== UPSTREAM_COMMIT) throw new Error('Unsupported checkpoint or upstream revision.');
  return { checkpoint, network: ActorCritic.fromJSON(checkpoint.network) };
}
