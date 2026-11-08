import { parentPort } from 'node:worker_threads';
import { ActorCritic, type NetworkData } from '../network.ts';
import type { MatchCase } from '../experiments.ts';
import { initialTreeNetwork, type TreeConfig } from './tree.ts';
import { play, type Played } from './matches.ts';
import type { Gate } from './gate.ts';

export interface Job { index: number; match: MatchCase }
export interface Batch {
  kind: 'train' | 'validation' | 'evaluate' | 'calibrate' | 'confirm'; jobs: Job[]; network: NetworkData;
  tree: TreeConfig; trainedGames: number; initialSeed: number; gate?: Gate;
}
export interface JobResult { index: number; match: MatchCase; candidate: Played; cold?: Played; alternative?: Played; expert?: Played }
export type WorkerReply = { kind: 'case'; result: JobResult } | { kind: 'done' } | { kind: 'error'; error: string };

parentPort?.on('message', (batch: Batch) => {
  try {
    const net = ActorCritic.fromJSON(batch.network);
    const cold = batch.kind === 'evaluate' ? initialTreeNetwork(batch.initialSeed) : null;
    for (const job of batch.jobs) {
      const { match } = job;
      const candidate = () => play(match, batch.tree, net, batch.trainedGames, batch.kind === 'train', batch.gate ?? 'full');
      let result: JobResult;
      if (cold || batch.kind === 'calibrate' || batch.kind === 'confirm') {
        // Counterbalance execution order so warm-up is distributed across variants.
        const comparison = cold ? () => play(match, batch.tree, cold, 0)
          : () => play(match, batch.tree, net, batch.trainedGames, false, batch.kind === 'calibrate' ? 'endgame' : 'full');
        const variants = [candidate, comparison, () => play(match, batch.tree)];
        const played = new Array<Played>(variants.length);
        for (let k = 0; k < variants.length; k++) {
          const i = (job.index + k) % variants.length;
          played[i] = variants[i]!();
        }
        result = { ...job, candidate: played[0]!, ...(cold ? { cold: played[1]! } : { alternative: played[1]! }), expert: played[2]! };
      } else result = { ...job, candidate: candidate() };
      parentPort!.postMessage({ kind: 'case', result } satisfies WorkerReply);
    }
    parentPort!.postMessage({ kind: 'done' } satisfies WorkerReply);
  } catch (error) {
    parentPort!.postMessage({ kind: 'error', error: error instanceof Error ? error.stack ?? error.message : String(error) } satisfies WorkerReply);
  }
});
