import { Worker } from 'node:worker_threads';
import type { MatchCase } from '../experiments.ts';
import type { Batch, JobResult, WorkerReply } from './worker.ts';

/** One dictionary per worker. All cases in a training batch use frozen
 * weights. Results are restored to case order before seeded minibatch updates. */
export class MatchPool {
  private workers: Worker[];
  constructor(count: number) {
    this.workers = Array.from({ length: count }, () => new Worker(new URL('./worker.ts', import.meta.url)));
  }
  async run(cases: MatchCase[], settings: Omit<Batch, 'jobs'>, report?: (n: number) => void): Promise<JobResult[]> {
    const results = new Array<JobResult>(cases.length);
    let completed = 0;
    const chunks = this.workers.map(() => [] as Batch['jobs']);
    cases.forEach((match, index) => {
      // Rotate each group so repeated fields such as expensive duels are
      // distributed across workers. Training still consumes results in case order.
      const group = Math.floor(index / chunks.length);
      chunks[(index + group) % chunks.length]!.push({ index, match });
    });
    await Promise.all(this.workers.map((w, i) => new Promise<void>((resolve, reject) => {
      const clean = () => { w.off('message', message); w.off('error', failure); w.off('exit', exit); };
      const failure = (error: Error) => { clean(); reject(error); };
      const exit = (code: number) => failure(new Error('Match worker exited during batch with code ' + code));
      const message = (reply: WorkerReply) => {
        if (reply.kind === 'error') { failure(new Error(reply.error)); return; }
        if (reply.kind === 'done') { clean(); resolve(); return; }
        results[reply.result.index] = reply.result;
        report?.(++completed);
      };
      w.on('message', message); w.once('error', failure); w.once('exit', exit);
      w.postMessage({ ...settings, jobs: chunks[i]! } satisfies Batch);
    })));
    return results;
  }
  async close(): Promise<void> { await Promise.all(this.workers.map(w => w.terminate())); }
}
