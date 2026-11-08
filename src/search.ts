import { baselineAction } from './actions.ts';
import { ScrabbleEnv, utility, type Objective } from './env.ts';

export interface SearchOptions { worlds: number; seed: number; opponentModels?: string[]; objective?: Objective }
/** Reference root Monte Carlo comparison, not a full MCTS implementation.
 * It proves that observation -> sampled world -> cloned candidate -> rollout
 * works. One sampled world is shared across root alternatives, reducing noise.
 * Every rollout policy sees only its own observation at each future auction.
 */
export function compareRootActions(env: ScrabbleEnv, options: SearchOptions) {
  if (!Number.isInteger(options.worlds) || options.worlds < 1) throw new Error('At least one sampled world is required.');
  const point = env.decision(), actions = point.actions.filter(a => a.legal);
  const values = new Map(actions.map(a => [a.id, [] as number[]]));
  const started = performance.now();
  for (let i = 0; i < options.worlds; i++) {
    const root = env.searchRoot((options.seed + i) >>> 0, options.opponentModels);
    for (const action of actions) {
      const branch = root.clone();
      let step = branch.step(action.id);
      while (!branch.done) step = branch.step(baselineAction(step.observation!.actions));
      values.get(action.id)!.push(utility(branch.result().seats.map(s => s.score), branch.seat, options.objective ?? env.objective));
    }
  }
  const rows = actions.map(a => {
    const samples = values.get(a.id)!, mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.length > 1 ? samples.reduce((sum, x) => sum + (x - mean) ** 2, 0) / (samples.length - 1) : 0;
    return { action: a.id, label: a.name, bid: a.bid, worlds: samples.length, value: mean, standardError: samples.length > 1 ? Math.sqrt(variance / samples.length) : null };
  });
  rows.sort((a, b) => b.value - a.value || a.bid - b.bid);
  return { kind: 'root-monte-carlo-demo', informationSet: point.informationSet, round: point.observation.round,
    objective: options.objective ?? env.objective, bestAction: rows[0]!.action, elapsedMs: performance.now() - started, actions: rows };
}
