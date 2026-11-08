import '../../upstream/bot-starter/sim/strategies.ts';
import { runMatch } from '../../upstream/bot-starter/sim/engine.ts';
import { makeSeat } from '../../upstream/bot-starter/sim/registry.ts';
import { createAgent } from '../../upstream/bot-starter/src/strategy.ts';
import { FIELDS, matchMetrics, type EpisodeMetrics, type MatchCase } from '../experiments.ts';
import type { ActorCritic, SearchExample } from '../network.ts';
import { MctsPolicy, emptyCounters, type SearchCounters } from './policy.ts';
import { terminalValue } from './rollout.ts';
import type { TreeConfig } from './tree.ts';
import { marketContext, type Gate, type Scope } from './gate.ts';

export interface Played {
  context: string; scope: Scope;
  metrics: EpisodeMetrics; utility: number; counters: SearchCounters; latency: number[]; examples: SearchExample[];
}
export function play(c: MatchCase, config: TreeConfig, network?: ActorCritic, trainedGames = 0, collect = false, strategy: Scope | Gate = 'full'): Played {
  const seats = FIELDS[c.field]!.map(name => makeSeat(name));
  const sims = seats.map(s => s.sim ?? null);
  sims.splice(c.seat, 0, null);
  const context = marketContext(c.auctionType, sims, c.seat);
  const scope = network ? (typeof strategy === 'string' ? strategy : strategy.choices[context] ?? 'expert') : 'expert';
  const policy = network && scope !== 'expert' ? new MctsPolicy(network, config, trainedGames, collect, scope === 'endgame') : undefined;
  // Identical public labels and identical per-seat seeds in all paired variants.
  seats.splice(c.seat, 0, { name: 'learner', make: () => policy ?? createAgent('learner') });
  const start = performance.now();
  const match = runMatch(seats, { seed: c.seed, auctionType: c.auctionType, timings: true });
  const metrics = matchMetrics(match, c.seat, performance.now() - start);
  const target = terminalValue(match.seats.map(s => s.score), c.seat);
  return { metrics, context, scope, utility: target, counters: policy?.counters ?? emptyCounters(), latency: match.seats[c.seat]!.decisionMs,
    examples: (policy?.examples ?? []).map(e => ({ ...e, target })) };
}
