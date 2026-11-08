import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { createAgent } from '../../upstream/bot-starter/src/strategy.ts';
import type { AuctionOutcome, AuctionView, Decision, MatchStart, Policy, SimStrategy, WordPlayed } from '../../upstream/bot-starter/src/game/types.ts';
import type { AuctionType } from '../../upstream/bot-starter/src/game/letters.ts';
import type { ActorCritic } from '../network.ts';
import { MctsPolicy } from './policy.ts';
import type { TreeConfig } from './tree.ts';

export type Scope = 'expert' | 'endgame' | 'full';
export interface Gate {
  format: 'scrabble-search-gate-v1'; modelHash: string; seed: number; seedBlocks: number;
  fields: string[]; choices: Record<string, Scope>;
}
export function marketContext(mode: AuctionType, sims: readonly (SimStrategy | null)[], seat: number): string {
  const count = sims.filter((s, i) => i !== seat && s !== null).length;
  const table = count === 0 ? 'external' : count === sims.length - 1 ? 'module' : 'mixed';
  return mode + '/' + sims.length + '/' + table;
}
export function checkpointHash(model: string): string {
  return createHash('sha256').update(readFileSync(model)).digest('hex');
}
export function loadGate(path: string, model: string): Gate {
  const gate = JSON.parse(readFileSync(path, 'utf8')) as Gate;
  if (gate.format !== 'scrabble-search-gate-v1' || gate.modelHash !== checkpointHash(model) ||
    !Number.isSafeInteger(gate.seed) || gate.seed < 0 || gate.seed > 0xffffffff ||
    !Number.isSafeInteger(gate.seedBlocks) || gate.seedBlocks < 1 ||
    !gate.choices || !Object.values(gate.choices).every(v => ['expert', 'endgame', 'full'].includes(v)))
    throw new Error('Search gate and checkpoint do not match.');
  return gate;
}
export function forScope(scope: Scope, net: ActorCritic, tree: TreeConfig, games: number): Policy {
  return scope === 'expert' ? createAgent('mcts') : new MctsPolicy(net, tree, games, false, scope === 'endgame');
}
/** Select a strategy before the deal using only public table composition and
 * auction rules. Unknown contexts use the original tournament planner. */
export class CalibratedPolicy implements Policy {
  readonly name = 'mcts';
  private gate: Gate;
  private make: (s: Scope) => Policy;
  private policy!: Policy;
  constructor(gate: Gate, make: (scope: Scope) => Policy) { this.gate = gate; this.make = make; }
  onMatchStart(m: MatchStart): void {
    const key = marketContext(m.auctionType, m.simStrategies, m.seat);
    this.policy = this.make(this.gate.choices[key] ?? 'expert');
    this.policy.onMatchStart(m);
  }
  onAuction(v: AuctionView): Decision { return this.policy.onAuction(v); }
  onAuctionResult(r: AuctionOutcome): void { this.policy.onAuctionResult(r); }
  onWordPlayed(w: WordPlayed): void { this.policy.onWordPlayed(w); }
}
