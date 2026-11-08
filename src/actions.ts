export const ACTION_NAMES = ['skip', '0.5x', '0.75x', '1x', '1.25x', '1.5x', 'bid-1', 'bid-2', 'bid-4'] as const;
export const BASELINE_ACTION = 3;
export const MCTS_ACTION_NAMES = [...ACTION_NAMES, 'clear-median', 'clear-high', 'endgame-all-in'] as const;
export interface BidAction { id: number; name: string; bid: number; legal: boolean }

/** Low absolute bids permit exploration even when the planner bids zero.
 * Merge amounts after rounding/capping, favoring the baseline among duplicates.
 * External bots cannot use the final tile. Bidding can still deny a module
 * sim its immediate word play, so preserve final actions when sims are present.
 */
export function bidActions(baseline: number, available: number, finalAuction = false, allowFinalDenial = false): BidAction[] {
  const raw = [0, baseline * 0.5, baseline * 0.75, baseline, baseline * 1.25, baseline * 1.5, 1, 2, 4];
  const actions = raw.map((bid, id) => ({ id, name: ACTION_NAMES[id]!, bid: Math.max(0, Math.min(Math.floor(available), Math.round(bid))), legal: false }));
  const seen = new Set<number>();
  for (const id of [0, 3, 1, 2, 4, 5, 6, 7, 8]) {
    const a = actions[id]!;
    if ((finalAuction && !allowFinalDenial && id !== 0) || seen.has(a.bid)) continue;
    a.legal = true; seen.add(a.bid);
  }
  return actions;
}
export function baselineAction(actions: BidAction[]): number {
  return actions.find(a => a.legal && a.bid === actions[BASELINE_ACTION]!.bid)?.id ?? 0;
}
