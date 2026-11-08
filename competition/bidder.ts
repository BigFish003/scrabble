import { fileURLToPath } from 'node:url';
import type { AuctionView, WordPlayed } from '../upstream/bot-starter/src/game/types.ts';
import type { LiveOptions } from '../upstream/bot-starter/src/live/adapter.ts';
import { BAG_SIZE } from '../upstream/bot-starter/src/game/letters.ts';
import type { Observation, RoundRecord } from '../src/game.ts';
import { decisionFromPlan } from '../src/planning.ts';
import { loadGate, marketContext } from '../src/mcts/gate.ts';
import { MarketBelief } from '../src/mcts/market.ts';
import { loadTree } from '../src/mcts/storage.ts';
import { searchBids } from '../src/mcts/tree.ts';

export type Bidder = 'expert' | 'calibrated' | 'mcts';
type Time = { microsSinceUnixEpoch: bigint };
export interface PublicWord { id: bigint; botId: bigint; word: string; totalReward: bigint; playedAt: Time }
export interface Settlement { winnerBotId?: bigint | null; paid: bigint; closedAt: Time; letter: string; matchId: bigint }
export interface MarketFeed {
  result(auctionId: bigint): Settlement | undefined;
  words(matchId: bigint): Iterable<PublicWord>;
  closesAt(auctionId: bigint): Time | undefined;
}
const model = fileURLToPath(new URL('../models/mcts-pilot-128.json', import.meta.url));
const gateFile = fileURLToPath(new URL('../models/mcts-pilot-gate.json', import.meta.url));

/** Build a search observation from the same opening balances, private rack and
 * actual public plays the live expert saw. A skipped round or inconsistent
 * snapshot throws once and the live adapter keeps the expert for that match. */
export function competitionSelector(bidder: Bidder, feed: MarketFeed): LiveOptions['bidSelector'] {
  if (bidder === 'expert') return undefined;
  const { saved, net } = loadTree(model);
  const gate = bidder === 'calibrated' ? loadGate(gateFile, model) : undefined;
  return (start, matchId, ids) => {
    const scope = gate ? gate.choices[marketContext(start.auctionType, start.simStrategies, start.seat)] ?? 'expert' : 'full';
    if (scope === 'expert') return undefined;
    const players = ids.map((_, i) => ({ name: start.names[i]!, sim: start.simStrategies[i]!, offline: !!start.offline?.[i] }));
    const market = new MarketBelief(players), history: RoundRecord[] = [], ownBids: number[] = [];
    const seenWords = new Set<bigint>(), seatOf = new Map(ids.map((id, i) => [id, i]));
    const rackSize = start.rack.reduce((a, b) => a + b, 0);
    let previous: { id: bigint; opening: AuctionView } | undefined;
    return (view, plan, auctionId, started) => {
      if (view.round !== history.length + (previous ? 2 : 1) || view.bagTotal !== BAG_SIZE - ids.length * rackSize - view.round)
        throw new Error('auction history or tile count changed');
      const newWords = Array.from(feed.words(matchId)).filter(w => !seenWords.has(w.id)).sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0);
      if (previous) {
        const r = feed.result(previous.id), opening = previous.opening;
        if (!r || r.matchId !== matchId || r.letter !== opening.letter) throw new Error('previous settlement unavailable');
        const winner = r.winnerBotId == null ? null : seatOf.get(r.winnerBotId);
        if (winner === undefined) throw new Error('settlement participant unavailable');
        const words: WordPlayed[] = [], after: WordPlayed[] = [];
        const balances = [...opening.balances], scores = [...opening.scores], rack = [...opening.rack];
        if (winner !== null) balances[winner]! -= Number(r.paid);
        if (winner === start.seat) rack[r.letter.charCodeAt(0) - 65]!++;
        for (const w of newWords) {
          const player = seatOf.get(w.botId);
          if (player === undefined) throw new Error('word participant unavailable');
          const reward = Number(w.totalReward), time = w.playedAt.microsSinceUnixEpoch, close = r.closedAt.microsSinceUnixEpoch;
          const afterResult = time > close || (time === close && player === winner && !!players[player]!.sim);
          (afterResult ? after : words).push({ player, word: w.word, reward });
          scores[player]! += reward; balances[player]! += reward;
          if (player === start.seat) for (const char of w.word) rack[char.charCodeAt(0) - 65]!--;
          seenWords.add(w.id);
        }
        if (scores.some((v, i) => v !== view.scores[i] || balances[i] !== view.balances[i]) || rack.some((v, i) => v !== view.rack[i]))
          throw new Error('live rack, score or balance has not reconciled');
        history.push({ round: opening.round, letter: opening.letter, bagTotal: opening.bagTotal,
          balances: [...opening.balances], scores: [...opening.scores], words, simWords: after,
          outcome: { round: opening.round, letter: r.letter, winner, paid: Number(r.paid) } });
      } else if (newWords.length || view.scores.some(s => s !== 0) || start.rack.some((v, i) => v !== view.rack[i])) {
        throw new Error('first auction snapshot already contains plays');
      }
      const observation: Observation = { ...view, seat: start.seat, auctionType: start.auctionType, rackSize,
        ownInitialRack: start.rack.slice(), players, history, ownBids, done: false };
      let bid = plan.bid;
      const close = feed.closesAt(auctionId);
      const elapsed = performance.now() - started;
      const timeLeft = close ? Number(close.microsSinceUnixEpoch / 1000n) - Date.now() - 100 : 0;
      const budget = Math.min(saved.search.timeMs, Math.floor(elapsed + timeLeft));
      if (budget > elapsed && (scope === 'full' || view.bagTotal < saved.search.horizon)) {
        const point = decisionFromPlan(observation, plan);
        bid = searchBids(point, net, market, { ...saved.search, timeMs: budget }, saved.progress.games, started).bid;
      }
      previous = { id: auctionId, opening: structuredClone(view) };
      ownBids.push(bid);
      return bid;
    };
  };
}
