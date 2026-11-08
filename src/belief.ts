import { createHash } from 'node:crypto';
import { BAG_DISTRIBUTION, countsOf, emptyCounts, type Counts } from '../upstream/bot-starter/src/game/letters.ts';
import { createGame, type GameState, type Observation } from './game.ts';
import { Rng } from './rng.ts';

export interface Evidence {
  won: Counts[]; played: Counts[]; minimumDeal: Counts[]; minimumRack: Counts[];
  hiddenDealSizes: number[]; unseen: Counts;
}

/** Condition on every public word at the time it was played. A later win
 * cannot retroactively explain letters used in an earlier word. */
export function publicEvidence(o: Observation): Evidence {
  const n = o.players.length;
  const won = Array.from({ length: n }, emptyCounts), played = Array.from({ length: n }, emptyCounts);
  const minimumDeal = Array.from({ length: n }, emptyCounts);
  const revealed = emptyCounts();
  const noteWord = (p: number, word: string) => {
    const used = countsOf(word);
    for (let c = 0; c < 26; c++) {
      played[p]![c]! += used[c]!;
      minimumDeal[p]![c] = Math.max(minimumDeal[p]![c]!, played[p]![c]! - won[p]![c]!);
    }
  };
  for (const h of o.history) {
    const c = h.letter.charCodeAt(0) - 65;
    revealed[c]!++;
    for (const w of h.words) noteWord(w.player, w.word);
    if (h.outcome.winner !== null) won[h.outcome.winner]![c]!++;
    for (const w of h.simWords) noteWord(w.player, w.word);
  }
  if (!o.done) revealed[o.letter.charCodeAt(0) - 65]!++;
  const unseen = BAG_DISTRIBUTION.map((v, c) => v - revealed[c]! - o.ownInitialRack[c]! -
    minimumDeal.reduce((sum, r, p) => sum + (p === o.seat ? 0 : r[c]!), 0));
  const hiddenDealSizes = minimumDeal.map((r, p) => p === o.seat ? 0 : o.rackSize - r.reduce((a, b) => a + b, 0));
  if (unseen.some(v => v < 0) || hiddenDealSizes.some(v => v < 0)) throw new Error('Observation contradicts the tile distribution or public history.');
  const minimumRack = minimumDeal.map((r, p) => r.map((v, c) => v + won[p]![c]! - played[p]![c]!));
  return { won, played, minimumDeal, minimumRack, hiddenDealSizes, unseen };
}

/** Uniform physical-consistency sampler, NOT an opponent-behavior posterior.
 * Only this observation and a fresh independent seed enter the sampler.
 * Remaining unseen initial tiles are distributed without replacement; the
 * rest is the bag. Thus all past words and every tile count stay feasible.
 */
export function sampleWorld(o: Observation, seed: number, e: Evidence = publicEvidence(o)): GameState {
  const rng = new Rng(seed);
  const s = createGame({ seed: rng.nextU32(), auctionType: o.auctionType, rackSize: o.rackSize, players: o.players });
  const pool = e.unseen.slice();
  const take = () => {
    let j = rng.int(pool.reduce((a, b) => a + b, 0));
    for (let c = 0; c < 26; c++) { if (j < pool[c]!) { pool[c]!--; return c; } j -= pool[c]!; }
    throw new Error('Hidden-tile accounting failed.');
  };
  s.players.forEach((p, i) => {
    const initial = i === o.seat ? o.ownInitialRack.slice() : e.minimumDeal[i]!.slice();
    for (let k = 0; k < e.hiddenDealSizes[i]!; k++) initial[take()]!++;
    p.initialRack = initial;
    p.rack = initial.map((v, c) => v + e.won[i]![c]! - e.played[i]![c]!);
    if (p.rack.some(v => v < 0)) throw new Error('Sampled an impossible rack.');
    p.balance = o.balances[i]!; p.score = o.scores[i]!;
    p.lettersWon = o.history.filter(h => h.outcome.winner === i).length;
    p.spent = o.history.reduce((sum, h) => sum + (h.outcome.winner === i ? h.outcome.paid : 0), 0);
    p.wordsPlayed = o.history.reduce((sum, h) => sum + [...h.words, ...h.simWords].filter(w => w.player === i).length, 0);
    p.bids = i === o.seat ? o.ownBids.slice() : [];
  });
  if (pool.reduce((a, b) => a + b, 0) !== o.bagTotal) throw new Error('Sampled bag size disagrees with observation.');
  if (s.players[o.seat]!.rack.some((v, c) => v !== o.rack[c])) throw new Error('Own rack disagrees with public history.');
  s.bag = pool; s.bagTotal = o.bagTotal; s.letter = o.letter.charCodeAt(0) - 65;
  s.round = o.round; s.done = o.done; s.auctions = o.history.length;
  s.unsold = o.history.filter(h => h.outcome.winner === null).length;
  s.history = structuredClone(o.history);
  // Never carry the real simulator's future PRNG into search.
  const random = new Rng(rng.nextU32());
  s.random = { bag: random.fork(1).snapshot(), order: random.fork(2).snapshot(), sim: random.fork(4).snapshot(), tie: random.fork(5).snapshot() };
  return s;
}

/** A search node groups worlds by the learner's information, including its
 * own past bids; sampled racks, bag contents and RNG state are absent. */
export function informationSetKey(o: Observation): string {
  return createHash('sha256').update(JSON.stringify([
    o.seat, o.auctionType, o.rackSize, o.players, o.ownInitialRack, o.rack,
    o.round, o.letter, o.bagTotal, o.balances, o.scores, o.history, o.ownBids, o.done,
  ])).digest('hex');
}
