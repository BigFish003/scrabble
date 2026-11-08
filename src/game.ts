/** Step-wise forward model of the pinned upstream sim/engine.ts.
 * Pure JSON game state; policies are deliberately outside the transition.
 * One step accepts a complete simultaneous joint action. Module sims are
 * engine controlled and bid using their balance at auction opening.
 */
import { sharedIndex, type WordIndex } from '../upstream/bot-starter/src/game/dictionary.ts';
import { BAG_DISTRIBUTION, BAG_SIZE, STARTING_BALANCE, STARTING_RACK_SIZE, canSpell, emptyCounts, letterAt, wordReward, type AuctionType, type Counts } from '../upstream/bot-starter/src/game/letters.ts';
import { simBid, SIM_BID_JITTER, type MatchResult, type SeatResult } from '../upstream/bot-starter/sim/engine.ts';
import type { AuctionOutcome, AuctionView, Decision, MatchStart, SimStrategy, WordPlayed } from '../upstream/bot-starter/src/game/types.ts';
import { Rng, type RngState } from './rng.ts';

export interface PlayerMeta { name: string; sim: SimStrategy | null; offline: boolean }
export interface PlayerState extends PlayerMeta {
  initialRack: Counts; rack: Counts; policySeed: number; balance: number; score: number;
  lettersWon: number; spent: number; wordsPlayed: number; rejectedWords: number; rejectedBids: number; bids: number[];
}
export interface RoundRecord {
  round: number; letter: string; bagTotal: number; balances: number[]; scores: number[];
  words: WordPlayed[]; outcome: AuctionOutcome; simWords: WordPlayed[];
}
export interface GameState {
  version: 1; auctionType: AuctionType; rackSize: number; players: PlayerState[];
  bag: Counts; bagTotal: number; letter: number; round: number; done: boolean;
  auctions: number; unsold: number; history: RoundRecord[];
  random: { bag: RngState; order: RngState; sim: RngState; tie: RngState };
}
/** The complete policy-visible information. No game seed, hidden deal,
 * opponent policy instances, bag composition, PRNG or sealed bids. */
export interface Observation extends AuctionView {
  seat: number; auctionType: AuctionType; rackSize: number; ownInitialRack: Counts;
  players: PlayerMeta[]; history: RoundRecord[]; ownBids: number[]; done: boolean;
}
export interface GameConfig { seed: number; auctionType: AuctionType; players: PlayerMeta[]; rackSize?: number }
export type PublicEvent = { kind: 'word'; value: WordPlayed } | { kind: 'result'; value: AuctionOutcome };

function draw(s: GameState, rng: Rng): number {
  if (s.bagTotal === 0) return -1;
  let j = rng.int(s.bagTotal);
  for (let c = 0; c < 26; c++) {
    if (j < s.bag[c]!) { s.bag[c]!--; s.bagTotal--; return c; }
    j -= s.bag[c]!;
  }
  throw new Error('Tile accounting failed.');
}

export function createGame(config: GameConfig): GameState {
  const n = config.players.length;
  const rackSize = config.rackSize ?? STARTING_RACK_SIZE;
  if (!Number.isInteger(n) || n < 2 || n > 12) throw new Error('Use 2–12 players.');
  if (!Number.isInteger(rackSize) || rackSize < 0 || rackSize > 20 || rackSize * n >= BAG_SIZE) throw new Error('Deal must leave at least one auction tile.');
  if (!['Vickrey', 'FirstPrice'].includes(config.auctionType)) throw new Error('Invalid auction mode.');
  const root = new Rng(config.seed);
  const bag = root.fork(1), order = root.fork(2), seeds = root.fork(3), sim = root.fork(4), tie = root.fork(5);
  const s: GameState = {
    version: 1, auctionType: config.auctionType, rackSize,
    players: config.players.map(p => ({ ...p, rack: emptyCounts(), initialRack: emptyCounts(), policySeed: 0,
      balance: STARTING_BALANCE, score: 0, lettersWon: 0, spent: 0, wordsPlayed: 0, rejectedWords: 0, rejectedBids: 0, bids: [] })),
    bag: BAG_DISTRIBUTION.slice(), bagTotal: BAG_SIZE, letter: -1, round: 1, done: false,
    auctions: 0, unsold: 0, history: [], random: { bag: bag.snapshot(), order: order.snapshot(), sim: sim.snapshot(), tie: tie.snapshot() },
  };
  for (const p of s.players) {
    for (let k = 0; k < rackSize; k++) p.rack[draw(s, bag)]!++;
    p.initialRack = p.rack.slice();
    if (!p.sim) p.policySeed = seeds.nextU32();
  }
  s.letter = draw(s, bag);
  s.random.bag = bag.snapshot();
  return s;
}

export function matchStart(s: GameState, seat: number): MatchStart {
  const p = s.players[seat]!;
  return { seat, nPlayers: s.players.length, names: s.players.map(p => p.name), simStrategies: s.players.map(p => p.sim),
    offline: s.players.map(p => p.offline), auctionType: s.auctionType, rack: p.initialRack.slice(), seed: p.policySeed };
}
export function observe(s: GameState, seat: number, copyRecords = true): Observation {
  if (!s.players[seat]) throw new Error('Invalid observing seat.');
  return { seat, auctionType: s.auctionType, rackSize: s.rackSize, ownInitialRack: s.players[seat]!.initialRack.slice(),
    players: s.players.map(({ name, sim, offline }) => ({ name, sim, offline })),
    round: s.round, letter: letterAt(s.letter), bagTotal: s.bagTotal, rack: s.players[seat]!.rack.slice(),
    balances: s.players.map(p => p.balance), scores: s.players.map(p => p.score),
    history: copyRecords ? structuredClone(s.history) : s.history.slice(), ownBids: s.players[seat]!.bids.slice(), done: s.done };
}
export function auctionView(s: GameState, seat: number): AuctionView {
  return { round: s.round, letter: letterAt(s.letter), bagTotal: s.bagTotal, rack: s.players[seat]!.rack.slice(),
    balances: s.players.map(p => p.balance), scores: s.players.map(p => p.score) };
}

/** Mutates only this explicit state. Clone before branching. The returned event
 * order is exactly words -> auction result -> module winner's word. */
export function advance(s: GameState, decisions: readonly (Decision | null)[], index: WordIndex = sharedIndex()): PublicEvent[] {
  if (s.done) throw new Error('Cannot step a terminal match.');
  if (decisions.length !== s.players.length) throw new Error('One joint-action slot per player is required.');
  const sim = Rng.restore(s.random.sim), order = Rng.restore(s.random.order), tie = Rng.restore(s.random.tie);
  const record: RoundRecord = { round: s.round, letter: letterAt(s.letter), bagTotal: s.bagTotal,
    balances: s.players.map(p => p.balance), scores: s.players.map(p => p.score), words: [],
    outcome: { round: s.round, letter: letterAt(s.letter), winner: null, paid: 0 }, simWords: [] };
  const bids: { player: number; amount: number }[] = [];
  const external: number[] = [];
  s.players.forEach((p, i) => {
    if (!p.sim) { if (!decisions[i]) throw new Error(`Missing action for seat ${i}`); external.push(i); return; }
    const b = simBid(p.sim, s.letter, p.balance, sim.int(2 * SIM_BID_JITTER + 1) - SIM_BID_JITTER);
    p.bids.push(b);
    if (b > 0) bids.push({ player: i, amount: b });
  });
  const play = (seat: number, raw: string): WordPlayed | null => {
    const p = s.players[seat]!;
    const word = String(raw).toUpperCase();
    if (word.length < 2 || !/^[A-Z]+$/.test(word) || !index.isWord(word) || !canSpell(p.rack, word)) { p.rejectedWords++; return null; }
    for (const ch of word) p.rack[ch.charCodeAt(0) - 65]!--;
    const reward = wordReward(word);
    p.score += reward; p.balance += reward; p.wordsPlayed++;
    return { player: seat, word, reward };
  };
  for (const i of order.shuffle(external)) {
    const p = s.players[i]!, decision = decisions[i]!;
    for (const word of decision.words) { const w = play(i, word); if (w) record.words.push(w); }
    const b = Math.floor(Number(decision.bid) || 0);
    p.bids.push(b);
    if (b <= 0) continue;
    if (!Number.isFinite(b) || b > p.balance) { p.rejectedBids++; continue; }
    bids.push({ player: i, amount: b });
  }
  // Same stable-sort-after-shuffle as upstream; sealed actions never appear
  // in observations or in the public history.
  const sorted = tie.shuffle(bids).sort((a, b) => b.amount - a.amount);
  if (sorted.length) {
    const top = sorted[0]!;
    const paid = s.auctionType === 'FirstPrice' ? top.amount : sorted.length === 1 ? 1 : sorted[1]!.amount;
    const p = s.players[top.player]!;
    p.balance -= paid; p.spent += paid; p.rack[s.letter]!++; p.lettersWon++;
    record.outcome = { round: s.round, letter: letterAt(s.letter), winner: top.player, paid };
    if (p.sim) { const word = index.longestWord(p.rack, 3); if (word) { const w = play(top.player, word); if (w) record.simWords.push(w); } }
  } else s.unsold++;
  s.auctions++;
  s.history.push(record);
  const events: PublicEvent[] = [
    ...record.words.map(value => ({ kind: 'word' as const, value })),
    { kind: 'result', value: record.outcome },
    ...record.simWords.map(value => ({ kind: 'word' as const, value })),
  ];
  const bag = Rng.restore(s.random.bag);
  const next = draw(s, bag);
  if (next < 0) s.done = true;
  else { s.letter = next; s.round++; }
  s.random = { bag: bag.snapshot(), order: order.snapshot(), sim: sim.snapshot(), tie: tie.snapshot() };
  return events;
}

export function result(s: GameState): MatchResult {
  return { auctions: s.auctions, unsold: s.unsold, seats: s.players.map(p => ({
    name: p.name, score: p.score, balance: p.balance, rank: 1 + s.players.filter(q => q.score > p.score).length,
    lettersWon: p.lettersWon, spent: p.spent, wordsPlayed: p.wordsPlayed, rejectedWords: p.rejectedWords,
    rejectedBids: p.rejectedBids, unplayedTiles: p.rack.reduce((a, b) => a + b, 0), decisionMs: [],
  } satisfies SeatResult)) };
}
