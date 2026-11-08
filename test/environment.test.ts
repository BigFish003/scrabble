import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../upstream/bot-starter/sim/strategies.ts';
import { runMatch } from '../upstream/bot-starter/sim/engine.ts';
import { makeSeat } from '../upstream/bot-starter/sim/registry.ts';
import { createAgent } from '../upstream/bot-starter/src/strategy.ts';
import { Rng as OriginalRng } from '../upstream/bot-starter/src/game/rng.ts';
import { BAG_DISTRIBUTION, countsOf } from '../upstream/bot-starter/src/game/letters.ts';
import { Rng } from '../src/rng.ts';
import { advance, auctionView, createGame, matchStart, observe, result, type Observation } from '../src/game.ts';
import { ScrabbleEnv } from '../src/env.ts';
import { baselineAction, bidActions } from '../src/actions.ts';
import { informationSetKey, publicEvidence, sampleWorld } from '../src/belief.ts';

const fields = [
  ['starter', 'greedy', 'overbidder', 'hoarder', 'random'],
  ['brutus-sim', 'valor-sim', 'cheapo-sim', 'random', 'greedy'],
  ['jitter', 'final'],
];

test('RNG matches upstream and restores the exact stream', () => {
  for (const seed of [0, 1, 99, 0xffffffff]) {
    const a = new Rng(seed), b = new OriginalRng(seed);
    for (let i = 0; i < 100; i++) assert.equal(a.nextU32(), b.nextU32());
    const restored = Rng.restore(a.snapshot());
    for (let i = 0; i < 100; i++) assert.equal(restored.nextU32(), a.nextU32());
  }
});

test('step-wise forward model reproduces upstream complete matches in both modes', () => {
  for (const auctionType of ['Vickrey', 'FirstPrice'] as const) for (const field of fields) for (const seed of [1, 7]) {
    const specs = field.map(name => makeSeat(name));
    const s = createGame({ seed, auctionType, players: specs.map(p => ({ name: p.name, sim: p.sim ?? null, offline: !!p.offline })) });
    const policies = specs.map(p => p.make?.() ?? null);
    policies.forEach((p, i) => p?.onMatchStart(matchStart(s, i)));
    while (!s.done) {
      const decisions = policies.map((p, i) => p?.onAuction(auctionView(s, i)) ?? null);
      for (const e of advance(s, decisions)) for (const p of policies) {
        if (e.kind === 'word') p?.onWordPlayed(e.value); else p?.onAuctionResult(e.value);
      }
    }
    assert.deepEqual(result(s), runMatch(specs, { seed, auctionType }), `${auctionType} ${field} seed ${seed}`);
  }
});

test('learner baseline agrees with upstream final, including mixed module sims', () => {
  for (const auctionType of ['Vickrey', 'FirstPrice'] as const) for (const field of fields.slice(0, 2)) {
    const env = new ScrabbleEnv({ opponents: field, auctionType });
    let p = env.reset(4), reward = 0;
    while (!env.done) { const s = env.step(baselineAction(p.actions)); reward += s.reward; p = s.observation!; }
    const expected = runMatch([{ name: 'learner', make: () => createAgent('learner') }, ...field.map(name => makeSeat(name))], { seed: 4, auctionType });
    assert.deepEqual(env.result(), expected);
    assert.ok(Math.abs(reward - expected.seats[0]!.score / 100) < 1e-10);
    assert.equal(expected.auctions, 68);
  }
});

test('clone, snapshot/restore and portable episode replay preserve future behavior', () => {
  const env = new ScrabbleEnv({ opponents: ['random', 'greedy'], learnerSeat: 1 });
  let p = env.reset(33);
  for (let i = 0; i < 12; i++) p = env.step(baselineAction(p.actions)).observation!;
  const before = env.debugState(), snapshot = env.snapshot(), fork = env.clone();
  fork.step(0);
  assert.deepEqual(env.debugState(), before, 'fork must not mutate parent');
  assert.deepEqual(env.decision(), p, 'repeated inspection must not resample planner');
  const replayed = ScrabbleEnv.fromReplay(JSON.parse(JSON.stringify(env.exportReplay())));
  assert.deepEqual(replayed.debugState(), before);
  const a = env.step(baselineAction(p.actions)), b = replayed.step(baselineAction(p.actions));
  assert.deepEqual(a, b);
  env.restore(snapshot);
  assert.deepEqual(env.debugState(), before);
  assert.deepEqual(env.step(baselineAction(p.actions)), a);
});

test('observations and information-set keys exclude hidden truth', () => {
  const env = new ScrabbleEnv({ opponents: ['random', 'greedy'] });
  const point = env.reset(19), a = env.debugState(), b = structuredClone(a);
  b.random.bag.a ^= 12345; b.bag.reverse(); b.players[1]!.rack.reverse(); b.players[1]!.initialRack.reverse(); b.players[1]!.policySeed = 7;
  const oa = observe(a, 0), ob = observe(b, 0);
  assert.deepEqual(oa, ob);
  assert.equal(informationSetKey(oa), informationSetKey(ob));
  assert.deepEqual(sampleWorld(oa, 77), sampleWorld(ob, 77));
  const keys: string[] = [];
  const walk = (value: unknown) => { if (value && typeof value === 'object') for (const [key, child] of Object.entries(value)) { keys.push(key); walk(child); } };
  walk(point);
  for (const forbidden of ['policySeed', 'random', 'initialRacks', 'bag', 'seed']) assert.ok(!keys.includes(forbidden), forbidden);
});

test('belief sampler respects time-ordered public words and later wins', () => {
  const players = [{ name: 'me', sim: null, offline: false }, { name: 'them', sim: null, offline: false }];
  const o: Observation = { seat: 0, auctionType: 'Vickrey', rackSize: 2, ownInitialRack: countsOf('XY'), rack: countsOf('XY'),
    players, round: 3, letter: 'C', bagTotal: 91, balances: [100, 101], scores: [0, 2], ownBids: [0, 0], done: false,
    history: [
      { round: 1, letter: 'B', bagTotal: 93, balances: [100, 100], scores: [0, 0], words: [{ player: 1, word: 'AT', reward: 2 }],
        outcome: { round: 1, letter: 'B', winner: null, paid: 0 }, simWords: [] },
      { round: 2, letter: 'A', bagTotal: 92, balances: [100, 102], scores: [0, 2], words: [],
        outcome: { round: 2, letter: 'A', winner: 1, paid: 1 }, simWords: [] },
    ] };
  assert.deepEqual(publicEvidence(o).minimumRack[1], countsOf('A'));
  for (let seed = 0; seed < 50; seed++) {
    const world = sampleWorld(o, seed);
    assert.deepEqual(world.players[1]!.initialRack, countsOf('AT'));
    assert.deepEqual(world.players[1]!.rack, countsOf('A'));
    assert.equal(world.bag.reduce((a, b) => a + b, 0), o.bagTotal);
  }
});

test('sampled search branches conserve tiles, replay public history and fork independently', () => {
  const env = new ScrabbleEnv({ opponents: ['random', 'valor-sim', 'greedy'] });
  let p = env.reset(42);
  for (let k = 0; k < 8; k++) p = env.step(baselineAction(p.actions)).observation!;
  const before = env.debugState();
  for (let seed = 100; seed < 105; seed++) {
    const root = env.searchRoot(seed, ['random', 'valor-sim', 'greedy']);
    assert.deepEqual(root.decision(), p);
    const s = root.debugState();
    const revealed = countsOf(p.observation.history.map(h => h.letter).join('') + p.observation.letter);
    for (let c = 0; c < 26; c++) assert.equal(s.bag[c]! + s.players.reduce((v, p) => v + p.initialRack[c]!, 0) + revealed[c]!, BAG_DISTRIBUTION[c]);
    const fork = root.clone(), a = root.step(baselineAction(p.actions)), b = fork.step(baselineAction(p.actions));
    assert.deepEqual(a, b);
  }
  assert.deepEqual(env.debugState(), before);
});

test('bid actions are affordable and deduplicated, including zero baseline exploration', () => {
  for (const baseline of [0, 1, 2, 10, 100]) for (const available of [0, 1, 17]) {
    const actions = bidActions(baseline, available), legal = actions.filter(a => a.legal);
    assert.equal(new Set(legal.map(a => a.bid)).size, legal.length);
    assert.ok(legal.every(a => a.bid >= 0 && a.bid <= available && Number.isInteger(a.bid)));
  }
  assert.ok(bidActions(0, 100).some(a => a.legal && a.bid > 0));
  assert.deepEqual(bidActions(20, 100, true).filter(a => a.legal).map(a => a.id), [0]);
  assert.ok(bidActions(0, 100, true, true).some(a => a.legal && a.bid > 0), 'final denial against a module sim remains possible');
});

test('a final denial bid can prevent a module sim from scoring its immediately played word', () => {
  const initial = createGame({ seed: 5, auctionType: 'Vickrey', players: [
    { name: 'me', sim: null, offline: false }, { name: 'sim', sim: 'Cheapskate', offline: false },
  ] });
  initial.bag.fill(0); initial.bagTotal = 0; initial.letter = 19; // final T
  initial.players[0]!.rack.fill(0); initial.players[1]!.rack = countsOf('CA');
  const skip = structuredClone(initial), deny = structuredClone(initial);
  advance(skip, [{ words: [], bid: 0 }, null]);
  advance(deny, [{ words: [], bid: 4 }, null]);
  assert.ok(skip.players[1]!.score > 0);
  assert.equal(deny.players[1]!.score, 0);
  assert.equal(deny.players[0]!.score, 0, 'external winner cannot play the final tile');
  assert.ok(skip.done && deny.done);
});

test('rank and win rewards arrive only at terminal; final tile gets no extra external play', () => {
  for (const objective of ['rank', 'win'] as const) {
    const env = new ScrabbleEnv({ opponents: ['idle'], objective, rackSize: 0 });
    let p = env.reset(9), sum = 0;
    while (!env.done) {
      if (p.observation.bagTotal === 0) assert.deepEqual(p.actions.filter(a => a.legal).map(a => a.id), [0]);
      const s = env.step(0); sum += s.reward;
      if (!s.terminated) assert.equal(s.reward, 0);
      p = s.observation!;
    }
    assert.equal(env.result().auctions, 98);
    assert.equal(sum, objective === 'rank' ? 1 : 0.5);
  }
});
