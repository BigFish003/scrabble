import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ScrabbleEnv } from '../src/env.ts';
import { sampleWorld } from '../src/belief.ts';
import { observe } from '../src/game.ts';
import { MCTS_ACTION_NAMES } from '../src/actions.ts';
import { FEATURE_NAMES } from '../src/features.ts';
import { ActorCritic } from '../src/network.ts';
import { outcomeLikelihood, MarketBelief } from '../src/mcts/market.ts';
import { DEFAULT_TREE, initialTreeNetwork, searchBids } from '../src/mcts/tree.ts';
import { chooseScopes, type Credit } from '../src/mcts/calibration.ts';
import { CalibratedPolicy, marketContext, type Gate, type Scope } from '../src/mcts/gate.ts';
import { play } from '../src/mcts/matches.ts';

test('public winner/payment likelihood matches all joint sealed bids, including ties, reserve and no sale', () => {
  const supports = [[0, 1, 2, 4], [0, 1, 1, 3], [0, 2, 3, 4]];
  const perOutcome = 1 / supports.reduce((n, s) => n * s.length, 1);
  for (const mode of ['Vickrey', 'FirstPrice'] as const) {
    const brute = new Map<string, number>();
    const add = (winner: number | null, paid: number, p: number) => {
      const key = JSON.stringify([winner, paid]);
      brute.set(key, (brute.get(key) ?? 0) + p);
    };
    for (const a of supports[0]!) for (const b of supports[1]!) for (const c of supports[2]!) {
      const bids = [a, b, c], high = Math.max(...bids);
      if (!high) { add(null, 0, perOutcome); continue; }
      const winners = [0, 1, 2].filter(i => bids[i] === high);
      const second = bids.toSorted((x, y) => y - x)[1]!;
      for (const w of winners) add(w, mode === 'FirstPrice' ? high : Math.max(1, second), perOutcome / winners.length);
    }
    const at = (price: number) => supports.map(s => ({
      less: s.filter(x => x < price).length / s.length, equal: s.filter(x => x === price).length / s.length,
      greater: s.filter(x => x > price).length / s.length,
    }));
    assert.equal(outcomeLikelihood(at(0), mode, null, 0), brute.get('[null,0]'));
    for (let paid = 1; paid <= 4; paid++) for (let winner = 0; winner < supports.length; winner++) {
      const expected = brute.get(JSON.stringify([winner, paid])) ?? 0;
      assert.ok(Math.abs(outcomeLikelihood(at(paid), mode, winner, paid) - expected) < 1e-12);
    }
  }
});

test('information-set search makes identical multi-level backups from indistinguishable hidden worlds', () => {
  const env = new ScrabbleEnv({ opponents: ['random', 'valor-sim', 'greedy'] });
  const point = env.reset(41), copy = structuredClone(point);
  const a = sampleWorld(point.observation, 210), b = sampleWorld(point.observation, 211);
  assert.notDeepEqual(a.bag, b.bag);
  const first = observe(a, env.seat), second = observe(b, env.seat);
  assert.deepEqual(first, second);
  const net = initialTreeNetwork(554);
  const config = { ...DEFAULT_TREE, simulations: 72, particles: 3, width: 3, depth: 4, horizon: 5, every: 1, timeMs: 0 };
  const run = (o: typeof first) => searchBids({ ...point, observation: o }, net, new MarketBelief(o.players), config);
  const result = run(first);
  assert.deepEqual(result, run(second));
  assert.deepEqual(point, copy);
  assert.equal(result.reason, 'searched');
  assert.equal(result.simulations, config.simulations);
  assert.equal(result.worlds, config.particles);
  assert.equal(result.visits.reduce((n, v) => n + v, 0), result.simulations);
  assert.ok(result.deepSelections > 0 && result.maxTreeDepth >= 2 && result.nodes > 1);
  assert.ok(Math.abs(result.target.reduce((n, v) => n + v, 0) - 1) < 1e-10);
  for (const [i, v] of result.target.entries()) if (!result.mask[i]) assert.equal(v, 0);
  assert.ok(result.mask[result.id] && Number.isInteger(result.bid));
  assert.ok(result.bid <= first.balances[first.seat]! + point.cashPoints);
});

test('endgame uses post-cash balance when the tile can deny scoring and passes on a dead tile', () => {
  const net = initialTreeNetwork(14);
  for (const [opponent, remaining, allIn] of [['random', 1, true], ['random', 0, false], ['valor-sim', 0, true]] as const) {
    const env = new ScrabbleEnv({ opponents: [opponent] });
    const point = env.reset(88);
    while (env.decision().observation.bagTotal > remaining) env.step(0);
    const end = env.decision(), market = new MarketBelief(end.observation.players);
    const result = searchBids(end, net, market, { ...DEFAULT_TREE, timeMs: 0 });
    const available = end.observation.balances[end.observation.seat]! + end.cashPoints;
    assert.ok(available > 0);
    assert.equal(result.bid, allIn ? available : 0);
    assert.equal(result.reason, 'endgame');
    assert.equal(result.target[result.id], 1);
    const expired = searchBids(point, net, new MarketBelief(point.observation.players), { ...DEFAULT_TREE, every: 1, timeMs: 1 }, 0, performance.now() - 1000);
    assert.equal(expired.bid, point.baselineBid);
    assert.equal(expired.simulations, 0);
    assert.ok(expired.timedOut);
  }
});

test('search imitation uses normalized legal visit targets and round-trips optimizer state', () => {
  const net = initialTreeNetwork(3);
  const features = FEATURE_NAMES.map((_, i) => Math.sin(i) / 4), mask = MCTS_ACTION_NAMES.map((_, i) => [3, 9, 10].includes(i));
  const policy = mask.map((_, i) => i === 9 ? 0.75 : i === 10 ? 0.25 : 0);
  const example = { features, mask, policy, target: 0.9 };
  const options = { entropy: 0.001, learningRate: 0.003, valueWeight: 0.6, maxGradientNorm: 1 };
  const analytic = net.lossAndGradient([example], options).gradients.bp;
  const delta = 1e-5;
  for (const i of [0, 3, 9, 10]) {
    const p = net.parameters.bp, old = p[i]!;
    p[i] = old + delta; const plus = net.lossAndGradient([example], options).loss;
    p[i] = old - delta; const minus = net.lossAndGradient([example], options).loss;
    p[i] = old; assert.ok(Math.abs((plus - minus) / (2 * delta) - analytic[i]!) < 1e-6);
  }
  const before = net.predict(features, mask).probabilities[9]!;
  for (let k = 0; k < 12; k++) net.learn([example], options);
  assert.ok(net.predict(features, mask).probabilities[9]! > before);
  const restored = ActorCritic.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
  net.learn([example], options); restored.learn([example], options);
  assert.deepEqual(net.toJSON(), restored.toJSON());
  assert.throws(() => net.learn([{ ...example, policy: policy.map((p, i) => i === 0 ? 0.2 : p) }], options), /policy target/);
});

test('scope selection uses paired seed blocks and protects win credit in each public table type', () => {
  const credits: Credit[] = [];
  for (let block = 0; block < 8; block++) for (const mode of ['full', 'endgame'] as const) {
    credits.push({ block, context: 'late', mode, winDelta: 0, utilityDelta: mode === 'endgame' ? 0.025 : 0 });
    credits.push({ block, context: 'lose', mode, winDelta: -0.1, utilityDelta: 0.03 });
    credits.push({ block, context: 'noisy', mode, winDelta: 0, utilityDelta: block % 2 === 0 ? 0.31 : -0.29 });
    if (block < 4) credits.push({ block, context: 'small', mode, winDelta: 0.5, utilityDelta: 0.2 });
  }
  const { choices } = chooseScopes(credits);
  assert.deepEqual(choices, { late: 'endgame', lose: 'expert', noisy: 'expert', small: 'expert' });
});

test('registered gate uses only public table type, falls back on unfamiliar tables and bounds endgame search', () => {
  const gate: Gate = { format: 'scrabble-search-gate-v1', modelHash: 'test', seed: 1, seedBlocks: 8, fields: [],
    choices: { 'Vickrey/3/mixed': 'endgame' } };
  const chosen: Scope[] = [];
  const policy = new CalibratedPolicy(gate, scope => {
    chosen.push(scope);
    return { name: 'test', onMatchStart() {}, onAuction() { return { words: [], bid: 3 }; }, onAuctionResult() {}, onWordPlayed() {} };
  });
  const start = { seat: 0, nPlayers: 3, names: ['me', 'unknown A', 'unknown B'], simStrategies: [null, null, 'ValueBidder'] as const,
    seed: 131, rack: new Array<number>(26).fill(0), auctionType: 'Vickrey' as const };
  policy.onMatchStart(start);
  policy.onMatchStart({ ...start, names: ['nobody', 'different', 'renamed'] });
  policy.onMatchStart({ ...start, auctionType: 'FirstPrice' });
  assert.deepEqual(chosen, ['endgame', 'endgame', 'expert']);
  assert.equal(marketContext('Vickrey', [null, null, null], 1), 'Vickrey/3/external');
  const config = { ...DEFAULT_TREE, horizon: 3, depth: 2, simulations: 16, particles: 2, width: 3, timeMs: 0 };
  const result = play({ seed: 80001, field: 'mixed', block: 0, auctionType: 'FirstPrice', seat: 3 },
    config, initialTreeNetwork(1), 0, false, 'endgame');
  assert.equal(result.scope, 'endgame');
  assert.ok(result.counters.searched > 0 && result.counters.searched + result.counters.endgames <= config.horizon);
  assert.equal(result.metrics.rejectedBids + result.metrics.rejectedWords, 0);
});
