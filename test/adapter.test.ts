import { test } from 'node:test';
import assert from 'node:assert/strict';
import '../upstream/bot-starter/sim/strategies.ts';
import { runMatch } from '../upstream/bot-starter/sim/engine.ts';
import { makeSeat } from '../upstream/bot-starter/sim/registry.ts';
import { ActorCritic } from '../src/network.ts';
import { LearnedBidPolicy } from '../src/learned-policy.ts';
import { ScrabbleEnv } from '../src/env.ts';
import { Rng } from '../src/rng.ts';

test('callback adapter produces the same changed bidding policy as the RL environment', () => {
  const network = new ActorCritic(55, 8);
  network.parameters.bp[1] = 4; // Prefer half of the planner bid, not the original.
  const field = ['brutus-sim', 'random', 'greedy'];
  for (const auctionType of ['Vickrey', 'FirstPrice'] as const) {
    const env = new ScrabbleEnv({ opponents: field, auctionType });
    let p = env.reset(88), rng = new Rng(0);
    while (!env.done) p = env.step(network.act(p.features, p.actions.map(a => a.legal), rng, true)).observation!;
    const expected = runMatch([{ name: 'learner', make: () => new LearnedBidPolicy(network, { name: 'learner' }) }, ...field.map(name => makeSeat(name))], { seed: 88, auctionType });
    assert.deepEqual(env.result(), expected);
  }
});
