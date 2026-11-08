import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ActorCritic, type Example } from '../src/network.ts';
import { ACTION_NAMES } from '../src/actions.ts';
import { FEATURE_NAMES } from '../src/features.ts';

test('actor-critic analytical gradients match finite differences through both hidden layers', () => {
  const net = new ActorCritic(3, 4);
  // Nonzero actor weights ensure the actor's gradient reaches the shared layers.
  net.parameters.wp.forEach((_, i, a) => a[i] = Math.sin(i) / 10);
  const example: Example = { features: FEATURE_NAMES.map((_, i) => Math.sin(i + 1) / 5),
    mask: ACTION_NAMES.map((_, i) => i !== 1), action: 4, target: 1.2, advantage: 0.7 };
  const options = { entropy: 0.03, valueWeight: 0.5 };
  const { gradients } = net.lossAndGradient([example], options), epsilon = 1e-5;
  for (const key of Object.keys(net.parameters) as (keyof typeof net.parameters)[]) {
    const a = net.parameters[key];
    for (const i of new Set([0, Math.floor(a.length / 2), a.length - 1])) {
      const old = a[i]!;
      a[i] = old + epsilon; const plus = net.lossAndGradient([example], options).loss;
      a[i] = old - epsilon; const minus = net.lossAndGradient([example], options).loss;
      a[i] = old;
      assert.ok(Math.abs((plus - minus) / (2 * epsilon) - gradients[key][i]!) < 1e-6, `${key}[${i}]`);
    }
  }
});

test('learning increases rewarded action probability; masked actions stay at zero', () => {
  const net = new ActorCritic(5, 8), features = FEATURE_NAMES.map(() => 0.1), mask = ACTION_NAMES.map((_, i) => i === 0 || i === 3);
  const before = net.predict(features, mask).probabilities[0]!;
  for (let i = 0; i < 30; i++) net.learn([{ features, mask, action: 0, target: 1, advantage: 1 }],
    { learningRate: 0.01, entropy: 0.01, valueWeight: 0.5, maxGradientNorm: 1 });
  const after = net.predict(features, mask);
  assert.ok(after.probabilities[0]! > before + 0.5);
  after.probabilities.forEach((p, i) => { if (!mask[i]) assert.equal(p, 0); });
});

test('checkpoint round trip retains predictions and Adam state, and rejects schema drift', () => {
  const net = new ActorCritic(9, 8), features = FEATURE_NAMES.map(() => 0.2), mask = ACTION_NAMES.map(() => true);
  const example = { features, mask, action: 3, target: 2, advantage: 0.5 };
  const options = { learningRate: 0.001, entropy: 0.01, valueWeight: 0.5, maxGradientNorm: 0.5 };
  net.learn([example], options);
  const restored = ActorCritic.fromJSON(JSON.parse(JSON.stringify(net.toJSON())));
  assert.deepEqual(net.predict(features, mask), restored.predict(features, mask));
  net.learn([example], options); restored.learn([example], options);
  assert.deepEqual(net.toJSON(), restored.toJSON());
  const wrong = net.toJSON(); wrong.featureHash = 'wrong';
  assert.throws(() => ActorCritic.fromJSON(wrong), /schema mismatch/);
});
