/** Two tanh layers with bid and value heads. The episodic bidder uses
 * advantage-weighted actions. MCTS uses a search distribution as its target.
 * Both share masked softmax, value regression, entropy and Adam.
 */
import { createHash } from 'node:crypto';
import { ACTION_NAMES, MCTS_ACTION_NAMES } from './actions.ts';
import { FEATURE_NAMES, FEATURE_VERSION } from './features.ts';
import { Rng } from './rng.ts';

const PARAM_NAMES = ['w1', 'b1', 'w2', 'b2', 'wp', 'bp', 'wv', 'bv'] as const;
type ParamName = typeof PARAM_NAMES[number];
export type Parameters = Record<ParamName, Float64Array>;
export interface Prediction { probabilities: number[]; value: number }
export interface Example { features: number[]; mask: boolean[]; action: number; target: number; advantage: number }
export interface SearchExample { features: number[]; mask: boolean[]; policy: number[]; target: number }
export interface LearnOptions { learningRate: number; entropy: number; valueWeight: number; maxGradientNorm: number }
export interface NetworkData {
  version: 1; featureVersion: number; featureHash: string; actionNames: string[]; hidden: number;
  parameters: Record<ParamName, number[]>; adam: { step: number; m: Record<ParamName, number[]>; v: Record<ParamName, number[]> };
}
export const FEATURE_HASH = createHash('sha256').update(JSON.stringify(FEATURE_NAMES)).digest('hex');

function zerosLike(params: Parameters): Parameters {
  return Object.fromEntries(PARAM_NAMES.map(k => [k, new Float64Array(params[k].length)])) as Parameters;
}
function affine(w: Float64Array, b: Float64Array, x: ArrayLike<number>): number[] {
  const out = new Array<number>(b.length);
  for (let i = 0; i < out.length; i++) {
    let y = b[i]!;
    for (let j = 0; j < x.length; j++) y += w[i * x.length + j]! * x[j]!;
    out[i] = y;
  }
  return out;
}
function outerAdd(g: Float64Array, dy: number[], x: number[]): void {
  for (let i = 0; i < dy.length; i++) for (let j = 0; j < x.length; j++) g[i * x.length + j]! += dy[i]! * x[j]!;
}
function transpose(w: Float64Array, dy: number[], width: number): number[] {
  const dx = new Array<number>(width).fill(0);
  for (let i = 0; i < dy.length; i++) for (let j = 0; j < width; j++) dx[j]! += w[i * width + j]! * dy[i]!;
  return dx;
}

export class ActorCritic {
  readonly hidden: number;
  readonly actionNames: readonly string[];
  readonly parameters: Parameters;
  private m: Parameters;
  private v: Parameters;
  private adamStep = 0;
  constructor(seed = 1, hidden = 64, actionNames: readonly string[] = ACTION_NAMES) {
    if (!Number.isInteger(hidden) || hidden < 1 || hidden > 256) throw new Error('Use 1–256 hidden units.');
    this.hidden = hidden;
    this.actionNames = [...actionNames];
    const input = FEATURE_NAMES.length, actions = actionNames.length;
    this.parameters = {
      w1: new Float64Array(hidden * input), b1: new Float64Array(hidden), w2: new Float64Array(hidden * hidden), b2: new Float64Array(hidden),
      wp: new Float64Array(actions * hidden), bp: new Float64Array(actions), wv: new Float64Array(hidden), bv: new Float64Array(1),
    };
    const rng = new Rng(seed);
    for (const [key, fanIn, fanOut] of [['w1', input, hidden], ['w2', hidden, hidden], ['wv', hidden, 1]] as const) {
      const scale = Math.sqrt(6 / (fanIn + fanOut));
      for (let i = 0; i < this.parameters[key].length; i++) this.parameters[key][i] = (rng.next() * 2 - 1) * scale;
    }
    this.parameters.bp.set([-2, -1, -0.5, 2, -0.5, -1, -2, -2, -2]);
    this.m = zerosLike(this.parameters); this.v = zerosLike(this.parameters);
  }
  private forward(features: number[], mask: boolean[]) {
    if (features.length !== FEATURE_NAMES.length || features.some(v => !Number.isFinite(v))) throw new Error('Invalid network inputs.');
    if (mask.length !== this.actionNames.length || !mask.some(Boolean)) throw new Error('Action mask does not match the network.');
    const p = this.parameters;
    const h1 = affine(p.w1, p.b1, features).map(Math.tanh), h2 = affine(p.w2, p.b2, h1).map(Math.tanh);
    const logits = affine(p.wp, p.bp, h2);
    const max = Math.max(...logits.filter((_, i) => mask[i]));
    const exp = logits.map((l, i) => mask[i] ? Math.exp(l - max) : 0), z = exp.reduce((a, b) => a + b, 0);
    const probabilities = exp.map(v => v / z);
    let value = p.bv[0]!;
    for (let j = 0; j < h2.length; j++) value += p.wv[j]! * h2[j]!;
    return { h1, h2, probabilities, value };
  }
  predict(features: number[], mask: boolean[]): Prediction {
    const { probabilities, value } = this.forward(features, mask);
    return { probabilities, value };
  }
  act(features: number[], mask: boolean[], rng: Rng, deterministic = false): number {
    const { probabilities } = this.predict(features, mask);
    if (deterministic) return probabilities.indexOf(Math.max(...probabilities));
    let r = rng.next();
    for (let i = 0; i < probabilities.length; i++) { r -= probabilities[i]!; if (r < 0) return i; }
    return mask.lastIndexOf(true);
  }

  /** Advantages are fixed targets: the baseline is not differentiated through
   * the actor. Averaging and entropy include only the current legal actions. */
  lossAndGradient(examples: (Example | SearchExample)[], options: Pick<LearnOptions, 'entropy' | 'valueWeight'>): { loss: number; gradients: Parameters; entropy: number; valueLoss: number } {
    if (!examples.length) throw new Error('Cannot train an empty batch.');
    const p = this.parameters, g = zerosLike(p);
    let loss = 0, entropyTotal = 0, valueLoss = 0;
    for (const e of examples) {
      if (!Number.isFinite(e.target)) throw new Error('Invalid value target.');
      let weights: number[];
      let mass: number;
      if ('policy' in e) {
        weights = e.policy;
        mass = weights.reduce((sum, p) => sum + p, 0);
        if (weights.length !== e.mask.length || weights.some((p, i) => !Number.isFinite(p) || p < 0 || (!e.mask[i] && p > 0)) || Math.abs(mass - 1) > 1e-6) {
          throw new Error('Invalid search policy target.');
        }
      } else {
        if (!e.mask[e.action] || !Number.isFinite(e.advantage)) throw new Error('Invalid training example.');
        mass = e.advantage;
        weights = e.mask.map((_, i) => i === e.action ? mass : 0);
      }
      const f = this.forward(e.features, e.mask), probs = f.probabilities;
      const entropy = -probs.reduce((sum, v) => sum + (v > 0 ? v * Math.log(v) : 0), 0);
      const error = f.value - e.target, vloss = 0.5 * options.valueWeight * error * error;
      const crossEntropy = -weights.reduce((sum, w, i) => sum + w * Math.log(Math.max(1e-30, probs[i]!)), 0);
      loss += crossEntropy - options.entropy * entropy + vloss;
      entropyTotal += entropy; valueLoss += vloss;
      const dz = probs.map((prob, i) => e.mask[i] ? mass * prob - weights[i]! +
        options.entropy * prob * (Math.log(Math.max(1e-30, prob)) + entropy) : 0);
      const dv = options.valueWeight * error;
      outerAdd(g.wp, dz, f.h2);
      dz.forEach((v, i) => g.bp[i]! += v);
      const dh2 = transpose(p.wp, dz, this.hidden);
      for (let j = 0; j < this.hidden; j++) { g.wv[j]! += dv * f.h2[j]!; dh2[j]! += dv * p.wv[j]!; }
      g.bv[0]! += dv;
      const d2 = dh2.map((v, j) => v * (1 - f.h2[j]! ** 2));
      outerAdd(g.w2, d2, f.h1); d2.forEach((v, i) => g.b2[i]! += v);
      const d1 = transpose(p.w2, d2, this.hidden).map((v, j) => v * (1 - f.h1[j]! ** 2));
      outerAdd(g.w1, d1, e.features); d1.forEach((v, i) => g.b1[i]! += v);
    }
    for (const k of PARAM_NAMES) for (let i = 0; i < g[k].length; i++) g[k][i]! /= examples.length;
    return { loss: loss / examples.length, gradients: g, entropy: entropyTotal / examples.length, valueLoss: valueLoss / examples.length };
  }
  learn(examples: (Example | SearchExample)[], options: LearnOptions): { loss: number; entropy: number; valueLoss: number; gradientNorm: number } {
    const { loss, entropy, valueLoss, gradients } = this.lossAndGradient(examples, options);
    let norm2 = 0;
    for (const k of PARAM_NAMES) for (const v of gradients[k]) norm2 += v * v;
    const gradientNorm = Math.sqrt(norm2);
    if (!Number.isFinite(loss) || !Number.isFinite(gradientNorm)) throw new Error('Training diverged; checkpoint not updated.');
    const scale = Math.min(1, options.maxGradientNorm / Math.max(1e-12, gradientNorm));
    this.adamStep++;
    const c1 = 1 - 0.9 ** this.adamStep, c2 = 1 - 0.999 ** this.adamStep;
    for (const k of PARAM_NAMES) for (let i = 0; i < gradients[k].length; i++) {
      const d = gradients[k][i]! * scale;
      this.m[k][i] = 0.9 * this.m[k][i]! + 0.1 * d;
      this.v[k][i] = 0.999 * this.v[k][i]! + 0.001 * d * d;
      this.parameters[k][i]! -= options.learningRate * (this.m[k][i]! / c1) / (Math.sqrt(this.v[k][i]! / c2) + 1e-8);
    }
    return { loss, entropy, valueLoss, gradientNorm };
  }
  toJSON(): NetworkData {
    const arrays = (p: Parameters) => Object.fromEntries(PARAM_NAMES.map(k => [k, [...p[k]]])) as Record<ParamName, number[]>;
    return { version: 1, featureVersion: FEATURE_VERSION, featureHash: FEATURE_HASH, actionNames: [...this.actionNames], hidden: this.hidden,
      parameters: arrays(this.parameters), adam: { step: this.adamStep, m: arrays(this.m), v: arrays(this.v) } };
  }
  static fromJSON(data: NetworkData): ActorCritic {
    const knownActions = [ACTION_NAMES, MCTS_ACTION_NAMES].some(names => JSON.stringify(names) === JSON.stringify(data.actionNames));
    if (data.version !== 1 || data.featureVersion !== FEATURE_VERSION || data.featureHash !== FEATURE_HASH || !knownActions) throw new Error('Checkpoint feature/action schema mismatch.');
    const net = new ActorCritic(0, data.hidden, data.actionNames);
    for (const key of PARAM_NAMES) {
      for (const [target, source] of [[net.parameters[key], data.parameters[key]], [net.m[key], data.adam.m[key]], [net.v[key], data.adam.v[key]]] as const) {
        if (!source || source.length !== target.length || source.some(v => !Number.isFinite(v))) throw new Error(`Invalid checkpoint parameter: ${key}`);
        target.set(source);
      }
    }
    if (!Number.isInteger(data.adam.step) || data.adam.step < 0) throw new Error('Invalid optimizer step.');
    net.adamStep = data.adam.step;
    return net;
  }
}
