import { MCTS_ACTION_NAMES, baselineAction, type BidAction } from '../actions.ts';
import { publicEvidence, sampleWorld, type Evidence } from '../belief.ts';
import { observe } from '../game.ts';
import { ActorCritic } from '../network.ts';
import type { DecisionPoint } from '../planning.ts';
import { Rng } from '../rng.ts';
import { MarketBelief, quantile } from './market.ts';
import { WordCache, continuation, copyParticleState, innerPosition, observationBranch, publicValue,
  simulateAuction, terminalValue, type Particle, type Position } from './rollout.ts';

export interface TreeConfig {
  simulations: number; particles: number; width: number; depth: number; horizon: number;
  every: number; timeMs: number; seed: number;
}
export const DEFAULT_TREE: TreeConfig = {
  simulations: 80, particles: 8, width: 5, depth: 4, horizon: 6, every: 4, timeMs: 150, seed: 82931,
};
export function checkTree(config: TreeConfig): void {
  const bounds: [keyof TreeConfig, number, number][] = [['simulations', 1, 100000], ['particles', 1, 256],
    ['width', 2, MCTS_ACTION_NAMES.length], ['depth', 1, 32], ['horizon', 1, 98], ['every', 1, 98],
    ['timeMs', 0, 60000], ['seed', 0, 0xffffffff]];
  for (const [key, low, high] of bounds) {
    if (!Number.isInteger(config[key]) || config[key] < low || config[key] > high) throw new Error('Invalid tree setting ' + key);
  }
  if (config.depth > config.horizon) throw new Error('Tree depth must fit the rollout horizon.');
}
export function initialTreeNetwork(seed: number): ActorCritic {
  const net = new ActorCritic(seed, 32, MCTS_ACTION_NAMES);
  net.parameters.bp.fill(0);
  net.parameters.bp[3] = 1.3;
  net.parameters.wv.fill(0);
  net.parameters.bv[0] = 0.5;
  return net;
}

export interface Analysis {
  bid: number; id: number; reason: 'searched' | 'expert' | 'endgame' | 'forced' | 'deadline';
  mask: boolean[]; visits: number[]; target: number[]; simulations: number; worlds: number;
  nodes: number; deepSelections: number; maxTreeDepth: number; timedOut: boolean;
}
interface Edge {
  action: BidAction; prior: number; visits: number; sum: number; children: Map<string, TreeNode>;
  worldValues: Map<number, { visits: number; sum: number }>;
}
interface TreeNode { point: Position; edges: Edge[]; visits: number; estimate: number }
type Prices = ReturnType<MarketBelief['clearingPrices']>;

function availableActions(p: Position, prices: Prices): BidAction[] {
  const o = p.observation;
  const balance = o.balances[o.seat]! + p.cashPoints;
  const bids = [...p.actions.map(a => a.bid), quantile(prices, 0.5) + 1, quantile(prices, 0.9) + 1, balance];
  const actions = bids.map((b, id) => ({ id, name: MCTS_ACTION_NAMES[id]!, bid: Math.min(balance, b), legal: false }));
  const finalPass = o.bagTotal === 0 && o.players.every(p => !p.sim);
  const seen = new Set<number>();
  for (const id of [0, 3, 9, 10, 11, 1, 2, 4, 5, 6, 7, 8]) {
    const a = actions[id]!;
    if ((finalPass && id !== 0) || (id === 11 && o.bagTotal > 2) || seen.has(a.bid)) continue;
    seen.add(a.bid);
    a.legal = true;
  }
  return actions;
}

function endgameBid(p: Position): number | undefined {
  const o = p.observation;
  const modules = o.players.some(p => p.sim);
  const balance = o.balances[o.seat]! + p.cashPoints;
  if (o.bagTotal === 0) return modules ? balance : 0;
  // At an all-external table this is the last usable tile. Money cannot buy
  // another scoring opportunity. Acquiring it can also deny a rival a word.
  if (o.bagTotal === 1 && !modules) return balance;
  return undefined;
}

function makeNode(point: Position, evidence: Evidence, network: ActorCritic,
  cache: WordCache, width: number, valueWeight: number, prices: Prices, actions: BidAction[]): TreeNode {
  const mask = actions.map(a => a.legal);
  const prediction = network.predict(point.features, mask);
  const o = point.observation;
  const gain = cache.gain(point.postCashRack, o.letter.charCodeAt(0) - 65);
  const reservation = Math.max(1, point.baselineBid * (o.auctionType === 'FirstPrice' ? 1.3 : 1), gain * 0.9);
  const legal = actions.filter(a => a.legal);
  const marketScores = legal.map(a => prices.reduce((ev, price) => {
    if (a.bid === 0 || price.amount > a.bid) return ev;
    const win = price.amount === a.bid ? 0.5 : 1;
    const cost = o.auctionType === 'FirstPrice' ? a.bid : Math.max(1, price.amount);
    return ev + price.p * win * (reservation - cost);
  }, 0));
  const maximum = Math.max(...marketScores), scale = Math.max(1.5, reservation * 0.25);
  const marketExp = marketScores.map(v => Math.exp((v - maximum) / scale));
  const total = marketExp.reduce((a, b) => a + b, 0);
  const candidates = legal.map((action, i) => ({ action, prior: 0.46 * prediction.probabilities[action.id]! +
    0.46 * marketExp[i]! / total + 0.08 / legal.length }));
  candidates.sort((a, b) => b.prior - a.prior || a.action.bid - b.action.bid);
  const baseline = baselineAction(actions);
  const chosen = new Set<number>();
  const take = (id: number | undefined) => { if (id !== undefined && chosen.size < width) chosen.add(id); };
  const exact = endgameBid(point);
  if (exact !== undefined) {
    take(legal.find(a => a.bid === exact)!.id);
  } else {
    take(baseline);
    take(legal[marketScores.indexOf(maximum)]!.id);
    take(candidates[0]!.action.id);
    if (o.bagTotal <= 2) take(legal.find(a => a.bid === o.balances[o.seat]! + point.cashPoints)?.id);
    take(candidates.find(a => a.action.bid > actions[baseline]!.bid)?.action.id);
    take(candidates.find(a => a.action.bid < actions[baseline]!.bid)?.action.id);
    for (const c of candidates) take(c.action.id);
  }
  const edges = candidates.filter(c => chosen.has(c.action.id)).map(c => ({ ...c, visits: 0, sum: 0,
    children: new Map<string, TreeNode>(), worldValues: new Map<number, { visits: number; sum: number }>() }));
  const priorSum = edges.reduce((sum, edge) => sum + edge.prior, 0);
  for (const e of edges) e.prior /= priorSum;
  const estimate = (1 - valueWeight) * publicValue(o, evidence, cache) + valueWeight * Math.max(0, Math.min(1, prediction.value));
  return { point, edges, visits: 0, estimate };
}

function select(node: TreeNode): Edge {
  let best = node.edges[0]!, bestScore = -Infinity;
  for (const edge of node.edges) {
    const value = edge.visits ? edge.sum / edge.visits : node.estimate;
    const score = value + 0.32 * edge.prior * Math.sqrt(node.visits + 1) / (1 + edge.visits);
    if (score > bestScore) { best = edge; bestScore = score; }
  }
  return best;
}

/** Matched particles reduce noise between bids. The standard-error discount
 * is a decision heuristic for a small adaptive tree, not a confidence bound. */
function preferredEdge(root: TreeNode, baseline: number): Edge {
  const reference = root.edges.find(e => e.action.id === baseline)!;
  let best = reference, bestGain = 0.0015;
  for (const edge of root.edges) {
    if (edge === reference) continue;
    const differences: number[] = [];
    for (const [world, a] of edge.worldValues) {
      const b = reference.worldValues.get(world);
      if (b) differences.push(a.sum / a.visits - b.sum / b.visits);
    }
    if (differences.length < 2) continue;
    const mean = differences.reduce((a, b) => a + b, 0) / differences.length;
    const variance = differences.reduce((sum, d) => sum + (d - mean) ** 2, 0) / (differences.length - 1);
    const adjusted = mean - 0.5 * Math.sqrt(variance / differences.length);
    if (adjusted > bestGain) { bestGain = adjusted; best = edge; }
  }
  return best;
}

/** Single-observer information-set MCTS. The entry point accepts only a
 * decision prepared by the learner and its public market model. One new
 * observation node can be expanded per simulation. Backups always use the
 * learner's utility in this simultaneous multiplayer game. */
export function searchBids(point: DecisionPoint, network: ActorCritic, market: MarketBelief,
  config: TreeConfig, trainedGames = 0, started = performance.now()): Analysis {
  const o = point.observation, seat = o.seat;
  const deadline = config.timeMs === 0 ? Infinity : started + config.timeMs;
  market.sync(o);
  const prices = market.clearingPrices(o);
  const actions = availableActions(point, prices);
  const baseline = baselineAction(actions);
  const stats: Analysis = { bid: actions[baseline]!.bid, id: baseline, reason: 'expert', mask: actions.map(a => a.legal),
    target: actions.map(() => 0), visits: actions.map(() => 0), simulations: 0, nodes: 0, deepSelections: 0, worlds: 0,
    maxTreeDepth: 0, timedOut: false };
  const finish = (id: number, reason: Analysis['reason']): Analysis => {
    stats.id = id; stats.bid = actions[id]!.bid; stats.reason = reason;
    if (!stats.simulations) stats.target[id] = 1;
    return stats;
  };
  const solved = endgameBid(point);
  if (solved !== undefined) return finish(actions.find(a => a.legal && a.bid === solved)!.id, 'endgame');
  if (actions.filter(a => a.legal).length === 1) return finish(baseline, 'forced');
  const cache = new WordCache();
  const consequential = o.bagTotal < 8 || (point.baselineBid >= 5 && point.baselineBid >= 0.28 * (o.balances[seat]! + point.cashPoints)) ||
    cache.gain(point.postCashRack, o.letter.charCodeAt(0) - 65) >= 15;
  if (!consequential && (o.round - 1) % config.every !== 0) return finish(baseline, 'expert');
  if (performance.now() >= deadline) { stats.timedOut = true; return finish(baseline, 'deadline'); }

  const evidence = publicEvidence(o);
  const weight = 0.35 * trainedGames / (trainedGames + 64);
  const root = makeNode(point, evidence, network, cache, config.width, weight, prices, actions);
  stats.nodes = 1;
  const limit = Math.min(o.bagTotal + 1, config.horizon);
  const rng = new Rng((Number.parseInt(point.informationSet.slice(0, 8), 16) ^ config.seed) >>> 0);
  const particles: Particle[] = [];
  const particleAt = (i: number): Particle => {
    while (particles.length <= i) {
      const state = sampleWorld(o, rng.nextU32(), evidence);
      const styles = o.players.map((_, j) => market.sample(j, rng));
      const noise = Array.from({ length: limit }, () => o.players.map(() => rng.next()));
      particles.push({ state, styles, noise });
    }
    return particles[i]!;
  };
  // Every shortlisted bid receives common initial worlds before adaptive PUCT.
  // The baseline goes first so an interrupted comparison has a reference.
  const initial = [root.edges.find(e => e.action.id === baseline)!, ...root.edges.filter(e => e.action.id !== baseline)];
  for (let sim = 0; sim < config.simulations; sim++) {
    if (performance.now() >= deadline) { stats.timedOut = true; break; }
    const first = sim < initial.length * 2 ? initial[sim % initial.length]! : select(root);
    const world = first.visits % config.particles;
    const particle = particleAt(world);
    const state = copyParticleState(particle.state);
    const path: { node: TreeNode; edge: Edge }[] = [];
    let node = root, depth = 0, expanded = false;
    while (!state.done && depth < limit) {
      if (performance.now() >= deadline) { stats.timedOut = true; break; }
      if (expanded || depth >= config.depth) {
        simulateAuction(state, seat, continuation(state, seat, cache), particle, depth++, cache);
        continue;
      }
      const edge = depth === 0 ? first : select(node);
      path.push({ node, edge });
      simulateAuction(state, seat, { words: node.point.words, bid: edge.action.bid }, particle, depth++, cache);
      if (state.done || depth >= config.depth || depth >= limit) continue;
      const key = observationBranch(state, seat);
      let child = edge.children.get(key);
      if (!child) {
        const inner = innerPosition(state, seat, cache), marketPrices = market.clearingPrices(inner.point.observation);
        child = makeNode(inner.point, inner.evidence, network, cache, config.width, weight, marketPrices, availableActions(inner.point, marketPrices));
        edge.children.set(key, child);
        stats.nodes++;
        expanded = true;
      }
      node = child;
    }
    if (stats.timedOut) break; // Never back up a partially simulated horizon.
    let value: number;
    if (state.done) value = terminalValue(state.players.map(p => p.score), seat);
    else if (weight > 0) {
      const inner = innerPosition(state, seat, cache);
      const leafActions = availableActions(inner.point, market.clearingPrices(inner.point.observation));
      const prediction = network.predict(inner.point.features, leafActions.map(a => a.legal));
      value = (1 - weight) * publicValue(inner.point.observation, inner.evidence, cache) + weight * Math.max(0, Math.min(1, prediction.value));
    } else {
      const visible = observe(state, seat, false);
      value = publicValue(visible, publicEvidence(visible), cache);
    }
    for (const { node: visited, edge } of path) {
      visited.visits++; edge.visits++; edge.sum += value;
    }
    const sample = first.worldValues.get(world) ?? { visits: 0, sum: 0 };
    sample.visits++; sample.sum += value; first.worldValues.set(world, sample);
    stats.simulations++;
    stats.deepSelections += Math.max(0, path.length - 1);
    stats.maxTreeDepth = Math.max(stats.maxTreeDepth, path.length);
  }
  stats.worlds = particles.length;
  if (!stats.simulations) return finish(baseline, 'deadline');
  const choice = preferredEdge(root, baseline);
  for (const edge of root.edges) {
    stats.visits[edge.action.id] = edge.visits;
    stats.target[edge.action.id] = 0.55 * edge.visits / stats.simulations;
  }
  stats.target[choice.action.id]! += 0.45;
  return finish(choice.action.id, 'searched');
}
