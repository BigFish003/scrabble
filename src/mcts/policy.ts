import type { AuctionView, MatchStart } from '../../upstream/bot-starter/src/game/types.ts';
import { LearnedBidPolicy } from '../learned-policy.ts';
import type { ActorCritic, SearchExample } from '../network.ts';
import type { DecisionPoint } from '../planning.ts';
import { MarketBelief } from './market.ts';
import { checkTree, searchBids, type TreeConfig } from './tree.ts';

export function emptyCounters() {
  return { decisions: 0, searched: 0, simulations: 0, deepSelections: 0, maxDepth: 0, worlds: 0, nodes: 0,
    overrides: 0, searchOverrides: 0, endgames: 0, timeStops: 0 };
}
export type SearchCounters = ReturnType<typeof emptyCounters>;
export class MctsPolicy extends LearnedBidPolicy {
  private treeNetwork: ActorCritic;
  private config: TreeConfig;
  private trainedGames: number;
  private collect: boolean;
  private endgameOnly: boolean;
  private market!: MarketBelief;
  private started = 0;
  counters = emptyCounters();
  examples: Omit<SearchExample, 'target'>[] = [];
  constructor(network: ActorCritic, config: TreeConfig, trainedGames: number, collect = false, endgameOnly = false) {
    super(network, { name: 'mcts' });
    checkTree(config);
    this.treeNetwork = network;
    this.config = config;
    this.trainedGames = trainedGames;
    this.collect = collect;
    this.endgameOnly = endgameOnly;
  }
  override onMatchStart(m: MatchStart): void {
    super.onMatchStart(m);
    this.counters = emptyCounters();
    this.examples = [];
    this.market = new MarketBelief(m.names.map((name, i) => ({ name, sim: m.simStrategies[i]!, offline: !!m.offline?.[i] })));
  }
  override onAuction(v: AuctionView) {
    this.started = performance.now();
    return super.onAuction(v);
  }
  protected override selectBid(point: DecisionPoint): number {
    if (this.endgameOnly && point.observation.bagTotal + 1 > this.config.horizon) {
      this.market.sync(point.observation);
      this.counters.decisions++;
      return point.baselineBid;
    }
    const result = searchBids(point, this.treeNetwork, this.market, this.config, this.trainedGames, this.started);
    const counts = this.counters;
    counts.decisions++;
    counts.searched += Number(result.reason === 'searched');
    counts.simulations += result.simulations;
    counts.deepSelections += result.deepSelections;
    counts.maxDepth = Math.max(counts.maxDepth, result.maxTreeDepth);
    counts.nodes += result.nodes;
    counts.worlds += result.worlds;
    counts.overrides += Number(result.bid !== point.baselineBid);
    counts.searchOverrides += Number(result.reason === 'searched' && result.bid !== point.baselineBid);
    counts.endgames += Number(result.reason === 'endgame');
    counts.timeStops += Number(result.timedOut);
    if (this.collect && (result.reason === 'endgame' || (result.reason === 'searched' && result.simulations >= 2 * this.config.width))) {
      this.examples.push({ features: point.features, mask: result.mask, policy: result.target });
    }
    return result.bid;
  }
}
