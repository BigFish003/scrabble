import type { Agent } from '../upstream/bot-starter/src/strategy/agent.ts';
import type { Decision } from '../upstream/bot-starter/src/game/types.ts';
import { sharedIndex } from '../upstream/bot-starter/src/game/dictionary.ts';
import { canSpell, wordReward, type Counts } from '../upstream/bot-starter/src/game/letters.ts';
import { bidActions, type BidAction } from './actions.ts';
import { informationSetKey } from './belief.ts';
import { encode } from './features.ts';
import type { Observation } from './game.ts';

export interface DecisionPoint {
  observation: Observation; words: string[]; postCashRack: Counts; cashPoints: number;
  baselineBid: number; actions: BidAction[]; features: number[]; informationSet: string;
}
export function planDecision(planner: Agent, observation: Observation): DecisionPoint {
  return decisionFromPlan(observation, planner.onAuction(observation));
}
export function decisionFromPlan(observation: Observation, planned: Decision): DecisionPoint {
  const rack = observation.rack.slice();
  let cashPoints = 0;
  for (const word of planned.words) {
    if (!sharedIndex().isWord(word) || !canSpell(rack, word)) throw new Error(`Planner proposed an invalid word: ${word}`);
    for (const ch of word) rack[ch.charCodeAt(0) - 65]!--;
    cashPoints += wordReward(word);
  }
  const available = observation.balances[observation.seat]! + cashPoints;
  const baselineBid = Math.max(0, Math.min(available, Math.floor(planned.bid)));
  return { observation, words: planned.words.slice(), postCashRack: rack, cashPoints, baselineBid,
    actions: bidActions(baselineBid, available, observation.bagTotal === 0, observation.players.some(p => p.sim !== null)),
    features: encode(observation, rack, baselineBid, cashPoints), informationSet: informationSetKey(observation) };
}
