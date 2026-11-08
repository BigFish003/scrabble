import { resolve } from 'node:path';
import { register } from '../upstream/bot-starter/sim/registry.ts';
import { MctsPolicy } from './mcts/policy.ts';
import { loadTree } from './mcts/storage.ts';
import { checkTree } from './mcts/tree.ts';
import { CalibratedPolicy, forScope, loadGate } from './mcts/gate.ts';

const path = process.env.SCRABBLE_MCTS_MODEL;
if (!path) throw new Error('Set SCRABBLE_MCTS_MODEL to the search checkpoint path.');
const { saved, net } = loadTree(resolve(path));
const tree = { ...saved.search, timeMs: Number(process.env.SCRABBLE_MCTS_MS ?? saved.search.timeMs) };
checkTree(tree);
const gate = process.env.SCRABBLE_MCTS_GATE ? loadGate(resolve(process.env.SCRABBLE_MCTS_GATE), resolve(path)) : undefined;
register('mcts', () => ({ name: 'mcts', make: () => gate
  ? new CalibratedPolicy(gate, scope => forScope(scope, net, tree, saved.progress.games))
  : new MctsPolicy(net, tree, saved.progress.games) }));
