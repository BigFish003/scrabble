import { resolve } from 'node:path';
import { register } from '../upstream/bot-starter/sim/registry.ts';
import { readCheckpoint } from './checkpoint.ts';
import { LearnedBidPolicy } from './learned-policy.ts';

const path = process.env.SCRABBLE_MODEL;
if (!path) throw new Error('Set SCRABBLE_MODEL to an absolute checkpoint path before importing register-learned.ts.');
const { network } = readCheckpoint(resolve(path));
register('learned', () => ({ name: 'learned', make: () => new LearnedBidPolicy(network) }));
