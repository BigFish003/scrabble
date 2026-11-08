import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { train } from './train.ts';
import { checkFields, evaluateCases, evaluationPlan, FIELDS, makeEnv, runEpisode, summarize, trainingCase, type Mode } from './experiments.ts';
import { readCheckpoint, writeJSON, UPSTREAM_COMMIT } from './checkpoint.ts';
import { baselineAction } from './actions.ts';
import { compareRootActions } from './search.ts';
import type { Objective } from './env.ts';
import { DEFAULT_TREE, checkTree } from './mcts/tree.ts';
import { evaluateTree, trainTree } from './mcts/experiment.ts';
import { runCalibration } from './mcts/calibration.ts';

const help = `Scrabble auction RL / search laboratory (Node >=22.18; no npm install)

  node scripts/setup.mjs
  node --test "test/*.test.ts"
  node src/cli.ts bench --games 6
  node src/cli.ts train --games 512 --max-minutes 120 --out runs/bid-policy
  node src/cli.ts evaluate --model runs/bid-policy/latest.json --seeds 5
  node src/cli.ts search-demo --round 60 --worlds 2
  node src/cli.ts mcts-train --games 192 --workers 4 --out runs/mcts-pilot
  node src/cli.ts mcts-evaluate --model runs/mcts-pilot/best.json --seeds 16 --rotations 1
  node src/cli.ts mcts-calibrate --model models/mcts-pilot-128.json --gate runs/search-gate.json
  node src/cli.ts mcts-confirm --model models/mcts-pilot-128.json --gate runs/search-gate.json

Shared: --seed N --fields base,module,mixed --mode vickrey|first|both --out PATH
Train: --games TOTAL --batch 32 --objective score|win|rank|margin --lr 0.0003
       --entropy 0.01 --eval-every 128 (0 disables) --eval-games 12
       --max-minutes 120 --resume CHECKPOINT | --init CHECKPOINT
       Resume preserves checkpoint settings; TOTAL counts earlier games too.
Evaluate: --model CHECKPOINT --seeds 2 --fields base,module,strong,duel
          --stochastic (default is greedy inference)
          Each seed block rotates the learner through EVERY seat in each field/mode.
Search demo: --field module --round 60 --worlds 2 --search-seed 3000000
             --objective rank --models comma,separated,opponent,policy,models
             This compares root rollouts. Use mcts-train for multi-level tree search.

MCTS train: --games TOTAL --batch 16 --epochs 3 --lr 0.0015 --workers 4 --minutes 120
            --val-every 64 --val-games 16 --simulations 80 --particles 8 --width 5
            --depth 4 --horizon 6 --every 4 --time-ms 0 --resume CHECKPOINT
            Training uses a fixed simulation count by default. Resume restores saved settings.
MCTS evaluate: --model CHECKPOINT --seeds 16 --rotations 1 --workers 4 --time-ms 150
               Compare trained search, initial search and original expert on identical cases.
               One rotation cycles seats across seed blocks. Zero runs every seat.
MCTS calibrate: choose full search, endgame-only search or the expert by public table type.
MCTS confirm: test that choice against full search and the expert on fresh deals.

Fields: ${Object.keys(FIELDS).join(', ')}
`;

function integer(value: string | undefined, fallback: number, name: string, min = 1, max = 1_000_000): number {
  const n = value === undefined ? fallback : Number(value);
  if (!Number.isSafeInteger(n) || n < min || n > max) throw new Error(`${name} must be an integer in [${min}, ${max}].`);
  return n;
}
function number(value: string | undefined, fallback: number, name: string, min = 0, max = Infinity): number {
  const n = value === undefined ? fallback : Number(value);
  if (!Number.isFinite(n) || n < min || n > max) throw new Error(`${name} must be in [${min}, ${max}].`);
  return n;
}
function objective(value: string | undefined, fallback: Objective): Objective {
  const v = value ?? fallback;
  if (!['score', 'win', 'rank', 'margin'].includes(v)) throw new Error('Objective must be score, win, rank or margin.');
  return v as Objective;
}
async function main() {
  const [command = 'help', ...args] = process.argv.slice(2);
  const { values: v } = parseArgs({ args, options: {
    help: { type: 'boolean' }, seed: { type: 'string' }, fields: { type: 'string' }, mode: { type: 'string' }, out: { type: 'string' },
    games: { type: 'string' }, batch: { type: 'string' }, objective: { type: 'string' }, lr: { type: 'string' }, entropy: { type: 'string' },
    'eval-every': { type: 'string' }, 'eval-games': { type: 'string' }, 'max-minutes': { type: 'string' }, resume: { type: 'string' }, init: { type: 'string' },
    model: { type: 'string' }, gate: { type: 'string' }, seeds: { type: 'string' }, stochastic: { type: 'boolean' }, field: { type: 'string' },
    round: { type: 'string' }, worlds: { type: 'string' }, 'search-seed': { type: 'string' }, models: { type: 'string' },
    workers: { type: 'string' }, minutes: { type: 'string' }, epochs: { type: 'string' }, 'val-every': { type: 'string' },
    'val-games': { type: 'string' }, simulations: { type: 'string' }, particles: { type: 'string' }, width: { type: 'string' },
    depth: { type: 'string' }, horizon: { type: 'string' }, every: { type: 'string' }, 'time-ms': { type: 'string' }, rotations: { type: 'string' },
  }, strict: true });
  if (v.help || ['help', '--help', '-h'].includes(command)) { console.log(help); return; }
  const mode = (v.mode ?? 'both') as Mode;
  const defaultFields = command === 'mcts-calibrate' ? 'base,module,mixed,duel,practice' : command === 'mcts-confirm' ? 'mixed,duel,strong,heldout,league'
    : command === 'mcts-evaluate' ? 'mixed,duel,strong,heldout' : command === 'mcts-train' ? 'base,module,mixed,duel'
    : command === 'evaluate' ? 'base,module,strong,duel' : 'base,module,mixed';
  const fields = (v.fields ?? defaultFields).split(',');
  checkFields(fields);
  if (!['both', 'vickrey', 'first'].includes(mode)) throw new Error('Mode must be vickrey, first or both.');
  const defaultSeed = command === 'mcts-calibrate' ? 8250000 : command === 'mcts-confirm' ? 9600000
    : command === 'mcts-evaluate' ? 7900000 : command === 'mcts-train' ? 47001 : command === 'evaluate' ? 2_000_000 : 1;
  const seed = integer(v.seed, defaultSeed, 'seed', 0, 0xffffffff);
  if (command === 'mcts-train') {
    const tree = { simulations: integer(v.simulations, DEFAULT_TREE.simulations, 'simulations'), particles: integer(v.particles, DEFAULT_TREE.particles, 'particles'),
      width: integer(v.width, DEFAULT_TREE.width, 'width'), depth: integer(v.depth, DEFAULT_TREE.depth, 'depth'), horizon: integer(v.horizon, DEFAULT_TREE.horizon, 'horizon'),
      every: integer(v.every, DEFAULT_TREE.every, 'every'), timeMs: integer(v['time-ms'], 0, 'time-ms', 0, 60000),
      seed: integer(v['search-seed'], DEFAULT_TREE.seed, 'search-seed', 0, 0xffffffff) };
    checkTree(tree);
    await trainTree({ out: resolve(v.out ?? 'runs/mcts-pilot'), games: integer(v.games, 192, 'games'),
      workers: integer(v.workers, 4, 'workers', 1, 8), minutes: number(v.minutes, 120, 'minutes', 0.001), resume: v.resume },
    { seed, fields, mode, batchGames: integer(v.batch, 16, 'batch', 1, 512), epochs: integer(v.epochs, 3, 'epochs', 1, 64),
      learningRate: number(v.lr, 0.0015, 'lr', 1e-8, 0.1), validationEvery: integer(v['val-every'], 64, 'val-every', 0),
      validationGames: integer(v['val-games'], 16, 'val-games', 0, 4096), tree });
    return;
  }
  if (['mcts-evaluate', 'mcts-calibrate', 'mcts-confirm'].includes(command)) {
    if (!v.model) throw new Error('Supply --model CHECKPOINT.');
    const options = { model: resolve(v.model), out: resolve(v.out ?? 'runs/' + command + '.json'), workers: integer(v.workers, 4, 'workers', 1, 8),
      seed, seeds: integer(v.seeds, command === 'mcts-confirm' ? 24 : command === 'mcts-calibrate' ? 20 : 16, 'seeds', 1, 4096),
      rotations: integer(v.rotations, 1, 'rotations', 0, 6), fields, mode,
      timeMs: v['time-ms'] === undefined ? undefined : integer(v['time-ms'], 150, 'time-ms', 0, 60000) };
    if (command === 'mcts-evaluate') await evaluateTree(options);
    else {
      if (!v.gate) throw new Error('Supply --gate PATH for the calibrated strategy choices.');
      await runCalibration(options, resolve(v.gate), command === 'mcts-confirm');
    }
    return;
  }
  if (command === 'train') {
    if (v.resume && v.init) throw new Error('Choose resume or init.');
    await train({ out: resolve(v.out ?? 'runs/bid-policy'), games: integer(v.games, 512, 'games'), batchSize: integer(v.batch, 32, 'batch', 1, 1024),
      seed, fields, mode, objective: objective(v.objective, 'score'), learningRate: number(v.lr, 0.0003, 'lr', 1e-8, 0.1),
      entropy: number(v.entropy, 0.01, 'entropy', 0, 1), evalEvery: integer(v['eval-every'], 128, 'eval-every', 0),
      evalGames: integer(v['eval-games'], 12, 'eval-games', 0, 1000), maxMinutes: number(v['max-minutes'], 120, 'max-minutes'),
      resume: v.resume, init: v.init });
    return;
  }
  if (command === 'bench') {
    const games = integer(v.games, 6, 'games', 1, 10000), rows = [];
    const start = performance.now();
    for (let i = 0; i < games; i++) {
      const c = trainingCase(i, seed, fields, mode), result = runEpisode(c);
      rows.push({ ...c, ...result }); console.log(JSON.stringify({ event: 'benchmark-game', game: i + 1, ...rows.at(-1) }));
    }
    const elapsedSeconds = (performance.now() - start) / 1000;
    const report = { kind: 'baseline-throughput', node: process.versions.node, upstreamCommit: UPSTREAM_COMMIT, games, elapsedSeconds,
      decisions: rows.reduce((sum, r) => sum + r.decisions, 0), decisionsPerSecond: rows.reduce((sum, r) => sum + r.decisions, 0) / elapsedSeconds,
      note: 'Includes word planning and feature extraction. Training and search add work; warm-up and opponent field affect throughput.', rows };
    const out = resolve(v.out ?? 'runs/benchmark.json'); writeJSON(out, report); console.log(JSON.stringify({ ...report, rows: undefined, out }));
    return;
  }
  if (command === 'evaluate') {
    if (!v.model) throw new Error('Supply --model CHECKPOINT.');
    const { network } = readCheckpoint(v.model);
    const plan = evaluationPlan(seed, integer(v.seeds, 2, 'seeds', 1, 1000), fields, mode);
    let last = performance.now();
    console.log(JSON.stringify({ event: 'evaluation-start', pairedMatches: plan.length, actualMatches: plan.length * 2, seed, fields, mode, stochastic: !!v.stochastic }));
    const rows = evaluateCases(plan, network, (done, total) => {
      if (performance.now() - last > 20000 || done === total) { console.log(JSON.stringify({ event: 'evaluation-progress', done, total })); last = performance.now(); }
    }, undefined, !!v.stochastic);
    const report = { kind: 'paired-heldout-evaluation', model: resolve(v.model), upstreamCommit: UPSTREAM_COMMIT, seed, seeds: Number(v.seeds ?? 2), fields, mode,
      inference: v.stochastic ? 'stochastic' : 'greedy', summary: summarize(rows), rows };
    const out = resolve(v.out ?? 'runs/evaluation.json'); writeJSON(out, report); console.log(JSON.stringify({ ...report, rows: undefined, out }));
    return;
  }
  if (command === 'search-demo') {
    const field = v.field ?? 'module'; checkFields([field]);
    const c = trainingCase(0, seed, [field], mode === 'both' ? 'vickrey' : mode), env = makeEnv(c, objective(v.objective, 'rank'));
    let point = env.reset(seed);
    const round = integer(v.round, 60, 'round', 1, 98 - 5 * point.observation.players.length);
    while (point.observation.round < round) point = env.step(baselineAction(point.actions)).observation!;
    console.log(JSON.stringify({ event: 'search-start', round, field, legalActions: point.actions.filter(a => a.legal).length }));
    const report = compareRootActions(env, { worlds: integer(v.worlds, 2, 'worlds', 1, 10000), seed: integer(v['search-seed'], 3_000_000, 'search-seed', 0, 0xffffffff),
      // Module strategies are public; all-external models here are explicit
      // laboratory assumptions, not access to real opponent objects.
      opponentModels: v.models?.split(',') ?? FIELDS[field] });
    const out = resolve(v.out ?? 'runs/search-demo.json'); writeJSON(out, report); console.log(JSON.stringify({ ...report, out }));
    return;
  }
  throw new Error(`Unknown command: ${command}. Run node src/cli.ts help`);
}
await main().catch(error => { console.error(error instanceof Error ? error.message : error); process.exitCode = 1; });
