# Scrabble auction RL and search laboratory

A TypeScript environment for the tournament’s **auction ScrabbleBot game**, where players buy letters and cash in words. The original planner supplies the actual words and cash timing. You can train and run a multi-level information-set MCTS bidder or the original small policy-gradient bidder.

Everything lives in this folder and runs on CPU with **Node ≥22.18**. The simulator and training commands use Node directly. The competition client uses the pinned SpacetimeDB SDK and TypeScript runner installed on this remote host.

## Enter the competition

To play one live casual match, run `.\competition\play.ps1 -Casual` from this folder. Enter the team bot token at the hidden prompt and keep the remote terminal open until it shows your final score.

From this folder, run `node --import tsx competition/main.ts check` to list tournaments taking registrations. When your event opens, run `.\competition\play.ps1` in PowerShell. It asks for the tournament ID and reads the team bot token through a hidden prompt. Keep the remote process running throughout the tournament. The default bidder is the tested `final` expert. The trained MCTS and its calibrated selector are available as launcher options. [Competition setup and commands](competition/README.md)

## Start here

From this folder:

```powershell
node scripts/setup.mjs
node --test "test/*.test.ts"
node src/cli.ts bench --games 6
```

Setup downloads the pinned upstream repository and applies the planner and live-competition hooks. They record the selected bid in the planner and let the live match adapter use the trained policy. Setup is idempotent and verifies the upstream revision. [Upstream pin](upstream.json)

The project includes a frozen evaluation checkpoint in `models/`. The pilot's training log, resumable checkpoint and held-out report can also travel with the source. The downloaded `upstream/` checkout and other generated runs are gitignored. On a new machine run setup once with Git and network access. Subsequent simulation and training run offline on Windows or Linux.

## Train and run the MCTS bidder

The pilot uses four CPU workers and frozen batches of 16 full games. Each consequential bid gets up to 80 simulations. The small network learns both the search's bid preferences and the eventual match outcome. Training mixes baseline opponents, module sims, a mixed field and duels. [Search design](docs/MCTS.md) · [Training and evaluation results](docs/MCTS-results.md)

For a competitive match today, the original `final` expert has the stronger evidence. On two unseen six-player first-price mixed fields, the trained search earned 27.08% win credit and the expert earned 44.79%. The commands below run and develop the experimental bidder.

```powershell
node src/cli.ts mcts-train --games 192 --out runs/my-mcts
node src/cli.ts mcts-evaluate --model runs/my-mcts/best.json --seeds 16 --rotations 1 --fields mixed,duel,strong,heldout --seed 7900000 --out runs/my-mcts/heldout.json
```

`latest.json` contains the search configuration, network weights, optimizer state and training progress. `best.json` is selected by mean terminal utility on a separate validation set. The default training search completes a fixed simulation budget. Evaluation and registered play use a 150 ms soft budget per bid, including the initial word-planning time. The callback checks the clock between simulations and simulated auctions. A single planner call can exceed the budget.

The held-out report compares the selected checkpoint, the same search before training and the original expert using identical deals, seats and opponent seeds. `--rotations 1` cycles the learner seat across seed blocks. Use `--rotations 0` to evaluate every seat on every deal. The report records split win credit, rank, score, paired seed-block intervals, decision latency and rejected moves. Use fresh seed ranges for subsequent confirmation tests.

To continue the existing remote pilot to 1,024 total games, run:

```powershell
node src/cli.ts mcts-train --resume runs/mcts-pilot/latest.json --games 1024 --workers 4 --out runs/mcts-pilot
```

To put the selected bidder in the original simulator's parallel match harness:

```powershell
$env:SCRABBLE_MCTS_MODEL = (Resolve-Path 'models/mcts-pilot-128.json').Path
$env:SCRABBLE_MCTS_GATE = (Resolve-Path 'models/mcts-pilot-gate.json').Path
node --import ./src/register-mcts.ts upstream/bot-starter/sim/run.ts --field mcts,final,brutus-sim,valor-sim,greedy,random --matches 2 --mode vickrey --workers 2
```

The optional `SCRABBLE_MCTS_GATE` selects full search, search over the last six auctions, or the original expert using the public table type. Without that variable the harness always uses the trained search strategy. The gate records the exact model file hash. A replacement checkpoint requires its own calibration. Set `SCRABBLE_MCTS_MS` to override the per-bid budget.

To recalibrate another checkpoint and test its strategy selection on fresh deals:

```powershell
node src/cli.ts mcts-calibrate --model runs/my-mcts/best.json --gate runs/my-mcts/gate.json --seeds 20 --seed 8250000 --out runs/my-mcts/calibration.json
node src/cli.ts mcts-confirm --model runs/my-mcts/best.json --gate runs/my-mcts/gate.json --seeds 24 --seed 9600000 --out runs/my-mcts/confirmation.json
```

Choose new confirmation seeds when making another change after inspecting a result. `node src/cli.ts help` lists search depth, simulation count, hidden deals, rollout horizon and budget flags.

## Train the policy-gradient bidder

Measure throughput first, then choose a budget. A practical starting experiment is:

```powershell
node src/cli.ts train --games 512 --batch 32 --max-minutes 120 --out runs/bid-policy
```

Defaults: a fixed mixture of `base,module,mixed` fields, both payment modes, rotated learner seats, two 64-unit tanh layers, learning rate 0.0003, entropy coefficient 0.01, complete-game batches and gamma 1. This is **episodic policy gradient with a learned value baseline**, not PPO. All neural computation and Adam optimization run on CPU inside Node.

The actor starts biased toward the original `1x` bid. Its choices are skip, 0.5x, 0.75x, 1x, 1.25x, 1.5x, and absolute bids of 1, 2 or 4 coins. The absolute bids allow exploration when the planner bids zero. Bids are rounded, capped at balance **including valid planned word rewards**, and duplicates are masked. The final auction exposes only skip at all-external tables, because its newly won tile cannot be played. When module sims are present, final-auction bids remain available: denying a sim its immediate final word can change the result.

There are 186 features: the post-cash rack, current letter, inferred unseen letter counts, balance, score/leader gap, remaining auctions, original bid, immediate completion value, public prices, and up to 11 opponents' public balance/score/rack/history summaries. No true opponent starting rack, bag composition, future draw, sealed rival bid or simulator RNG enters the network.

Outputs:

| File | Meaning |
|---|---|
| `initial.json` | Model before this training run |
| `latest.json` | Last completed update, including optimizer and sampling RNG |
| `best.json` | Highest quick-validation win credit observed, once validation has run |
| `training.jsonl` | Update metrics and validation comparisons |

Quick validation runs every 128 training games, on 12 fixed cases with separate seeds. `best.json` means best among validated training checkpoints; it does **not** guarantee a gain over the original bidder. Check the paired results. Validation is a selection set; reserve fresh seeds and fields for final testing.

Resume to a total of 1,024 games:

```powershell
node src/cli.ts train --resume runs/bid-policy/latest.json --games 1024 --max-minutes 120 --out runs/bid-policy
```

Resume restores the mixture, objective, batch size, learning rate and sampling/optimizer state. The game count is cumulative. `--max-minutes` is the budget for the new invocation. Ctrl+C requests a save after the current game and partial batch update. Time limits are checked between games; the current game or validation pair can finish after the limit.

For a different objective, initialize a new experiment from a checkpoint; this resets Adam:

```powershell
node src/cli.ts train --init runs/bid-policy/latest.json --objective win --games 512 --out runs/win-policy
```

| Objective | Reward |
|---|---|
| `score` (default) | Score increase / 100, each auction |
| `win` | Terminal 1 for sole first place; ties share credit |
| `rank` | Terminal fraction of opponents not strictly ahead |
| `margin` | Terminal (own score minus highest opponent score) / 100 |

Coin expenditure is **not subtracted** from score reward. Coins constrain future decisions; playing words increases both balance and score. Choose the objective to match the actual tournament phase. These are match-level objectives, not a complete Swiss/bracket/three-game-final tournament environment.

## Evaluate on fresh seeds

```powershell
node src/cli.ts evaluate --model runs/bid-policy/best.json --seeds 10 --seed 2000000 --fields base,module,strong,duel --mode both --out runs/final-evaluation.json
```

If validation has not run, evaluate `latest.json`. Each seed block runs **every learner seat** in each selected field/mode, pairing the learned bidder with the original planner on the same deal, seat and opponent seeds. The default fields contain 6-, 6-, 6- and 2-player tables; 10 seed blocks in both modes mean 400 paired cases / 800 actual matches. Start with `--seeds 1` to check cost, then increase it.

Reports include win credit, first-place frequency including ties, mean rank, score, score margin, rejected actions, per-field results and paired differences. Confidence intervals resample whole seed blocks to keep correlated seats together. Fewer than five blocks produces no interval; even more blocks can give an overconfident result when opponents lack diversity. Greedy inference is the default; use `--stochastic` if that is how you intend to run the bot.

The policy-gradient trainer defaults to `base,module,mixed`. Other fields include `duel`, `self`, `strong` and `heldout`. The MCTS experiments add `practice` for calibration, `league` for confirmation, and `mixed-rivals` and `mixed-modules` for transfer testing. These are simulator opponent lineups.

## Search interfaces

```typescript
import { ScrabbleEnv, baselineAction } from './src/index.ts';

const env = new ScrabbleEnv({ opponents: ['random', 'greedy'], objective: 'win' });
const point = env.reset(1);
const transition = env.step(baselineAction(point.actions));

const sampledRoot = env.searchRoot(3000000, ['random', 'greedy']);
const branch = sampledRoot.clone();
branch.step(branch.decision().actions.find(a => a.legal)!.id);
```

| API | Purpose |
|---|---|
| `reset(seed)`, `decision()`, `step(actionId)` | One learner decision per simultaneous auction |
| `clone()`, `snapshot()`, `restore(snapshot)` | Exact in-memory copies, including planner/opponent memory and RNGs |
| `exportReplay()`, `ScrabbleEnv.fromReplay(...)` | JSON episode configuration and actions for offline replay |
| `searchRoot(freshSeed, assumedOpponentModels)` | Sample a plausible world using only own information/memory |
| `ScrabbleEnv.fromObservation(...)` | Start search from an independently supplied observation |
| `sampleWorld(observation, seed)` | Sample unknown starting tiles and remaining bag without replacement |
| `informationSetKey(observation)` | Key tree statistics by what the learner knows |
| `createGame`, `advance(state, jointDecisions)` | Lower-level forward model; also accepts alternate word plans |

An exact clone of the **real training environment** contains hidden simulator truth. It is useful for tests/replay, but fair search must start with `searchRoot()` or `fromObservation()`. Clone that **sampled** root for candidate branches. `debugState()` deliberately exposes simulator truth for diagnostics only.

The belief sampler conditions on tile counts, all revealed auction letters, own deal and time-ordered public word plays. It does not infer an opponent's personality or weight worlds by how likely their past bids/words were. Search opponents are assumed models warmed by replaying public history with their sampled racks. Their past sealed bids and true internal memories are unknown; modeled memory is approximate. Own planner memory is copied exactly.

Cloning preserves class prototypes, maps and independent policy RNGs. Dictionary tables/scratch are shared within a thread and called sequentially. Clone snapshots are in-memory objects, not JSON policy serialization; use the replay format for ordinary portable episodes. Replays rely on the pinned planner; its existing wall-clock search cutoffs can affect rare very large racks under different machine load.

Try the working search plumbing:

```powershell
node src/cli.ts search-demo --round 60 --worlds 8 --objective score
```

The demo samples a plausible world, clones it for each root bid, rolls each alternative to the end with the original continuation planner and averages the terminal objective. The multi-level MCTS bidder above also branches on future public observations, infers rival bidding styles and trains a policy/value network.

## Use a checkpoint in the upstream simulator

`LearnedBidPolicy` implements the original `Policy` callbacks, with the same feature/action construction as training. A registration module works in the upstream parallel harness:

```powershell
$env:SCRABBLE_MODEL = (Resolve-Path 'runs/smoke/latest.json').Path
node --import ./src/register-learned.ts upstream/bot-starter/sim/run.ts --field learned,final,brutus-sim,valor-sim,greedy,random --matches 12 --mode vickrey --workers 2
```

On another shell, set the `SCRABBLE_MODEL` environment variable to an absolute checkpoint path. The Node import propagates to worker threads. Run one simulator or training job at a time because each worker loads a dictionary. This command plays offline matches. Use the [competition launcher](competition/README.md) to enter a live event.

## Source and verification

The upstream checkout is [ManushPatel08/Scrabble_Bot_Tournament](https://github.com/ManushPatel08/Scrabble_Bot_Tournament/tree/6d61af7daf2f32b4048740de60d75ab2eb5c9d81/bot-starter), MIT licensed; see [third-party notes](THIRD_PARTY.md). Rules are pinned to that simulator, including random equal-bid ties, Vickrey reserve/second-price payments, first-price payments, module jitter, unsold tile discard and final-tile timing. This is not a claim that the current live server has been independently audited.

Tests cover complete-match parity with upstream, callback parity, RNG restoration, branch isolation, public-history beliefs, auction likelihoods, legal bid masks, multi-level tree backups, time-budget fallback, endgame decisions, policy/value gradients, gate selection and reproducible training after resume. The setup smoke runs are recorded in [VERIFICATION.md](VERIFICATION.md).
