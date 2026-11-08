# Auction Scrabble MCTS experiment

**I would use the original expert for competition today.** The calibrated bidder matched it on the broad confirmation set, where it chose the expert in 90% of cases. On two new opponent lineups where it always chose search, it earned 27.08% win credit against the expert’s 44.79%.

## Training and selection

The bidder played 192 training matches, rotating seats across baseline, module, mixed and duel fields in both payment modes. Four CPU workers generated 14,016 real bidding decisions, 459,280 search simulations and 6,029 labeled positions in 3.56 minutes. The tree selected a future bid below the root 367,015 times and reached four decisions deep. A two-layer, 32-unit network received 297 Adam updates. The word planner chose all words played in real matches. [Training log](../runs/mcts-pilot/training.jsonl)

Training seeds ran from 47,001 through 47,192. The same 16 distinct validation deals, starting at 1,047,001, were used to choose among three checkpoints. The mean validation utilities were 0.623 at 64 games, 0.694 at 128 and 0.618 at 192. The 128-game checkpoint was frozen before testing. The 192-game checkpoint includes optimizer state for continuing training. [Frozen model](../models/mcts-pilot-128.json) · [Resumable checkpoint](../runs/mcts-pilot/latest.json)

## First held-out comparison

The simulator ran the trained MCTS, its untrained version and the original expert in identical seats, with identical opponent seeds and deals. Sixteen independent seed blocks starting at 7,900,000 covered four fields and both auction modes, giving 128 paired cases and 384 actual matches. First-place ties split win credit equally.

| Bidder | Win credit | Mean score | Mean rank | 95th-percentile decision |
|---|---:|---:|---:|---:|
| Original expert | 39.45% | 150.48 | 1.836 | 31.4 ms |
| MCTS before network training | 37.50% | 146.46 | 1.898 | 32.4 ms |
| Trained MCTS | 37.50% | 144.16 | 1.875 | 33.7 ms |

The trained bidder's paired win-credit difference from the expert was −1.95 percentage points, with a 95% bootstrap interval from −10.16 to +6.25. Its mean score difference was −6.31 points, with an interval from −11.45 to −1.07. Against untrained MCTS the win-credit difference was zero, with an interval from −10.94 to +12.50 percentage points. The trained bidder made 9,344 decisions, all within 72 ms, and recorded zero rejected bids or words. The largest individual win-credit deficit was in first-price duels, where the trained bidder won 3 of 16 and the expert won 8. [First held-out results and per-match data](../runs/mcts-pilot/heldout.json)

## Calibrating where to use search

The first test motivated a second version that can choose the original expert, full search or search only when its six-auction rollout can reach the end. We fitted the selector on 200 new paired cases, playing all three strategies in each case. Twenty independent seeds starting at 8,250,000 covered five fields and both formats. Selection used only the auction format, seat count and presence of module opponents. It compared within-deal changes in terminal utility and split win credit, applying a one-standard-error discount to the utility estimates. [Selection rule](MCTS.md) · [Calibration report](../runs/mcts-pilot/calibration.json)

It chose full search for six-player first-price mixed tables and the expert in the other seven observed contexts. On the 20 calibration cases for that selected table type, full search gained 15 percentage points of win credit. Those cases were used to make the choice. The gate and network were fixed before the confirmation and transfer evaluations. [Frozen selector](../models/mcts-pilot-gate.json)

## Confirmation on fresh deals

Twenty-four independent seeds starting at 9,600,000 covered five fields, both auction formats and cyclic seats. Each six-player seat appeared four times per field and format. Each duel seat appeared twelve times. The fifth field brought together five established strategies in a lineup first used for confirmation. The simulator played 720 matches in 240 paired cases. [Complete confirmation report](../runs/mcts-pilot/confirmation.json)

| Bidder | Win credit | Mean score | Mean rank | 95th-percentile decision |
|---|---:|---:|---:|---:|
| Calibrated MCTS | 41.67% | 137.44 | 1.950 | 33.7 ms |
| Original expert | 41.46% | 137.42 | 1.958 | 32.9 ms |
| Trained MCTS across all table types | 28.75% | 128.26 | 2.138 | 36.4 ms |

The calibrated agent gained 0.21 percentage points over the expert. The paired 95% interval ranged from −2.50 to +2.71 points. Its gain over using full search across all table types was 12.92 percentage points, with an interval from +6.46 to +19.58. The calibrated agent made 17,280 decisions across these fixtures, with a maximum callback time of 77.3 ms and zero rejected bids or words.

The selector chose the expert for 216 of the 240 cases and reproduced its score in every one. In the remaining 24 cases the trained search earned 58.33% win credit against the expert’s 56.25%. The interval for that subset was wide, from −25.00 to +27.08 percentage points of paired win-credit difference. The two new mixed lineups in the transfer test below exercise the selected search behavior in every case.

## Transfer to unfamiliar opponents

Both lineups were six-player first-price mixed tables, the public context in which the frozen selector chose search. Forty-eight new seeds starting at 10,400,000 rotated the learner through every seat eight times per field. This produced 96 paired cases and 288 actual matches. The calibrated and full-search variants made the same choices throughout this comparison. [Transfer report and per-case results](../runs/mcts-pilot/transfer.json)

| Opponent lineup | Cases | Search win credit | Expert win credit |
|---|---:|---:|---:|
| Shadow, sniper, valor sim, late and hoarder | 48 | 33.33% | 46.88% |
| Final, brutus sim, cheapo sim, denier and botlike-hi | 48 | 20.83% | 42.71% |
| Combined | 96 | 27.08% | 44.79% |

The paired win-credit difference was −17.71 percentage points, with a 95% interval from −27.08 to −7.81. The mean score difference was −4.73 points, with an interval from −7.60 to −2.10. Search completed 6,528 decisions with zero rejections and a 31.0 ms 95th-percentile callback. The calibration gain on one familiar mixed lineup did not transfer to either new lineup.

## Next experiment

I would test whether the simulated continuations rank the candidate bids correctly. Pick real positions where MCTS changed the expert bid. Replay each candidate on the same sampled deals, using the full word planner for subsequent learner decisions and several plausible rival strategies. Compare those rankings with the current six-auction forecast, then test the revised bidder on fresh, wholly reserved opponent lineups. This would help distinguish mistakes in rival-price forecasts from mistakes about which letters lead to playable words.

## Reproduction

All runs used Node 24.22.1, the pinned simulator revision `6d61af7daf2f32b4048740de60d75ab2eb5c9d81`, and CPU workers. The main comparisons ran in the upstream match engine. Each interval above resamples independent seed blocks 2,000 times, keeping same-seed cases together. Timings measure the offline policy callback, including word planning. The upstream parallel harness loaded the frozen policy and selector in both formats. All 21 project tests passed, including optimizer continuity on resume and indistinguishable hidden states producing identical search results. [Setup and run instructions](../README.md)

Run the transfer check from the project folder with the frozen files:

```powershell
node src/cli.ts mcts-confirm --model models/mcts-pilot-128.json --gate models/mcts-pilot-gate.json --workers 6 --seed 10400000 --seeds 48 --rotations 1 --fields mixed-rivals,mixed-modules --mode first --out runs/transfer-replay.json
```
