# Verification on 2026-11-07

Host: Windows, Node **24.22.1**, Git **2.47.0**. npm is absent; all commands below ran directly through Node. Upstream revision: `6d61af7daf2f32b4048740de60d75ab2eb5c9d81`, with only the submitted-bid adapter patch.

## Automated checks

```powershell
node --test "test/*.test.ts"
```

**14 tests passed**, zero failures; final run approximately 27 seconds. Coverage includes:

- Seeded random streams match upstream and restore exactly.
- Complete-match forward-model parity across both auction modes, external policies, module sims and strong policies.
- Original learner-bidder results match upstream.
- Changed learned bids give matching results through the RL environment and the original callback interface.
- In-memory clones, snapshots/restores and JSON episode replays preserve tested behavior.
- Observations/information-set keys exclude hidden simulator state.
- Sampled worlds conserve tiles and respect time-ordered public word evidence.
- Duplicate/affordability masks and zero-baseline exploration work.
- Final denial bids can prevent a module sim's final word; external winners cannot play the final tile.
- Terminal rank/win rewards and unpenalized score increments match the specified objectives.
- Neural gradients match finite differences; learning increases a rewarded toy action's probability.
- Network/checkpoint round trips preserve predictions and subsequent Adam updates.

## Pipeline smoke runs

| Check | Observed result |
|---|---|
| Setup | Downloaded pinned upstream; adapter applied; no npm packages needed |
| Benchmark | Six matches across `base,module,mixed` and both modes: 408 learner decisions in 5.96 s, approximately **68.5 decisions/s** |
| Initial training | 12 complete games, 816 decisions, three updates; checkpoints and validation log produced |
| Resume | Loaded the 12-game checkpoint and completed games 13–16 with another 272 decisions/update |
| Fresh-seed paired evaluation | 16 paired cases / 32 actual matches, `base,duel`, both modes, all learner seats, seed 2,000,000 |
| Upstream parallel harness | Loaded checkpoint through `--import`, ran two workers and two matches successfully |
| Search plumbing | At auction 60, compared eight root bids over two sampled worlds and full terminal rollouts in approximately 0.40 s, excluding initial match setup |

The initial short-trained model's greedy policy matched the original bidder on the six validation cases and all 16 fresh-seed paired cases. Both had 87.5% win credit in that small held-out set; the paired improvement was **zero**. There were no rejected learner bids or words. One seed block is insufficient for confidence intervals or a strength claim.

These initial runs verified the environment, optimization, persistence and integration. The original short-trained bidder matched the expert. Strong duels were slower in the local benchmark. A later experiment added multi-level information-set MCTS, opponent-style inference and search-guided training. Its design is documented in [Searching for the winning bid](docs/MCTS.md).

Generated local artifacts:

- `runs/benchmark.json`
- `runs/smoke/initial.json`, `runs/smoke/latest.json`, `runs/smoke/best.json`, `runs/smoke/training.jsonl`
- `runs/resume-smoke/latest.json`, `runs/resume-smoke/training.jsonl`
- `runs/smoke-evaluation.json`
- `runs/search-demo.json`

The simulator uses the upstream auction-window abstraction. The source planner retains its original wall-clock partition cutoffs, so rare oversized-rack replays may differ under substantially different load. The explicit game transition and stored random streams themselves do not depend on wall time.

## Competition client

On November 7, 2026, the tournament launcher verified 11 table and view bindings, three reducer signatures and the read-only WebSocket subscriptions against the deployed `scrabblebot` database. The server listed no events taking registrations. All 35 local tests passed across the simulator, upstream live adapter and tournament client. They covered signup and reconnects, both auction formats, 250 ms rounds and the frozen MCTS bidding through a 68-auction match. In a separate match with a missing settlement, the expert took over and finished with zero rejected moves. Live tournament signup still awaits an open event. [Competition commands](competition/README.md)

The one-match casual launcher passed two additional tests for normal seating and reconnecting after a match ends. Its live preflight verified 13 table and view bindings and four reducers. The expert then played casual match 45168 as `scrabblefish` and placed second of six with 95 points. The server recorded SQUATLY for 57, COADMIT for 36 and AR for 2. The process exited after that match. [Official match record](https://scrabblebot.spacetimedb.com/matches/45168)
