# Run the ScrabbleBot tournament client

Run these commands in the remote project's `scrabble-rl` folder. The dependencies and frozen models are installed on this host.

## Check for an event

```powershell
node --import tsx competition/main.ts check
```

The command compares the deployed schema with the bindings and opens a read-only WebSocket subscription. It lists events accepting registrations. Add `--tournament 123` to look up a particular ID. The check works without a team account or token. [Tournament page](https://scrabblebot.spacetimedb.com/tournament)

## Play one live casual match

Run this from `scrabble-rl` and enter your bot token at the hidden prompt.

```powershell
.\competition\play.ps1 -Casual
```

The client takes one seat in the rolling lobby. A match starts when the lobby fills or its timer expires. Leave the terminal open through the result. The client exits after that match and shows your place and score. To use the trained search policy, add `-Policy mcts`. The equivalent Node command is `node --import tsx competition/main.ts casual` with `BOT_TOKEN` set. [Live matches and lobby](https://scrabblebot.spacetimedb.com/)

## Enter and play

1. Link your SpacetimeDB identity and create or join the team whose bot will play. On the Team page choose **Generate token** and save the credential privately. [Account](https://scrabblebot.spacetimedb.com/account) · [Team](https://scrabblebot.spacetimedb.com/team)
2. Launch the command below and enter the numeric tournament ID, then the team bot token at the hidden prompt. An existing `BOT_TOKEN` environment variable also works. The launcher keeps the token out of project files and shell history.
3. Look for `Entry confirmed for bot … in tournament …`. Keep that one bot process running on an awake, connected host through all rounds. A bot that goes offline can still be seated. [Official competition guide](https://scrabblebot.spacetimedb.com/docs)

```powershell
.\competition\play.ps1
```

The client registers for the specified event, resumes an existing entry, and watches its own matches. It subscribes to each active match's participants, auctions, word plays and private rack. It reconnects and restores subscriptions after an established connection drops. It exits when the bot is eliminated or the tournament ends. Ctrl+C stops the local process. Tournament registration persists on the server.

The default is the `final` expert. On two unfamiliar first-price mixed fields the expert earned 44.8% win credit and trained search earned 27.1%. To run the frozen experimental agent explicitly, use `.\competition\play.ps1 -Policy mcts`; `-Policy calibrated` uses its learned table selector. If search starts mid-match with missing private bid history or cannot reconcile a public settlement with the observed rack and balances, the expert handles that match. [Evaluation results](../docs/MCTS-results.md)

On Linux or a service host, supply `BOT_TOKEN` through the environment and invoke `node --import tsx competition/main.ts play --tournament ID`, replacing `ID` with the actual number. `--policy` accepts the same three names.

## Set up a fresh machine

Use Node 22.18 or newer, Git and pnpm. From `scrabble-rl` run:

```powershell
node scripts/setup.mjs
pnpm install --frozen-lockfile --ignore-scripts
node --import tsx --test competition/competition.test.ts
node --import tsx competition/main.ts check
```

The pinned checkout includes the generated bindings. The preflight validates the row layout and enum order for the thirteen tables and views the client reads and the argument types for its four reducers. It runs again at startup.
