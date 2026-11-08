# Third-party sources

The setup script downloads [ManushPatel08/Scrabble_Bot_Tournament](https://github.com/ManushPatel08/Scrabble_Bot_Tournament) at revision `6d61af7daf2f32b4048740de60d75ab2eb5c9d81`. Its root license is MIT, copyright 2026 Clockwork Labs. The repository and license remain intact under `upstream/`.

`src/game.ts` adapts that repository's `bot-starter/sim/engine.ts` into a step-wise explicit-state model. `src/rng.ts` adapts its `bot-starter/src/game/rng.ts` to expose serializable stream state. Other modules import its dictionary, letters, planner, policy types and opponent factories. The competition client also uses the checked-in web SDK bindings and live match adapter. `patches/submitted-bid.patch` records the selected bid in the planner. `patches/competition-bid.patch` adds an optional live bid selector and current-auction checks.

The following notice applies to the upstream-derived portions:

```text
MIT License

Copyright (c) 2026 Clockwork Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
