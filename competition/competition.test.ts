import assert from 'node:assert/strict';
import { test } from 'node:test';
import { setImmediate as nextTurn } from 'node:timers/promises';
import type { Identity } from 'spacetimedb';
import type { DbConnection } from '../upstream/src/module_bindings/index.ts';
import { FakeServer, ME_IDENTITY } from '../upstream/bot-starter/test/fakeserver.ts';
import { sharedIndex } from '../upstream/bot-starter/src/game/dictionary.ts';
import { LiveBot } from '../upstream/bot-starter/src/live/adapter.ts';
import { competitionSelector } from './bidder.ts';
import { attachPlayer, casualQueries, entryAction, roomQueries } from './client.ts';

function table<T>(rows: T[], key: (r: T) => bigint) {
  const inserts: (() => void)[] = [], updates: (() => void)[] = [];
  return { iter: () => rows, id: { find: (id: bigint) => rows.find(r => key(r) === id) }, onInsert: (f: () => void) => inserts.push(f),
    onUpdate: (f: () => void) => updates.push(f), signalInsert: () => inserts.forEach(f => f()), signalUpdate: () => updates.forEach(f => f()) };
}

test('tournament entry is scoped to the requested ID and confirmed through the subscription', async () => {
  const wanted = 61n, other = 63n, me = 24n, identity = {} as Identity;
  const tournaments = [wanted, other].map(id => ({ id, status: { tag: 'Registration' } }));
  const entries: { tournamentId: bigint; botId: bigint; eliminated: boolean }[] = [];
  const asked: bigint[] = [], listening: { queries: string[]; applied: () => void }[] = [];
  let lobby = 0, ended = 0;
  const empty = table([] as { id: bigint }[], r => r.id);
  const db = {
    tournament: table(tournaments, t => t.id), tournamentEntry: table(entries, e => e.tournamentId),
    tournamentMatch: table([{ tournamentId: other, matchId: 99n }, { tournamentId: wanted, matchId: 41n }], t => t.matchId),
    matchState: table([41n, 99n].map(id => ({ id, status: { tag: 'Running' } })), m => m.id),
    botCredential: { iter: () => [], identity: { find: () => ({ botId: me }) } }, bot: empty,
    matchParticipant: empty, auction: empty, auctionResult: empty, wordPlay: empty,
  };
  const conn = { db, isActive: true, reducers: {
    registerForTournament: async (r: { tournamentId: bigint }) => { asked.push(r.tournamentId); }, joinLobby: () => { lobby++; },
  }, subscriptionBuilder() {
    let applied: () => void;
    return { onApplied(fn: () => void) { applied = fn; return this; }, onError() { return this; },
      subscribe(queries: string[]) { listening.push({ applied, queries }); return { unsubscribe() {} }; } };
  } };
  const ctl = attachPlayer(conn as unknown as DbConnection, identity, wanted, 'expert', e => { assert.equal(e, undefined); ended++; });
  listening[0]!.applied(); listening[1]!.applied(); await nextTurn();
  db.tournament.signalUpdate(); await nextTurn();
  assert.deepEqual(asked, [wanted]);
  assert.equal(listening.length, 2);
  entries.push({ tournamentId: wanted, botId: me, eliminated: false });
  db.tournamentEntry.signalInsert(); await nextTurn();
  assert.deepEqual(listening[2]!.queries, roomQueries(41n));
  ctl.disconnected();
  listening[0]!.applied(); listening[1]!.applied(); listening[2]!.applied(); await nextTurn();
  assert.deepEqual(asked, [wanted]);
  assert.equal(listening.length, 3);
  entries[0]!.eliminated = true; db.tournamentEntry.signalUpdate(); await nextTurn();
  assert.equal(ended, 1); assert.equal(lobby, 0);
  assert.equal(entryAction('Swiss', undefined), 'closed');
  assert.equal(entryAction(undefined, undefined), 'missing');
});

for (const offlineAtSeating of [false, true]) {
  test(`casual mode requests one seat and finishes its match, offline at seating = ${offlineAtSeating}`, async () => {
    const me = 24n, id = 42n;
    const previous = { id: 8n, status: { tag: 'Resolved' }, resolvedMatchId: 40n };
    const lobbies: { id: bigint; status: { tag: string }; resolvedMatchId?: bigint }[] = [previous];
    const matches = [{ id: 40n, status: { tag: 'Ended' } }];
    const empty = table([] as { id: bigint }[], r => r.id);
    const db = {
      bot: empty, botCredential: { identity: { find: () => ({ botId: me }) } }, tournament: empty, tournamentEntry: empty,
      tournamentMatch: empty, wordPlay: empty, auction: empty, auctionResult: empty,
      lobby: table(lobbies, l => l.id), matchState: table(matches, m => m.id),
      matchParticipant: table([{ id: 1n, botId: me, matchId: id, score: 73n }], p => p.id),
    };
    const subs: { applied: () => void; queries: string[] }[] = [];
    let joins = 0, completed = 0;
    const conn = { isActive: true, db, reducers: { joinLobby: async () => { joins++; } }, subscriptionBuilder() {
      let applied: () => void;
      return { onApplied(fn: () => void) { applied = fn; return this; }, onError() { return this; },
        subscribe(queries: string[]) { subs.push({ applied, queries }); return { unsubscribe() {} }; } };
    } };
    const ctrl = attachPlayer(conn as unknown as DbConnection, {} as Identity, 'casual', 'expert', err => {
      assert.equal(err, undefined); completed++;
    });
    subs[0]!.applied(); subs[1]!.applied(); await nextTurn();
    assert.deepEqual(subs[1]!.queries, casualQueries(me)); assert.equal(joins, 1);
    db.lobby.signalUpdate(); await nextTurn(); assert.equal(joins, 1);
    const current = { id: 9n, status: { tag: 'Open' }, resolvedMatchId: undefined as bigint | undefined };
    if (offlineAtSeating) { ctrl.disconnected(); }
    lobbies.push(current);
    if (!offlineAtSeating) { db.lobby.signalInsert(); await nextTurn(); }
    current.status.tag = 'Resolved'; current.resolvedMatchId = id;
    const match = { id, status: { tag: offlineAtSeating ? 'Ended' : 'Running' } };
    matches.push(match);
    if (offlineAtSeating) { subs[1]!.applied(); subs[0]!.applied(); }
    db.matchState.signalInsert(); db.lobby.signalUpdate(); await nextTurn();
    assert.deepEqual(subs[2]!.queries, roomQueries(id));
    subs[2]!.applied(); await nextTurn();
    if (!offlineAtSeating) {
      assert.equal(completed, 0);
      match.status.tag = 'Ended'; db.lobby.signalUpdate(); await nextTurn();
    }
    assert.equal(completed, 1); assert.equal(joins, 1); assert.equal(subs.length, 3);
  });
}

test('frozen MCTS adjusts live bids across a whole match with reversed callback order and real word timing', () => {
  const server = new FakeServer(42n, 'FirstPrice', ['Aggressive', 'Cheapskate', 'ValueBidder', 'Cheapskate', 'Aggressive'], 22);
  server.auctionFirst = true; server.auctionMs = 500n;
  const close = new Map<bigint, bigint>(), when = new Map<bigint, bigint>();
  let clock = 0n, decisions = 0, overrides = 0;
  const recordWords = (time: bigint) => {
    for (const w of server.db.wordPlay.iter()) if (!when.has(w.id)) when.set(w.id, time);
  };
  const factory = competitionSelector('mcts', {
    result: id => {
      const row = server.db.auctionResult.auctionId.find(id);
      return row ? { ...row, closedAt: { microsSinceUnixEpoch: close.get(id)! } } : undefined;
    },
    words: () => Array.from(server.db.wordPlay.iter(), w => ({ ...w, playedAt: { microsSinceUnixEpoch: when.get(w.id)! } })),
    closesAt: () => ({ microsSinceUnixEpoch: BigInt(Date.now() + 1000) * 1000n }),
  })!;
  const queue: (() => void)[] = [], logs: string[] = [];
  const bot = new LiveBot(server, { name: 'competition', index: sharedIndex(), autoRegister: false, autoLobby: false,
    logRounds: false, log: (...args) => logs.push(args.join(' ')), defer: cb => queue.push(cb),
    bidSelector: (...args) => {
      const decide = factory(...args)!;
      return (view, plan, id, start) => {
        const bid = decide(view, plan, id, start);
        decisions++; if (bid !== plan.bid) overrides++; return bid;
      };
    },
  });
  const drain = () => { while (queue.length) queue.shift()!(); };
  server.bots_.push(bot);
  bot.onApplied(ME_IDENTITY); server.start(1); drain();
  while (server.running) {
    clock += 10n; recordWords(clock - 1n);
    server.tick();
    const r = Array.from(server.db.auctionResult.iter()).at(-1)!;
    close.set(r.auctionId, clock); recordWords(clock); drain();
  }
  assert.deepEqual(server.errors, []); assert.equal(bot.errors, 0);
  assert.deepEqual(logs.filter(line => line.includes('continuing with expert bidding')), []);
  assert.equal(decisions, 68); assert.ok(overrides > 0);
  assert.equal(bot.finished.length, 1);
});

test('a missing settlement switches live MCTS to the expert for the rest of the match', () => {
  const server = new FakeServer(42n, 'Vickrey', ['Aggressive', 'Cheapskate', 'ValueBidder', 'Cheapskate', 'Aggressive'], 23);
  server.auctionMs = 250n;
  const logs: string[] = [], queue: (() => void)[] = [];
  let requestedSettlements = 0;
  const bot = new LiveBot(server, {
    name: 'competition', index: sharedIndex(), autoRegister: false, autoLobby: false, logRounds: false,
    log: (...args) => logs.push(args.join(' ')), defer: cb => queue.push(cb),
    bidSelector: competitionSelector('mcts', {
      result: () => { requestedSettlements++; return undefined; }, words: () => [],
      closesAt: () => ({ microsSinceUnixEpoch: BigInt(Date.now() + 250) * 1000n }),
    }),
  });
  server.bots_.push(bot);
  bot.onApplied(ME_IDENTITY); server.start(1);
  while (server.running) {
    while (queue.length) queue.shift()!();
    server.tick();
  }
  assert.equal(requestedSettlements, 1);
  assert.equal(logs.filter(line => line.includes('continuing with expert bidding (previous settlement unavailable)')).length, 1);
  assert.ok(server.bidsAtClose.size > 2);
  assert.deepEqual(server.errors, []); assert.equal(bot.errors, 0);
  assert.equal(bot.finished.length, 1);
});
