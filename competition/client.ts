import type { Identity } from 'spacetimedb';
import { DbConnection, type SubscriptionHandle } from '../upstream/src/module_bindings/index.ts';
import { sharedIndex } from '../upstream/bot-starter/src/game/dictionary.ts';
import { LiveBot, type LiveConn } from '../upstream/bot-starter/src/live/adapter.ts';
import { competitionSelector, type Bidder } from './bidder.ts';

export const entryQueries = (tournament: bigint, bot: bigint) => [
  `SELECT * FROM tournament_entry WHERE tournament_id = ${tournament} AND bot_id = ${bot}`,
  `SELECT t.* FROM tournament_match t JOIN match_participant p ON t.match_id = p.match_id WHERE t.tournament_id = ${tournament} AND p.bot_id = ${bot}`,
  `SELECT s.* FROM match_state s JOIN tournament_match t ON s.id = t.match_id WHERE t.tournament_id = ${tournament}`,
];
export const roomQueries = (id: bigint) => ['match_participant', 'auction', 'auction_result', 'word_play', 'my_rack']
  .map(table => `SELECT * FROM ${table} WHERE match_id = ${id}`);
export const casualQueries = (bot: bigint) => [
  `SELECT l.* FROM lobby l JOIN lobby_member p ON l.id = p.lobby_id WHERE p.bot_id = ${bot}`,
  `SELECT s.* FROM match_state s JOIN match_participant p ON s.id = p.match_id WHERE p.bot_id = ${bot}`,
];

export function entryAction(status: string | undefined, entry: { eliminated: boolean } | undefined) {
  if (!status) return 'missing';
  if (status === 'Ended' || status === 'ForceEnded' || entry?.eliminated) return 'finished';
  if (entry) return 'play';
  return status === 'Registration' ? 'register' : 'closed';
}
export function subscribe(conn: DbConnection, queries: string[], applied: () => void, fail: (e: Error) => void): SubscriptionHandle {
  return conn.subscriptionBuilder().onApplied(applied).onError(ctx => {
    const reason = ctx.event instanceof Error ? ctx.event.message : String(ctx.event ?? 'server rejected subscription');
    fail(new Error('Subscription failed (' + queries[0] + '): ' + reason));
  }).subscribe(queries);
}

/** Read the full room atomically before passing opens to the bot. */
export function attachPlayer(conn: DbConnection, identity: Identity, target: bigint | 'casual', policy: Bidder, finish: (error?: Error) => void) {
  const rooms = new Map<bigint, { ready: boolean; handle?: SubscriptionHandle }>();
  let baseReady = false, listingReady = false, stopped = false, subscribing = false, registering = false;
  let botId: bigint | undefined, readyEntry = false, scheduled = false, generation = 0;
  let casualLobby: bigint | undefined, previousLobbies: Set<bigint> | undefined;
  const live = new LiveBot(conn as unknown as LiveConn, {
    name: 'competition', index: sharedIndex(), autoLobby: false, autoRegister: false, logRounds: false,
    log: (...args) => console.log(...args),
    // Queued opens must drain the adapter's counter during a disconnect. The
    // readiness callback below checks the live connection before any decision.
    defer: cb => setImmediate(() => { if (!stopped) cb(); }),
    participantsReady: id => conn.isActive && !stopped && baseReady && listingReady && readyEntry && rooms.get(id)?.ready === true,
    bidSelector: competitionSelector(policy, {
      result: id => conn.db.auctionResult.auctionId.find(id) ?? undefined,
      words: id => conn.db.wordPlay.playByMatch.filter(id),
      closesAt: id => conn.db.auction.id.find(id)?.closesAt,
    }),
  });
  const stop = (e?: Error) => { if (!stopped) { stopped = true; finish(e); } };
  function watch(id: bigint) {
    if (rooms.has(id)) return;
    const room: { ready: boolean; handle?: SubscriptionHandle } = { ready: false };
    rooms.set(id, room);
    room.handle = subscribe(conn, roomQueries(id), () => {
      room.ready = true;
      refreshSoon();
    }, stop);
  }
  function playRoom(id: bigint) {
    watch(id);
    if (!rooms.get(id)?.ready) return;
    const mine = Array.from(conn.db.matchParticipant.iter()).find(p => p.matchId === id && p.botId === botId);
    if (mine) { live.onParticipantInsert(mine); live.onMatchReady(id); }
  }
  function refreshCasual() {
    // Both listing queries return only this bot's rows. Keep the chosen
    // lobby and the pre-join snapshot across reconnects so a match that
    // finishes offline can never cause a second signup.
    const mine = Array.from(conn.db.lobby.iter());
    if (casualLobby === undefined) {
      const assigned = mine.find(l => l.status.tag === 'Open' ||
        (l.resolvedMatchId != null && conn.db.matchState.id.find(l.resolvedMatchId)?.status.tag === 'Running')) ??
        mine.find(l => previousLobbies && !previousLobbies.has(l.id));
      if (assigned) {
        casualLobby = assigned.id;
        console.log(`Joined casual lobby ${casualLobby}.`);
      } else {
        if (Array.from(conn.db.matchState.iter()).some(m => m.status.tag === 'Running'))
          return stop(new Error('This bot is already seated in a match outside the casual lobby.'));
        if (!registering) {
          previousLobbies ??= new Set(mine.map(l => l.id));
          registering = true;
          const requestedOn = generation;
          console.log('Requesting one casual match.');
          conn.reducers.joinLobby({}).catch(e => {
            if (requestedOn === generation && conn.isActive) stop(new Error('Could not join the casual lobby: ' + String(e)));
          });
        }
        return;
      }
    }
    const lobby = conn.db.lobby.id.find(casualLobby);
    if (!lobby) return stop(new Error('The selected casual lobby is no longer available.'));
    if (lobby.status.tag === 'Cancelled') return stop(new Error('The lobby was cancelled before a match started.'));
    if (lobby.resolvedMatchId == null) return;
    const match = conn.db.matchState.id.find(lobby.resolvedMatchId);
    if (!match) return stop(new Error('The lobby resolved without seating this bot.'));
    if (match.status.tag === 'Ended') {
      // Reconnect snapshots can skip the usual match-ended callback.
      if (!live.finished.some(r => r.matchId === match.id)) {
        watch(match.id);
        if (!rooms.get(match.id)?.ready) return;
        const players = Array.from(conn.db.matchParticipant.iter()).filter(p => p.matchId === match.id);
        const player = players.find(p => p.botId === botId);
        if (!player) return stop(new Error('Final standings are unavailable for this bot.'));
        const place = 1 + players.filter(p => p.score > player.score).length;
        console.log(`Match ${match.id} ended. Place ${place}/${players.length}, score ${player.score}.`);
      }
      return stop();
    }
    if (match.status.tag === 'Running') { readyEntry = true; playRoom(match.id); }
  }
  function refresh() {
    if (stopped || !conn.isActive || !baseReady || !listingReady || botId === undefined) return;
    if (target === 'casual') return refreshCasual();
    const tournament = conn.db.tournament.id.find(target);
    const entry = Array.from(conn.db.tournamentEntry.iter()).find(e => e.tournamentId === target && e.botId === botId);
    const action = entryAction(tournament?.status.tag, entry);
    if (action === 'missing' || action === 'closed') return stop(new Error(action === 'missing'
      ? 'Tournament ' + target + ' does not exist.' : 'Registration is closed and this bot has no entry in tournament ' + target + '.'));
    if (action === 'finished') { console.log('Tournament completed or bot eliminated.'); return stop(); }
    if (action === 'register') {
      if (!registering) {
        registering = true;
        const requestedOn = generation;
        console.log('Requesting entry for tournament ' + target + '.');
        conn.reducers.registerForTournament({ tournamentId: target }).catch(e => {
          if (requestedOn !== generation || !conn.isActive || !baseReady || !listingReady) return;
          if (Array.from(conn.db.tournamentEntry.iter()).some(row => row.tournamentId === target && row.botId === botId)) refreshSoon();
          else stop(new Error('Registration failed: ' + String(e)));
        });
      }
      return;
    }
    if (!readyEntry) { console.log('Entry confirmed for bot ' + botId + ' in tournament ' + target + '. Waiting for its matches.'); readyEntry = true; }
    for (const tm of conn.db.tournamentMatch.iter()) {
      if (tm.tournamentId !== target) continue;
      const m = conn.db.matchState.id.find(tm.matchId);
      if (!m) continue;
      if (m.status.tag === 'Ended') {
        const room = rooms.get(m.id);
        if (room) { rooms.delete(m.id); room.handle?.unsubscribe(); live.myMatches.delete(m.id); live.sessions.delete(m.id); }
      } else if (m.status.tag === 'Running') playRoom(m.id);
    }
  }
  function refreshSoon() {
    if (scheduled || stopped) return;
    scheduled = true;
    setImmediate(() => { scheduled = false; try { refresh(); } catch (e) { stop(e instanceof Error ? e : new Error(String(e))); } });
  }
  conn.db.tournament.onUpdate(refreshSoon);
  conn.db.tournamentEntry.onInsert(refreshSoon);
  conn.db.tournamentEntry.onUpdate(refreshSoon);
  conn.db.tournamentMatch.onInsert(refreshSoon);
  if (target === 'casual') { conn.db.lobby.onInsert(refreshSoon); conn.db.lobby.onUpdate(refreshSoon); }
  conn.db.matchState.onInsert(refreshSoon);
  conn.db.matchState.onUpdate((_ctx, before, after) => { live.onMatchUpdate(before, after); refreshSoon(); });
  conn.db.matchParticipant.onInsert((_ctx, p) => live.onParticipantInsert(p));
  conn.db.auction.onInsert((_ctx, a) => { if (rooms.get(a.matchId)?.ready) live.onAuctionInsert(a); });
  conn.db.auctionResult.onInsert((_ctx, r) => live.onResultInsert(r));
  conn.db.wordPlay.onInsert((_ctx, w) => live.onWordInsert(w));
  const baseQueries = ['SELECT * FROM bot_credential', 'SELECT * FROM bot'];
  if (target !== 'casual') baseQueries.push(`SELECT * FROM tournament WHERE id = ${target}`);
  subscribe(conn, baseQueries, () => {
    const credential = conn.db.botCredential.identity.find(identity);
    if (!credential) return stop(new Error('BOT_TOKEN does not belong to a team bot. Generate a bot credential on the Team page.'));
    if (botId !== undefined && botId !== credential.botId) return stop(new Error('Bot identity changed during reconnect.'));
    botId = credential.botId; baseReady = true;
    if (!subscribing) {
      subscribing = true;
      console.log('Authenticated as ' + (conn.db.bot.id.find(botId)?.name ?? 'bot') + ' (id ' + botId + '), policy ' + policy + '.');
      const queries = target === 'casual' ? casualQueries(botId) : entryQueries(target, botId);
      subscribe(conn, queries, () => { listingReady = true; refreshSoon(); }, stop);
    }
    live.onApplied(identity);
    refreshSoon();
  }, stop);
  return { disconnected() {
    generation++;
    baseReady = false; listingReady = false; readyEntry = false; registering = false;
    for (const room of rooms.values()) room.ready = false;
  } };
}
