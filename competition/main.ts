import { parseArgs } from 'node:util';
import { DbConnection } from '../upstream/src/module_bindings/index.ts';
import { sharedIndex } from '../upstream/bot-starter/src/game/dictionary.ts';
import { attachPlayer, casualQueries, entryQueries, roomQueries, subscribe } from './client.ts';
import { DATABASE, HOST, verifyServer } from './schema.ts';
import type { Bidder } from './bidder.ts';

const help = [
  'node --import tsx competition/main.ts check [--tournament ID]',
  'node --import tsx competition/main.ts play --tournament ID [--policy expert|calibrated|mcts]',
  'node --import tsx competition/main.ts casual [--policy expert|calibrated|mcts]',
  'check reads the public server. casual uses BOT_TOKEN to play one rolling-lobby match.',
].join('\n');

async function main() {
  const [command = 'help', ...args] = process.argv.slice(2);
  if (['help', '--help', '-h'].includes(command)) { console.log(help); return; }
  if (!['check', 'play', 'casual'].includes(command)) throw new Error(help);
  const { values } = parseArgs({ args, options: { tournament: { type: 'string' }, policy: { type: 'string' } } });
  const rawId = values.tournament ?? (command === 'play' ? process.env.TOURNAMENT_ID : undefined);
  if (command === 'casual' && rawId !== undefined) throw new Error('A casual match uses the rolling lobby. Omit --tournament.');
  if (rawId !== undefined && (!/^\d{1,20}$/.test(rawId) || BigInt(rawId) > 0xffffffffffffffffn)) throw new Error('Tournament ID must be an unsigned 64-bit integer.');
  const id = rawId === undefined ? undefined : BigInt(rawId);
  const policy = (values.policy ?? 'expert') as Bidder;
  if (!['expert', 'calibrated', 'mcts'].includes(policy)) throw new Error('Unknown policy: ' + policy);
  if (command === 'play' && id === undefined) throw new Error('Supply --tournament ID or TOURNAMENT_ID.');
  const token = command === 'check' ? undefined : process.env.BOT_TOKEN?.trim();
  if (command !== 'check' && !token) throw new Error('BOT_TOKEN is required. Use the hidden prompt in competition/play.ps1.');
  const verified = await verifyServer();
  console.log(`Verified ${verified.tables} table/view bindings and ${verified.reducers} reducer signatures against ${DATABASE} on Maincloud.`);
  if (token) sharedIndex();

  let conn: DbConnection | undefined, playing: ReturnType<typeof attachPlayer> | undefined;
  await new Promise<void>((resolve, reject) => {
    let done = false;
    const timeout = setTimeout(() => finish(new Error('Timed out waiting for the server.')), 25000);
    function finish(error?: Error) {
      if (done) return;
      done = true; clearTimeout(timeout);
      process.off('SIGINT', cancel); process.off('SIGTERM', cancel);
      conn?.disconnect();
      if (error) reject(error); else resolve();
    }
    function cancel() { finish(); }
    process.once('SIGINT', cancel); process.once('SIGTERM', cancel);
    const b = DbConnection.builder().withUri(HOST).withDatabaseName(DATABASE)
      .onConnect((connected, identity) => {
        if (done) { connected.disconnect(); return; }
        if (command !== 'check') {
          clearTimeout(timeout);
          try { playing = attachPlayer(connected, identity, command === 'casual' ? 'casual' : id!, policy, finish); }
          catch (e) { finish(e instanceof Error ? e : new Error(String(e))); }
        } else {
          // The sentinel exercises the real tournament and private-view query
          // shapes without loading historical matches or acting as a team bot.
          const absent = 0xffffffffffffffffn;
          const q = id === undefined ? 'SELECT * FROM tournament' : `SELECT * FROM tournament WHERE id = ${id}`;
          subscribe(connected, [q, `SELECT * FROM bot WHERE id = ${absent}`, `SELECT * FROM bot_credential WHERE bot_id = ${absent}`,
            ...entryQueries(id ?? absent, absent), ...casualQueries(absent), ...roomQueries(absent)], () => {
            Array.from(connected.db.wordPlay.playByMatch.filter(absent));
            const rows = Array.from(connected.db.tournament.iter());
            const shown = id === undefined ? rows.filter(t => t.status.tag === 'Registration') : rows;
            for (const t of shown) console.log(`Tournament ${t.id}: ${t.status.tag}, ${t.auctionType.tag}, table size ${t.matchSize}.`);
            if (!shown.length) console.log(id === undefined ? 'No tournament is currently open for registration.' : 'Tournament ID was not found.');
            console.log('Read-only WebSocket subscription check passed.');
            finish();
          }, finish);
        }
      })
      .onConnectError((_ctx, e, next) => {
        if (next === undefined) finish(new Error('Connection failed: ' + e.message));
      })
      .onDisconnect((_ctx, e, next) => {
        if (done) return;
        playing?.disconnected();
        if (next === undefined) finish(new Error(e?.message ?? 'Server disconnected.'));
        else console.log('Connection lost. Reconnecting.');
      });
    if (token) b.withToken(token).withAutomaticReconnect().onAutomaticReconnect(() => console.log('Reconnected. Restoring subscriptions.'));
    conn = b.build();
  });
}
main().catch(e => {
  const message = e instanceof Error ? e.message : String(e);
  const token = process.env.BOT_TOKEN?.trim();
  console.error(token ? message.replaceAll(token, '[redacted]') : message);
  process.exitCode = 1;
});
