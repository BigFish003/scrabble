import { tables, reducers } from '../upstream/src/module_bindings/index.ts';

export const HOST = 'https://maincloud.spacetimedb.com';
export const DATABASE = 'scrabblebot';
const usedTables = [tables.bot, tables.botCredential, tables.tournament, tables.tournamentEntry, tables.tournamentMatch,
  tables.matchState, tables.matchParticipant, tables.auction, tables.auctionResult, tables.wordPlay, tables.myRack,
  tables.lobby, tables.lobbyMember];
const usedReducers = [reducers.registerForTournament, reducers.submitWord, reducers.submitBid, reducers.joinLobby];

interface RemoteType { Ref?: number; Array?: RemoteType; Product?: { elements: RemoteMember[] }; Sum?: { variants: RemoteMember[] } }
interface RemoteMember { name: { some: string } | string; algebraic_type: RemoteType }
interface LocalType { tag: string; value?: { elements?: LocalMember[]; variants?: LocalMember[] } | LocalType }
interface LocalMember { name?: string; algebraicType: LocalType }
export interface DatabaseSchema {
  typespace: { types: RemoteType[] }; tables: { name: string; product_type_ref: number }[];
  misc_exports: { View?: { name: string; return_type: RemoteType } }[];
  reducers: { name: string; params: { elements: RemoteMember[] } }[];
}
const norm = (name: string) => name.replaceAll('_', '').toLowerCase();
function remoteShape(t: RemoteType, types: RemoteType[]): string {
  if (t.Ref !== undefined) return remoteShape(types[t.Ref]!, types);
  if (t.Array) return 'Array[' + remoteShape(t.Array, types) + ']';
  const members = t.Product?.elements ?? t.Sum?.variants;
  if (!members) return Object.keys(t)[0]!;
  return (t.Product ? 'Product' : 'Sum') + '(' + members.map(m =>
    norm(typeof m.name === 'string' ? m.name : m.name.some) + '=' + remoteShape(m.algebraic_type, types)).join(',') + ')';
}
function localShape(t: LocalType): string {
  if (t.tag === 'Array') return 'Array[' + localShape(t.value as LocalType) + ']';
  if (t.tag !== 'Product' && t.tag !== 'Sum') return t.tag;
  const value = t.value as { elements?: LocalMember[]; variants?: LocalMember[] };
  return t.tag + '(' + (value.elements ?? value.variants ?? []).map(m => norm(m.name ?? '') + '=' + localShape(m.algebraicType)).join(',') + ')';
}

/** BSATN rows encode by position. Check names, order, types, enum variants,
 * the caller-private rack view and reducer arguments for the actual client. */
export function checkSchema(schema: DatabaseSchema): void {
  const types = schema.typespace.types, different: string[] = [];
  for (const t of usedTables) {
    const table = schema.tables.find(x => x.name === t.sourceName);
    const view = schema.misc_exports.find(x => x.View?.name === t.sourceName)?.View;
    const actual = table ? types[table.product_type_ref] : view?.return_type.Array;
    const expected = localShape({ tag: 'Product', value: t.tableDef.rowType });
    if (!actual || remoteShape(actual, types) !== expected) different.push(t.sourceName);
  }
  for (const r of usedReducers) {
    const actual = schema.reducers.find(x => x.name === r.name);
    if (!actual || remoteShape({ Product: actual.params }, types) !== localShape({ tag: 'Product', value: r.paramsType })) different.push(r.name);
  }
  if (different.length) throw new Error('Live bindings differ for ' + different.join(', ') + '. Update the client bindings before playing.');
}

export async function verifyServer() {
  const res = await fetch(HOST + '/v1/database/' + DATABASE + '/schema?version=9', { signal: AbortSignal.timeout(12000) });
  if (!res.ok) throw new Error('Schema lookup returned HTTP ' + res.status);
  checkSchema(await res.json() as DatabaseSchema);
  return { tables: usedTables.length, reducers: usedReducers.length };
}
