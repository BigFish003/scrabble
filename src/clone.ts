import { WordIndex } from '../upstream/bot-starter/src/game/dictionary.ts';

/** Preserve policy prototypes, maps, caches and PRNGs. Pinned policies use TS
 * private fields (ordinary properties), not JS #private fields or closures.
 * WordIndex is shared: its scratch workspace is safe for sequential calls;
 * separate worker threads load their own index. Do not invoke it reentrantly.
 */
export function cloneGraph<T>(value: T, seen = new Map<object, unknown>()): T {
  if (value === null || typeof value !== 'object') return value;
  if (value instanceof WordIndex) return value;
  if (seen.has(value)) return seen.get(value) as T;
  let copy: unknown;
  if (Array.isArray(value)) copy = [];
  else if (value instanceof Map) copy = new Map();
  else if (value instanceof Set) copy = new Set();
  else if (ArrayBuffer.isView(value)) return (value as unknown as { slice(): T }).slice();
  else copy = Object.create(Object.getPrototypeOf(value));
  seen.set(value, copy);
  if (value instanceof Map) {
    for (const [k, v] of value) (copy as Map<unknown, unknown>).set(cloneGraph(k, seen), cloneGraph(v, seen));
  } else if (value instanceof Set) {
    for (const v of value) (copy as Set<unknown>).add(cloneGraph(v, seen));
  } else {
    for (const key of Object.keys(value)) (copy as Record<string, unknown>)[key] = cloneGraph((value as Record<string, unknown>)[key], seen);
  }
  return copy as T;
}
