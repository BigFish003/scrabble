import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';

const root = fileURLToPath(new URL('..', import.meta.url));
const config = JSON.parse(readFileSync(resolve(root, 'upstream.json'), 'utf8'));
const upstream = resolve(root, 'upstream');
const git = (args, cwd = upstream) => execFileSync('git', args, { cwd, stdio: 'inherit' });
const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 22 || (major === 22 && minor < 18)) throw new Error('Node >=22.18 is required.');
if (!existsSync(resolve(upstream, '.git'))) {
  git(['clone', '--no-checkout', '--depth', '1', config.repository, 'upstream'], root);
  git(['fetch', '--depth', '1', 'origin', config.commit]);
  git(['checkout', '--detach', config.commit]);
}
const head = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: upstream, encoding: 'utf8' }).trim();
if (head !== config.commit) throw new Error(`Expected upstream ${config.commit}, found ${head}. Keep this dependency pinned.`);
for (const patch of [config.plannerPatch, config.competitionPatch]) {
  const path = resolve(root, patch);
  if (spawnSync('git', ['apply', '--reverse', '--check', path], { cwd: upstream }).status === 0) continue;
  git(['apply', '--check', path]);
  git(['apply', path]);
}
if (!existsSync(resolve(upstream, 'spacetimedb/wordlist.txt'))) throw new Error('Upstream dictionary is missing.');
console.log(`Ready: Node ${process.versions.node}, simulator ${head}, planner and competition adapters applied.`);
