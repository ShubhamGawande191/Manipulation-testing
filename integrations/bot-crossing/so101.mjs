/**
 * Harness adapter: SO-101 robot bench (RobotManipulator repo).
 *
 * Not an agent harness: the "threads" are robot runs (harness reach, survey, gap replay, PPO
 * training) and the stages of the digital-twin runbook. The repo writes them as a small JSON
 * feed (`uv run colony-feed --watch 3`); this adapter only reads that file, cached against its
 * mtime. Opening a thread hands the OS a file:// URL to the run folder (or the runbook).
 *
 * Feed: $BOT_CROSSING_SO101_FEED, default ~/.local/state/so101-colony/threads.json
 * Install: see integrations/bot-crossing/README.md in the RobotManipulator repo.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const FEED =
  process.env.BOT_CROSSING_SO101_FEED ||
  path.join(os.homedir(), '.local', 'state', 'so101-colony', 'threads.json')

let cache = { mtimeMs: -1, threads: [] }

function readFeed() {
  const st = fs.statSync(FEED)
  if (st.mtimeMs === cache.mtimeMs) return cache.threads
  const feed = JSON.parse(fs.readFileSync(FEED, 'utf8')) // replaced atomically by the writer
  const threads = Array.isArray(feed.threads) ? feed.threads : []
  // keep only well-formed threads with our own id prefix, so a stray feed can't collide
  const ok = threads.filter((t) => t && typeof t.id === 'string' && t.id.startsWith('so101:'))
  cache = { mtimeMs: st.mtimeMs, threads: ok }
  return ok
}

async function detect() {
  try {
    fs.accessSync(FEED, fs.constants.R_OK)
    return true
  } catch {
    return false
  }
}

async function scanThreads() {
  return readFeed().map((t) => ({ ...t, ref: { path: t.ref?.path || '' } }))
}

function openThread(ref) {
  const target = ref && typeof ref.path === 'string' ? ref.path : ''
  if (!target || !path.isAbsolute(target) || !target.startsWith(os.homedir())) {
    return { ok: false, error: 'no run folder recorded for this bot' }
  }
  if (!fs.existsSync(target)) return { ok: false, error: `${target} no longer exists` }
  return { ok: true, url: pathToFileURL(target).href }
}

function newSession() {
  return {
    ok: false,
    error: 'robot runs start from the repo CLI (uv run harness …), not from the colony',
  }
}

export default {
  id: 'so101',
  name: 'SO-101 bench',
  detect,
  scanThreads,
  openThread,
  newSession,
}
