/**
 * Harness adapter: Lab board — what is next in each repo, plus the latest robot runs.
 *
 * Not an agent harness: the "threads" are tasks from each repo's `.colony/tasks.yaml` (yours and
 * the agents') and the latest robot runs. They carry the repo's own name and path, so they land
 * in the same zone as that repo's agent threads. The RobotManipulator repo writes everything as
 * one JSON feed (`uv run colony-feed --watch 3`); this adapter only reads that file, cached
 * against its mtime. Opening a bot hands the OS a file:// URL to the task's link or run folder.
 *
 * Feed: $LAB_COLONY_FEED, default ~/.local/state/lab-colony/threads.json
 * Install: integrations/bot-crossing/README.md in the RobotManipulator repo.
 */
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const FEED =
  process.env.LAB_COLONY_FEED ||
  path.join(os.homedir(), '.local', 'state', 'lab-colony', 'threads.json')

let cache = { mtimeMs: -1, threads: [] }

function readFeed() {
  const st = fs.statSync(FEED)
  if (st.mtimeMs === cache.mtimeMs) return cache.threads
  const feed = JSON.parse(fs.readFileSync(FEED, 'utf8')) // replaced atomically by the writer
  const threads = Array.isArray(feed.threads) ? feed.threads : []
  // keep only well-formed threads with our own id prefix, so a stray feed can't collide
  const ok = threads.filter((t) => t && typeof t.id === 'string' && t.id.startsWith('lab:'))
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
    return { ok: false, error: 'nothing to open for this bot' }
  }
  if (!fs.existsSync(target)) return { ok: false, error: `${target} no longer exists` }
  return { ok: true, url: pathToFileURL(target).href }
}

function newSession() {
  return {
    ok: false,
    error: 'tasks live in <repo>/.colony/tasks.yaml; robot runs start from the repo CLI',
  }
}

export default {
  id: 'lab',
  name: 'Lab board',
  detect,
  scanThreads,
  openThread,
  newSession,
}
