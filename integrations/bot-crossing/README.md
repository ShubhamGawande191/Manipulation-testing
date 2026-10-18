# Bot Crossing: the SO-101 bench as a colony

[Bot Crossing](https://github.com/Station-Sciences/bot-crossing) (MIT, local-only) draws every
coding-agent thread on the machine as a bot on a hex zone per repo. This adapter adds the bench
itself, next to the Claude Code threads working on it:

| Zone | Bots | States |
|---|---|---|
| **SO-101 real arm** | live harness reaches, surveys, gap replays | ✓ confetti = clean live run, `!` = failed / below 90 % |
| **SO-101 sim** | sim and dry-run reaches, gap replays, PPO training runs | ⚒ = being written right now |
| **SO-101 twin** | one per stage of `docs/DIGITAL-TWIN-RUNBOOK.md`'s status table | `?` = 🟡 in progress, waiting on you |

Building size = run folder size. Clicking a bot's Open shows its run folder (or the runbook).

## Setup (once)

```bash
mkdir -p ~/tools && git clone https://github.com/Station-Sciences/bot-crossing ~/tools/bot-crossing
git -C ~/tools/bot-crossing checkout d05ac2f          # the commit this adapter was written against
(cd ~/tools/bot-crossing && npm ci --ignore-scripts)  # Node >= 22.13
integrations/bot-crossing/install.sh                  # symlink + register the adapter
```

## Run

```bash
uv run colony-feed --watch 3                  # terminal 1: keep the feed fresh (repo root)
(cd ~/tools/bot-crossing && npm run dev)      # terminal 2: http://127.0.0.1:5274
```

The feed (`~/.local/state/so101-colony/threads.json`, override with `BOT_CROSSING_SO101_FEED`) is
written by `src/robot_manipulator/viz/colony.py`; the adapter only reads it. The twin zone is only
as current as the runbook's status table, so keep that table up to date.
