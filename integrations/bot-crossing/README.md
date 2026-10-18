# Bot Crossing: a lab board across repos

[Bot Crossing](https://github.com/Station-Sciences/bot-crossing) (MIT, local-only) draws every
coding-agent thread on the machine as a bot on a hex zone per repo. The **Lab board** adapter adds,
to each repo's zone, what is *next* there, including work only you can do:

| Bot | Source | State |
|---|---|---|
| 👤 your to-do | `<repo>/.colony/tasks.yaml`, `owner: you` | `?` while `todo` (data collection, measurements, live runs) |
| 🤖 agent task | same file, `owner: agent` | hammering while `doing`; quiet while `todo`/`waiting` |
| any task | same file | `!` = `failed`; confetti for 2 days after `done`, then off the map |
| ⚙ robot run | `runs/` (RobotManipulator) | latest run of each kind from the last 2 days, plus anything running |
| agent thread | Bot Crossing's own Claude Code / Codex adapters | as usual |

Tasks carry the repo's name and path, so they share the zone with that repo's agent threads.

## Add a repo

Put a `.colony/tasks.yaml` in it (copy this repo's as a template); repos under `~/Desktop`,
`~/projects`, `~/code` and `~/src` are found automatically, others go in
`~/.config/lab-colony/projects.yaml` as `projects: [/path/to/repo, ...]`.

```yaml
tasks:
  - id: ruler-check                # stable within the repo
    title: Measure wrist_flex pivot to closed fingertip
    owner: you                     # you | agent
    status: todo                   # todo | doing | waiting | done | failed
    waits_for: wristcal-live       # optional, shown on waiting tasks
    detail: one or two lines shown on the card
    link: docs/RUNBOOK.md          # optional, what Open shows
    updated: 2026-10-18
```

Agents keep it current: `doing` when they start, `done`/`waiting` when they stop, bump `updated`.

## Setup (once)

```bash
mkdir -p ~/tools && git clone https://github.com/Station-Sciences/bot-crossing ~/tools/bot-crossing
git -C ~/tools/bot-crossing checkout d05ac2f          # the commit this adapter was written against
(cd ~/tools/bot-crossing && npm ci --ignore-scripts)  # Node >= 22.13
integrations/bot-crossing/install.sh                  # symlink + register the adapter
```

## Run

```bash
uv run colony-feed --watch 3                  # terminal 1 (RobotManipulator root)
(cd ~/tools/bot-crossing && npm run dev)      # terminal 2: http://127.0.0.1:5274
```

The feed (`~/.local/state/lab-colony/threads.json`, override with `LAB_COLONY_FEED`) is written by
`src/robot_manipulator/viz/colony.py`; the adapter only reads it. In Bot Crossing's settings,
"hide dormant repos" folds away repos with nothing in the last three days.
