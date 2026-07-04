#!/usr/bin/env bash
# One-time GitHub backlog setup: labels + prioritized issues for the MiniLab plan.
# Prereq: `gh auth login` (needs repo scope; add `project` scope for the board:
#   gh auth refresh -s project)
# Idempotence: safe to re-run for labels (--force); issues would duplicate, so
# this script refuses to run twice (marker label on the repo).
set -euo pipefail

REPO="ShubhamGawande191/Manipulation-testing"

if gh label list -R "$REPO" | grep -q "backlog-created"; then
  echo "backlog already created (marker label present); not duplicating issues."
  exit 0
fi

# --- labels: priority (from minilab/PLAN.md) + difficulty ---
gh label create P0 -R "$REPO" --color B60205 --description "Phases 0-1: MVP + shootout" --force
gh label create P1 -R "$REPO" --color D93F0B --description "Phases 2-3" --force
gh label create P2 -R "$REPO" --color FBCA04 --description "Phase 4 multi-arm" --force
gh label create P3 -R "$REPO" --color 0E8A16 --description "Phases 5-6" --force
gh label create easy -R "$REPO" --color C2E0C6 --force
gh label create medium -R "$REPO" --color BFD4F2 --force
gh label create hard -R "$REPO" --color 5319E7 --force
gh label create hardware -R "$REPO" --color 006B75 --description "needs the physical bench" --force

issue() { gh issue create -R "$REPO" --title "$1" --label "$2" --body "$3"; }

issue "M0: Phase 0 exit — scripted key-press primitive, scored + logged" "P0,medium,hardware" \
  "See docs/EXECUTION-ROADMAP.md M0. Blocked on follower arm enumerating (direct USB port; verify with \`uv run bench-check\`). Done = scorer success from live MIDI + parquet EpisodeRecord + evidence page update."
issue "M1: task suite v0 primitives (sequence, hold, fader->CC, encoder->dCC)" "P0,medium,hardware" \
  "Closed-loop fader/encoder servo using MidiState. Each task scored + logged. Roadmap M1."
issue "M2: LeRobot recording with MIDI state vector as proprio feature" "P0,hard" \
  "Record demos incl. MidiState.vector() + optional wav; train first ACT on RTX 3060. Roadmap M2."
issue "M3: executor shootout (Blog 1) — 4 executors x chunk_len x precision" "P0,hard" \
  "Grid over exec/scheduler.py executors; tables generated from runs/*.parquet only. Roadmap M3."
issue "Fix leader arm motor 2 cable" "P1,easy,hardware" \
  "Loose cable on motor ID 2; needed for VR/teleop DAgger corrections in Phase 2."
issue "MuJoCo: MJCF MiniLab 3 + Menagerie trs_so_arm100 (PLAN Phase 5)" "P3,hard" \
  "Keys=hinges+springs, faders=sliders, encoders=free hinges; sim2real2sim key-stiffness fit. Do after Phase 1."
issue "Evaluate pixi + RoboStack as env manager (replaces uv only if ROS2-in-conda wins)" "P2,medium" \
  "pixi manages conda + PyPI (uses uv internally); would merge the ROS2 and Python envs. See TODO.md."
issue "Dependabot: keep uv.lock fresh; triage alerts weekly" "P0,easy" \
  ".github/dependabot.yml groups weekly PRs; merge only with green pytest. Full-tree refresh done 2026-07-04."

gh label create backlog-created -R "$REPO" --color EEEEEE --description "marker: backlog script ran" --force
echo "done. Create the Project board with: gh project create --owner ShubhamGawande191 --title 'SO101 MiniLab'"
