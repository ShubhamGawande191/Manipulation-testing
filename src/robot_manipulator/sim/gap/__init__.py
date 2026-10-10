"""Actuation-gap toolkit (SAGE-lite): motion scripts, replay, comparison, and actuator fitting.

Motions are CSVs with a header of joint names and one row per 50 Hz tick, in radians, CMD order.
Replay runs a motion through the harness (`run_skill`), so the real arm and the sim take the same
path. `compare` measures how far apart two traces are, and `fit` tunes the sim's actuator
parameters to match a recorded real trace.
"""
