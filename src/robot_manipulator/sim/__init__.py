"""MuJoCo simulation of the SO101 arm: model, kinematics, RL env, actuation-gap tools.

Joint commands in this package are radians, in CMD order (`model.JOINTS`).
Conversion to LeRobot's degrees/0..100 happens only in `harness/units.py`.
"""
