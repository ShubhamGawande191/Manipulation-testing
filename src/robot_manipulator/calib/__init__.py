"""Calibration checks and (later) auto-calibration for LeRobot SO101 arms.

Conventions come from LeRobot's Feetech bus (`lerobot/motors/feetech/feetech.py`):
`Present = Actual - Homing_Offset`, and `DEGREES` zero is the midpoint of the calibrated range.
"""
