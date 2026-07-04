"""Fan-wave demo — sweep the SO101 base left/right at a commanded speed.

Pure joint control (no camera, no ROS2, no bridge): we oscillate the
``shoulder_pan`` joint between -amplitude and +amplitude.  "Speed" is just the
step size (degrees per command) and the delay between steps.

It holds the arm's CURRENT pose (and current grip — the fan is taped/held) and
waves the base continuously around wherever it already is.  Use ``--shift`` to
nudge the wave center left/right.  Ctrl+C to stop.

Run it:
    uv run python -m robot_manipulator.scripts.fan_wave             # med, around current pos
    uv run python -m robot_manipulator.scripts.fan_wave fast        # faster
    uv run python -m robot_manipulator.scripts.fan_wave --shift -15 # center shifted left
    uv run python -m robot_manipulator.scripts.fan_wave --amp 20    # narrower sweep
"""

from __future__ import annotations

import argparse
import time

from robot_manipulator.arm import build_follower_config

PORT = "/dev/so101_follower"
PAN = "shoulder_pan.pos"

# Gripper convention on this arm (from calibration): 100 = fully OPEN, 0 = fully
# CLOSED.  We grip the fan at ~15 (firm but with margin against overload —
# equivalent to the 0.85-closed grasp validated in pick_black_cube.py).  Lower
# = tighter.
GRIP_OPEN = 100.0
GRIP_HOLD = 15.0

# Pose the rest of the arm holds while the base waves (degrees).
# shoulder_pan and gripper are filled in at runtime; everything else stays put.
HOLD_POSE = {
    "shoulder_lift.pos": -30.0,
    "elbow_flex.pos": 60.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": GRIP_HOLD,
}

# speed name -> (degrees per step, seconds between steps)
SPEEDS = {
    "slow": (2.0, 0.040),
    "med": (4.0, 0.020),
    "fast": (7.0, 0.012),
    "1": (1.5, 0.050),
    "2": (3.0, 0.030),
    "3": (4.5, 0.020),
    "4": (6.0, 0.015),
    "5": (8.0, 0.010),
}


def _sweep_to(robot, start: float, target: float, step: float, dt: float) -> float:
    """Move shoulder_pan from start to target in `step`-degree increments."""
    direction = 1.0 if target >= start else -1.0
    pos = start
    while (direction > 0 and pos < target) or (direction < 0 and pos > target):
        pos += direction * step
        if (direction > 0 and pos > target) or (direction < 0 and pos < target):
            pos = target
        robot.send_action({PAN: pos, **HOLD_POSE})
        time.sleep(dt)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="SO101 fan-wave demo")
    parser.add_argument(
        "speed",
        nargs="?",
        help="slow / med / fast / 1-5 (default med). Fans continuously until Ctrl+C.",
    )
    parser.add_argument(
        "--amp", type=float, default=35.0, help="sweep amplitude in degrees (default 35)"
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help="shift the wave center from the current position, in degrees. "
        "Negative = left (flip the sign if it goes the wrong way).",
    )
    parser.add_argument(
        "--grip",
        type=float,
        default=None,
        help="gripper hold value, 0=closed/tight .. 100=open. "
        "Default: keep the current grip (the fan is already taped/held).",
    )
    args = parser.parse_args()

    cfg = build_follower_config(PORT, cameras=False)
    from lerobot.robots.so_follower import SOFollower

    robot = SOFollower(cfg)
    robot.connect()
    print("Connected.")

    # Hold the arm's CURRENT pose; only shoulder_pan will be driven.
    obs = robot.get_observation()
    hold = {k: v for k, v in obs.items() if k.endswith(".pos") and k != PAN}
    if args.grip is not None:
        hold["gripper.pos"] = args.grip  # else keep whatever the gripper is at now
    HOLD_POSE.clear()
    HOLD_POSE.update(hold)

    cur = obs.get(PAN, 0.0)
    center = cur + args.shift  # wave around the current position, nudged by --shift
    left = center - args.amp
    right = center + args.amp

    # Ease from the current position to the (slightly shifted) wave center.
    cur = _sweep_to(robot, cur, center, 3.0, 0.02)

    step, dt = SPEEDS.get((args.speed or "med").lower(), SPEEDS["med"])
    print(
        f"Fanning continuously at '{args.speed or 'med'}' "
        f"(center {center:.0f}°, amp ±{args.amp:.0f}°). Ctrl+C to stop."
    )
    try:
        while True:
            cur = _sweep_to(robot, cur, right, step, dt)
            cur = _sweep_to(robot, cur, left, step, dt)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        print("Returning to center and disconnecting...")
        _sweep_to(robot, cur, center, 3.0, 0.02)
        robot.disconnect()


if __name__ == "__main__":
    main()
