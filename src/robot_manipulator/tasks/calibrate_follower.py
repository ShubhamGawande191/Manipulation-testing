"""Re-calibrate the SO101 follower arm using LeRobot's FeetechMotorsBus.

This script:
  1. Connects to the follower arm's serial port directly.
  2. Sets homing offsets (half-turn homings) — the arm should be
     positioned at its "zero" / home pose before running this.
  3. Interactively records the range of motion for each joint.
  4. Saves calibration to a JSON file.
  5. Converts calibration to the ROS2 follower_joints.yaml format.

IMPORTANT: The ROS2 bringup must be STOPPED before running this,
since it holds the serial port exclusively.

Usage
-----
    # 1. Stop bringup first!
    # 2. Position the arm at home (all joints at zero / natural rest)
    # 3. Run:
    python -m robot_manipulator.tasks.calibrate_follower

    # The script will prompt you to move each joint to its limits.
"""

import argparse
import json
import os
import sys
import time

try:
    from lerobot.motors.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
except ImportError:
    print("lerobot[feetech] is required. pip install lerobot[feetech]")
    sys.exit(1)

# SO101 follower motor definitions
FOLLOWER_MOTORS = {
    "shoulder_pan": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(id=3, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(id=4, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(id=5, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
    "gripper": Motor(id=6, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100),
}

DEFAULT_PORT = "/dev/ttyACM0"
OUTPUT_JSON = "/home/beelzebub/Desktop/RobotManipulator/config/follower_calibration.json"
OUTPUT_YAML = "/home/beelzebub/ros2_ws/src/so101-ros-physical-ai/so101_bringup/config/hardware/follower_joints.yaml"


def calibration_to_yaml(calibration: dict, output_path: str):
    """Convert MotorCalibration dict to the follower_joints.yaml format."""
    lines = ["joints:"]
    for name, cal in calibration.items():
        motor = FOLLOWER_MOTORS[name]
        lines.append(f"  {name}:")
        lines.append(f"    id: {motor.id}")
        lines.append(f"    homing_offset: {cal.homing_offset}")
        lines.append(f"    range_min: {cal.range_min}")
        lines.append(f"    range_max: {cal.range_max}")
        lines.append("    p_coefficient: 16")
        lines.append("    i_coefficient: 0")
        lines.append("    d_coefficient: 32")
        lines.append("    return_delay_time: 0")
        lines.append("    acceleration: 254")
        # Gripper gets extra torque limits
        if name == "gripper":
            lines.append("    max_torque_limit: 500")
            lines.append("    protection_current: 250")
            lines.append("    overload_torque: 25")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nYAML saved → {output_path}")


def calibration_to_json(calibration: dict, output_path: str):
    """Save calibration as JSON (LeRobot format)."""
    data = {}
    for name, cal in calibration.items():
        data[name] = {
            "id": cal.id,
            "drive_mode": cal.drive_mode,
            "homing_offset": cal.homing_offset,
            "range_min": cal.range_min,
            "range_max": cal.range_max,
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"JSON saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate SO101 follower arm")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port")
    parser.add_argument(
        "--skip-homing", action="store_true", help="Skip homing offset step (use current offsets)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" SO101 FOLLOWER ARM CALIBRATION")
    print("=" * 60)
    print(f"\nPort: {args.port}")
    print("\nMake sure the ROS2 bringup is STOPPED before continuing!")
    print()

    # Connect
    bus = FeetechMotorsBus(port=args.port, motors=FOLLOWER_MOTORS)
    bus.connect()
    print("Connected to motors.")

    # Verify all motors respond
    for name, motor in FOLLOWER_MOTORS.items():
        try:
            pos = bus.read("Present_Position", name, normalize=False)
            print(f"  {name} (id={motor.id}): position={pos}")
        except Exception as e:
            print(f"  ⚠ {name} (id={motor.id}): FAILED — {e}")
            print("  Fix the connection and try again.")
            bus.disconnect()
            return

    # Step 1: Homing offsets
    if not args.skip_homing:
        print("\n── STEP 1: Set Homing Offsets ──")
        print("Position the arm at its 'home' / zero pose:")
        print("  - All joints roughly centred")
        print("  - Arm pointing straight up or at natural rest")
        print("  - Gripper open")
        print()
        input("Press ENTER when the arm is at home position... ")

        # Disable torque so we can read raw positions
        bus.disable_torque()
        time.sleep(0.2)

        offsets = bus.set_half_turn_homings()
        print("\nHoming offsets set:")
        for name, offset in offsets.items():
            print(f"  {name}: {offset}")
    else:
        print("\n── Skipping homing (using existing offsets) ──")

    # Step 2: Record ranges of motion
    print("\n── STEP 2: Record Range of Motion ──")
    print("Move each joint through its FULL range of motion.")
    print("Move ALL joints, one at a time, to their min and max limits.")
    print("The script will track the extremes.")
    print()
    input("Press ENTER to start recording (move joints now)... ")

    bus.disable_torque()
    time.sleep(0.2)

    mins, maxes = bus.record_ranges_of_motion()

    print("\nRange of motion recorded:")
    print(f"  {'Joint':<16} {'Min':>6} {'Max':>6}")
    print("  " + "-" * 30)
    for name in FOLLOWER_MOTORS:
        print(f"  {name:<16} {mins[name]:>6} {maxes[name]:>6}")

    # Step 3: Build calibration
    print("\n── STEP 3: Save Calibration ──")
    # Read the current offsets from the motors
    calibration = bus.read_calibration()

    # Update range min/max from the recorded ranges
    for name in calibration:
        calibration[name].range_min = int(mins[name])
        calibration[name].range_max = int(maxes[name])

    # Write calibration back to motors
    bus.write_calibration(calibration)
    print("Calibration written to motors.")

    # Save to files
    calibration_to_json(calibration, OUTPUT_JSON)
    calibration_to_yaml(calibration, OUTPUT_YAML)

    # Show summary
    print("\n" + "=" * 60)
    print(" CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"\n{'Joint':<16} {'ID':>3} {'Offset':>8} {'Min':>6} {'Max':>6}")
    print("-" * 45)
    for name, cal in calibration.items():
        print(
            f"{name:<16} {cal.id:>3} {cal.homing_offset:>8} {cal.range_min:>6} {cal.range_max:>6}"
        )

    print("\nNext steps:")
    print("  1. Rebuild the ROS2 workspace:  cd ~/ros2_ws && colcon build")
    print("  2. Restart the bringup")
    print("  3. Restart the bridge")

    bus.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
