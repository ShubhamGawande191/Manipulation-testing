"""Touch-table diagnostic for the SO101 arm.

Moves the arm to several positions and slowly lowers it until the user
confirms the gripper is touching the table.  Records the joint values
at each touch point so we can compute correct lowering offsets for
pick-and-place.

Usage
-----
    python -m robot_manipulator.tasks.touch_table

The script will:
  1. Move to a position at hover height.
  2. Slowly lower the arm (shoulder_lift++, elbow_flex--).
  3. User presses ENTER when gripper touches the table,
     or 's' to skip that position.
  4. After all positions, prints a summary + saves results to JSON.
"""

import argparse
import asyncio
import json
import logging
import sys

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("touch_table")

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
JOINT_LIMITS = {
    "shoulder_pan": (-1.5, 1.5),
    "shoulder_lift": (-2.0, 0.5),
    "elbow_flex": (0.3, 2.5),
    "wrist_flex": (-1.0, 1.5),
    "wrist_roll": (-1.5, 1.5),
    "gripper": (-0.4, 0.8),
}
HOME = [0.0, -1.57, 1.57, 0.0, 0.0, -0.4]


def _clamp(joints: list[float]) -> list[float]:
    return [
        max(lo, min(hi, v)) for v, (lo, hi) in zip(joints, (JOINT_LIMITS[n] for n in JOINT_NAMES))
    ]


class BridgeClient:
    def __init__(self, url="ws://localhost:8765"):
        self._url, self._ws = url, None

    async def connect(self):
        self._ws = await websockets.connect(self._url)

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def _cmd(self, action, params=None, timeout=15):
        await self._ws.send(json.dumps({"command": {"action": action, "parameters": params or {}}}))
        r = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=timeout))
        return r["telemetry"]["data"]

    async def state(self):
        return await self._cmd("arm.get_state")

    async def move(self, joints):
        return await self._cmd("arm.move_joints", {"joints": _clamp(joints)})

    async def home(self):
        return await self._cmd("arm.home")

    async def grip(self, pos):
        return await self._cmd("arm.gripper", {"position": pos})


# ── Test positions (subset of scan grid) ───────────────────────────
# (pan, shoulder_lift, elbow_flex, wrist_flex) — hover height configs
TEST_POSITIONS = [
    # Centre near
    {"label": "centre-near", "joints": [0.0, -1.30, 1.30, 0.50, 0.0, -0.4]},
    # Centre mid
    {"label": "centre-mid", "joints": [0.0, -1.05, 1.50, 0.25, 0.0, -0.4]},
    # Centre far
    {"label": "centre-far", "joints": [0.0, -0.80, 1.70, 0.00, 0.0, -0.4]},
    # Left near
    {"label": "left-near", "joints": [-0.4, -1.30, 1.30, 0.50, 0.0, -0.4]},
    # Right near
    {"label": "right-near", "joints": [0.4, -1.30, 1.30, 0.50, 0.0, -0.4]},
    # Left mid
    {"label": "left-mid", "joints": [-0.4, -1.05, 1.50, 0.25, 0.0, -0.4]},
    # Right mid
    {"label": "right-mid", "joints": [0.4, -1.05, 1.50, 0.25, 0.0, -0.4]},
]

# Lowering step size (radians per tick)
SL_STEP = 0.03  # shoulder_lift: positive = lower
EF_STEP = -0.02  # elbow_flex: negative = lower
WF_STEP = 0.01  # slight wrist_flex adjust to keep gripper pointing down


async def run_diagnostic(url: str):
    bridge = BridgeClient(url)
    await bridge.connect()
    logger.info("Connected to bridge")

    # Open gripper so fingers won't hit objects
    await bridge.grip(0.0)
    await asyncio.sleep(0.3)

    results = []

    print("\n" + "=" * 60)
    print(" TOUCH-TABLE DIAGNOSTIC")
    print("=" * 60)
    print("\nFor each position the arm will slowly lower.")
    print("  ENTER  = gripper is touching the table")
    print("  s      = skip this position")
    print("  q      = quit early")
    print("=" * 60 + "\n")

    for idx, pos in enumerate(TEST_POSITIONS):
        label = pos["label"]
        hover = list(pos["joints"])

        print(f"\n── Position {idx + 1}/{len(TEST_POSITIONS)}: {label} ──")
        print(f"   Hover: SL={hover[1]:.2f}  EF={hover[2]:.2f}  WF={hover[3]:.2f}")

        # Move to hover first
        await bridge.home()
        await asyncio.sleep(0.5)
        await bridge.move(hover)
        await asyncio.sleep(0.8)

        current = list(hover)
        step_count = 0

        print("   Lowering... (press ENTER when touching, 's' to skip, 'q' to quit)")

        # Use non-blocking stdin check
        import select

        done = False
        skipped = False

        while not done:
            # Lower one step
            current[1] += SL_STEP
            current[2] += EF_STEP
            current[3] += WF_STEP
            current = _clamp(current)
            step_count += 1

            await bridge.move(current)
            await asyncio.sleep(0.4)

            # Check limits
            sl_at_limit = current[1] >= JOINT_LIMITS["shoulder_lift"][1] - 0.05
            ef_at_limit = current[2] <= JOINT_LIMITS["elbow_flex"][0] + 0.05
            if sl_at_limit or ef_at_limit:
                print(f"   ⚠ Joint limit reached at step {step_count}")
                skipped = True
                break

            delta_sl = current[1] - hover[1]
            delta_ef = current[2] - hover[2]
            print(
                f"\r   Step {step_count:3d} | SL={current[1]:+.3f} (Δ{delta_sl:+.3f})  "
                f"EF={current[2]:+.3f} (Δ{delta_ef:+.3f})  ",
                end="",
                flush=True,
            )

            # Check for user input (non-blocking)
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                line = sys.stdin.readline().strip().lower()
                if line == "q":
                    print("\n   Quitting early.")
                    done = True
                    # Lift before exit
                    await bridge.home()
                    break
                elif line == "s":
                    print("\n   Skipped.")
                    skipped = True
                    break
                else:
                    # ENTER = touch confirmed
                    done = True

        if done and line == "q":
            break

        if not skipped and not (done and line == "q"):
            delta_sl = current[1] - hover[1]
            delta_ef = current[2] - hover[2]
            delta_wf = current[3] - hover[3]
            print(f"\n   ✓ TOUCH at step {step_count}")
            print(f"     Hover  → SL={hover[1]:.3f}  EF={hover[2]:.3f}  WF={hover[3]:.3f}")
            print(f"     Touch  → SL={current[1]:.3f}  EF={current[2]:.3f}  WF={current[3]:.3f}")
            print(f"     Delta  → ΔSL={delta_sl:+.3f}  ΔEF={delta_ef:+.3f}  ΔWF={delta_wf:+.3f}")
            results.append(
                {
                    "label": label,
                    "hover": hover[:4],
                    "touch": current[:4],
                    "delta_sl": delta_sl,
                    "delta_ef": delta_ef,
                    "delta_wf": delta_wf,
                    "steps": step_count,
                }
            )

        # Lift back to hover
        await bridge.move(hover)
        await asyncio.sleep(0.5)

    # Summary
    await bridge.home()

    if results:
        print("\n" + "=" * 60)
        print(" RESULTS SUMMARY")
        print("=" * 60)
        avg_sl = sum(r["delta_sl"] for r in results) / len(results)
        avg_ef = sum(r["delta_ef"] for r in results) / len(results)
        avg_wf = sum(r["delta_wf"] for r in results) / len(results)
        print(f"\n{'Label':<16} {'ΔSL':>8} {'ΔEF':>8} {'ΔWF':>8} {'Steps':>6}")
        print("-" * 50)
        for r in results:
            print(
                f"{r['label']:<16} {r['delta_sl']:+8.3f} {r['delta_ef']:+8.3f} "
                f"{r['delta_wf']:+8.3f} {r['steps']:6d}"
            )
        print("-" * 50)
        print(f"{'AVERAGE':<16} {avg_sl:+8.3f} {avg_ef:+8.3f} {avg_wf:+8.3f}")
        print()
        print("Recommended pick_and_place.py settings:")
        print(f"    LOWER_SL = {avg_sl:.3f}")
        print(f"    LOWER_EF = {avg_ef:.3f}")
        # The lift should go above hover, so negative of some fraction
        print(f"    LIFT_SL  = {-abs(avg_sl) * 0.6:.3f}")
        print()

        # Save to file
        out = {
            "positions": results,
            "average_delta_sl": avg_sl,
            "average_delta_ef": avg_ef,
            "average_delta_wf": avg_wf,
        }
        outpath = "/home/beelzebub/Desktop/RobotManipulator/config/touch_table_results.json"
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results → {outpath}")
    else:
        print("\nNo touch points recorded.")

    await bridge.close()


def main():
    parser = argparse.ArgumentParser(description="Touch-table diagnostic")
    parser.add_argument("--url", default="ws://localhost:8765")
    args = parser.parse_args()
    asyncio.run(run_diagnostic(args.url))


if __name__ == "__main__":
    main()
