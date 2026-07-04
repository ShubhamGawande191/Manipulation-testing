"""Pick-and-place task for the SO101 arm.

Uses a **scan + visual-servo** strategy:
  1.  Scan the workspace at a grid of arm positions.
  2.  At each position check the *wrist* camera for the target object.
  3.  Visual-servo to centre the gripper over the object.
  4.  Pick, carry to the container, and place.

No pixel-to-joint calibration is needed — the wrist camera provides
closed-loop feedback for precise positioning.

Usage
-----
    # Show what the workspace camera detects
    python -m robot_manipulator.tasks.pick_and_place detect

    # Scan workspace with wrist camera (debug: see what each position sees)
    python -m robot_manipulator.tasks.pick_and_place scan

    # Run pick & place  (optionally filter by color)
    python -m robot_manipulator.tasks.pick_and_place run [--color black]
"""

import argparse
import asyncio
import base64
import json
import logging
import math
import sys
from dataclasses import dataclass

import cv2
import numpy as np

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("pick_and_place")

# ── Joint names (CMD order) ────────────────────────────────────────
JOINT_NAMES = [
    "shoulder_pan",  # 0
    "shoulder_lift",  # 1
    "elbow_flex",  # 2
    "wrist_flex",  # 3
    "wrist_roll",  # 4
    "gripper",  # 5
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


@dataclass
class Detection:
    name: str
    pixel_x: int
    pixel_y: int
    area: float
    bbox: tuple[int, int, int, int]


def _clamp(joints: list[float]) -> list[float]:
    return [
        max(lo, min(hi, v)) for v, (lo, hi) in zip(joints, (JOINT_LIMITS[n] for n in JOINT_NAMES))
    ]


# ── Bridge client ──────────────────────────────────────────────────
class BridgeClient:
    def __init__(self, url: str = "ws://localhost:8765"):
        self._url, self._ws = url, None

    async def connect(self):
        self._ws = await websockets.connect(self._url)
        logger.info("Bridge connected: %s", self._url)

    async def close(self):
        if self._ws:
            await self._ws.close()

    async def _cmd(self, action: str, params: dict = None, timeout: float = 15) -> dict:
        await self._ws.send(json.dumps({"command": {"action": action, "parameters": params or {}}}))
        r = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=timeout))
        return r["telemetry"]["data"]

    async def state(self) -> dict:
        return await self._cmd("arm.get_state")

    async def move(self, joints: list[float]) -> dict:
        return await self._cmd("arm.move_joints", {"joints": _clamp(joints)})

    async def home(self) -> dict:
        return await self._cmd("arm.home")

    async def grip(self, pos: float) -> dict:
        return await self._cmd("arm.gripper", {"position": pos})

    async def image(self, cam: str = "workspace") -> np.ndarray:
        d = await self._cmd("arm.camera", {"camera": cam, "quality": 90})
        if not d.get("success"):
            raise RuntimeError(f"Camera {cam}: {d}")
        return cv2.imdecode(
            np.frombuffer(base64.b64decode(d["image"]), np.uint8),
            cv2.IMREAD_COLOR,
        )


# ── Vision ─────────────────────────────────────────────────────────
class Detector:
    COLORS: dict[str, dict] = {
        "pink": {
            "ranges": [([140, 40, 80], [175, 255, 255])],
            "min_area": 1500,
        },
        "black": {
            "ranges": [([0, 0, 0], [180, 255, 65])],
            "min_area": 400,
            "max_area": 25000,
        },
        "red": {
            "ranges": [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
            "min_area": 300,
        },
        "green": {
            "ranges": [([35, 80, 50], [85, 255, 255])],
            "min_area": 300,
        },
        "blue": {
            "ranges": [([100, 80, 50], [130, 255, 255])],
            "min_area": 300,
        },
        "yellow": {
            "ranges": [([20, 100, 100], [35, 255, 255])],
            "min_area": 300,
        },
        "orange": {
            "ranges": [([10, 100, 70], [20, 255, 255])],
            "min_area": 300,
        },
    }

    def detect(self, frame: np.ndarray, colors: list[str] = None) -> list[Detection]:
        if colors is None:
            colors = list(self.COLORS.keys())

        # Mask out only the very bottom where fingers show up prominently when zoomed
        # We must paint this WHITE so it doesn't get detected as a black cube!
        masked_frame = frame.copy()
        masked_frame[-80:, :] = 255  # Small crop at bottom

        hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out: list[Detection] = []
        for cname in colors:
            cfg = self.COLORS.get(cname)
            if not cfg:
                continue
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in cfg["ranges"]:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                if a < cfg.get("min_area", 100):
                    continue
                if "max_area" in cfg and a > cfg["max_area"]:
                    continue
                x, y, w, h = cv2.boundingRect(c)

                # Check aspect ratio to reject shadows and table edges
                aspect_ratio = float(w) / max(h, 1)
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue

                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"]) if M["m00"] else x + w // 2
                cy = int(M["m01"] / M["m00"]) if M["m00"] else y + h // 2
                out.append(Detection(cname, cx, cy, a, (x, y, w, h)))
        return out

    def find_container(self, frame: np.ndarray) -> Detection | None:
        pinks = self.detect(frame, ["pink"])
        return max(pinks, key=lambda d: d.area) if pinks else None

    def find_objects(self, frame: np.ndarray) -> list[Detection]:
        return self.detect(frame, ["black", "red", "green", "blue", "yellow", "orange"])

    def best_in_frame(self, frame: np.ndarray, colors: list[str]) -> Detection | None:
        """Largest target-color blob in the frame."""
        dets = self.detect(frame, colors)
        return max(dets, key=lambda d: d.area) if dets else None


# ── Scan grid ──────────────────────────────────────────────────────
# Per-reach-config lowering offsets from touch-table diagnostic (2026-03-11)
# Each tuple: (label, hover_sl, hover_ef, hover_wf, delta_sl, delta_ef, delta_wf)
REACH_CONFIGS = [
    # near — can't reach the table (hits joint limits), removed from scan grid
    # mid — works across full pan range
    ("mid", -1.05, 1.50, 0.25, +1.130, -0.753, +0.377),
    # far — works at centre, moderate lowering
    ("far", -0.80, 1.70, 0.00, +0.480, -0.320, +0.160),
]


def _reach_config_for(joints: list[float]) -> tuple[float, float, float]:
    """Find the closest reach config's lowering offsets for given joints."""
    best, best_dist = REACH_CONFIGS[0], float("inf")  # default to mid
    for cfg in REACH_CONFIGS:
        _, sl, ef, wf, *_ = cfg
        d = abs(joints[1] - sl) + abs(joints[2] - ef)
        if d < best_dist:
            best_dist, best = d, cfg
    _, _, _, _, d_sl, d_ef, d_wf = best
    return d_sl, d_ef, d_wf


def make_scan_grid(
    pan_range: tuple[float, float] = (-0.6, 0.6),
    pan_steps: int = 5,
) -> list[list[float]]:
    """Grid of joint waypoints covering the workspace (mid + far reach only)."""
    pans = np.linspace(pan_range[0], pan_range[1], pan_steps)
    grid: list[list[float]] = []
    for pan in pans:
        for _, sl, ef, wf, *_ in REACH_CONFIGS:
            grid.append([float(pan), sl, ef, wf, 0.0, -0.4])
    return grid


# ── Pick & Place ───────────────────────────────────────────────────
class PickAndPlace:
    """Scan-and-servo pick & place controller."""

    # Visual-servo gains (radians per pixel of offset)
    PAN_GAIN = 0.0015  # pan for X correction
    # Y correction: wrist_flex tilts camera up/down to centre vertically
    WF_GAIN = 0.002

    SERVO_TOL_X = 40  # pixels from centre (horizontal)
    SERVO_TOL_Y = 60  # pixels from centre (vertical — relaxed)
    SERVO_MAX_ITER = 10
    STUCK_THRESHOLD = 3  # bail after N iterations with no improvement

    # Fraction of lowering offset to back off when lifting above hover
    LIFT_FRACTION = 0.35

    def __init__(self, bridge: BridgeClient):
        self.bridge = bridge
        self.det = Detector()

    # ── low-level ──────────────────────────────────────────────────
    def _lower(self, joints: list[float], amt: float = 1.0) -> list[float]:
        """Lower the arm to the table using reach-specific offsets."""
        d_sl, d_ef, d_wf = _reach_config_for(joints)
        j = list(joints)
        j[1] += d_sl * amt
        j[2] += d_ef * amt
        j[3] += d_wf * amt
        return _clamp(j)

    def _lift(self, joints: list[float]) -> list[float]:
        """Lift above hover height using reach-specific offsets."""
        d_sl, d_ef, _ = _reach_config_for(joints)
        j = list(joints)
        j[1] -= abs(d_sl) * self.LIFT_FRACTION
        return _clamp(j)

    # ── scan ───────────────────────────────────────────────────────
    async def scan(
        self,
        colors: list[str],
        grid: list[list[float]] = None,
    ) -> tuple[list[float], Detection] | None:
        """Sweep grid, return (joints, det) where target is best centred
        in the wrist camera."""
        if grid is None:
            grid = make_scan_grid()
        best_j, best_d, best_dist = None, None, float("inf")
        cx_cam, cy_cam = 320, 240

        for i, joints in enumerate(grid):
            await self.bridge.move(joints)
            await asyncio.sleep(0.5)
            frame = await self.bridge.image("wrist")
            det = self.det.best_in_frame(frame, colors)
            if det:
                d = math.hypot(det.pixel_x - cx_cam, det.pixel_y - cy_cam)
                logger.info(
                    "  Scan %2d/%d  %s  wrist=(%d,%d) dist=%.0f area=%.0f",
                    i + 1,
                    len(grid),
                    det.name,
                    det.pixel_x,
                    det.pixel_y,
                    d,
                    det.area,
                )
                cv2.imwrite(f"/tmp/scan_{i:02d}.jpg", frame)
                if d < best_dist:
                    best_dist, best_j, best_d = d, list(joints), det
            else:
                logger.debug("  Scan %2d/%d: —", i + 1, len(grid))

        return (best_j, best_d) if best_j else None

    # ── visual servo ───────────────────────────────────────────────
    async def servo(
        self,
        joints: list[float],
        colors: list[str],
    ) -> list[float] | None:
        """Servo on wrist camera — pan for X, wrist_flex for Y.

        Only adjusts pan (joint 0) and wrist_flex (joint 3).
        Does NOT try to change reach (SL/EF) — that's fixed by the scan grid.
        """
        cx_cam, cy_cam = 320, 240
        cur = list(joints)
        prev_dist = float("inf")
        stuck_count = 0

        for it in range(self.SERVO_MAX_ITER):
            await self.bridge.move(cur)
            await asyncio.sleep(0.4)
            frame = await self.bridge.image("wrist")
            det = self.det.best_in_frame(frame, colors)
            if not det:
                logger.warning("  Servo %d: target lost", it)
                return None

            dx = det.pixel_x - cx_cam
            dy = det.pixel_y - cy_cam
            dist = math.hypot(dx, dy)
            logger.info(
                "  Servo %d: offset (%+d,%+d) dist=%.0f area=%.0f", it, dx, dy, dist, det.area
            )

            # Check convergence (X must be good; Y is best-effort)
            if abs(dx) < self.SERVO_TOL_X and abs(dy) < self.SERVO_TOL_Y:
                logger.info("  Servo converged")
                cv2.imwrite("/tmp/servo_final.jpg", frame)
                return cur

            # Stuck detection — if distance didn't improve, count
            if dist >= prev_dist - 5:
                stuck_count += 1
                if stuck_count >= self.STUCK_THRESHOLD:
                    logger.info(
                        "  Servo stuck after %d iters (dist=%.0f), using best position", it, dist
                    )
                    cv2.imwrite("/tmp/servo_final.jpg", frame)
                    return cur
            else:
                stuck_count = 0
            prev_dist = dist

            # Only adjust pan (X) and wrist_flex (Y)
            if abs(dx) >= self.SERVO_TOL_X:
                cur[0] += self.PAN_GAIN * dx
            if abs(dy) >= self.SERVO_TOL_Y:
                cur[3] += self.WF_GAIN * dy
            cur = _clamp(cur)

        logger.warning("  Servo: max iters reached (best effort)")
        cv2.imwrite("/tmp/servo_final.jpg", frame)
        return cur

    # ── main ───────────────────────────────────────────────────────
    async def run(self, target_color: str = None):
        logger.info("=== Pick & Place ===")
        t_colors = (
            [target_color]
            if target_color
            else [
                "black",
                "red",
                "green",
                "blue",
                "yellow",
                "orange",
            ]
        )

        # 1. Home + open gripper
        await self.bridge.home()
        await asyncio.sleep(0.5)
        await self.bridge.grip(0.0)
        await asyncio.sleep(0.3)

        # Workspace overview
        ws = await self.bridge.image("workspace")
        cv2.imwrite("/tmp/pnp_workspace.jpg", ws)
        objs = self.det.find_objects(ws)
        cont = self.det.find_container(ws)
        logger.info("Workspace: %d objects, container=%s", len(objs), bool(cont))

        # 2. Scan for target
        logger.info("── SCAN for %s ──", t_colors)
        res = await self.scan(t_colors)
        if not res:
            logger.error("Target not found!")
            await self.bridge.home()
            return False
        scan_j, scan_d = res
        logger.info("Best match: %s — servoing...", scan_d.name)

        # 3. Servo over target
        logger.info("── SERVO over target ──")
        hover = await self.servo(scan_j, t_colors)
        if not hover:
            logger.error("Lost target during servo")
            await self.bridge.home()
            return False

        # 4. Verify over target before picking
        logger.info("── VERIFY position ──")
        frame = await self.bridge.image("wrist")
        v_det = self.det.best_in_frame(frame, t_colors)
        if v_det:
            logger.info(
                "  Target visible at (%d,%d) area=%.0f — proceeding",
                v_det.pixel_x,
                v_det.pixel_y,
                v_det.area,
            )
            cv2.imwrite("/tmp/pnp_verify.jpg", frame)
        else:
            logger.warning("  Target NOT visible in wrist camera — proceeding anyway")

        # 5. Pick
        logger.info("── PICK ──")
        # Lower to ~85% of table height
        pre_grasp = self._lower(hover, 0.85)
        await self.bridge.move(pre_grasp)
        await asyncio.sleep(0.6)
        # Final lower to table
        grasp = self._lower(hover, 1.0)
        await self.bridge.move(grasp)
        await asyncio.sleep(0.5)
        # Close gripper firmly
        await self.bridge.grip(0.85)
        await asyncio.sleep(0.8)
        # Verify grasp — check if gripper closed around something
        logger.info("── VERIFY grasp ──")
        state = await self.bridge.state()
        if state.get("joints", {}).get("gripper", {}).get("position", 0):
            grip_pos = state["joints"]["gripper"]["position"]
            logger.info("  Gripper position: %.3f (>0.3 likely has object)", grip_pos)
        # Lift with object
        lift = self._lift(hover)
        await self.bridge.move(lift)
        await asyncio.sleep(0.5)

        # 6. Find container
        logger.info("── SCAN for container ──")
        c_res = await self.scan(["pink"])
        if not c_res:
            logger.error("Container not found!")
            await self.bridge.home()
            return False
        c_scan_j, _ = c_res
        logger.info("Container located — servoing...")
        c_hover = await self.servo(c_scan_j, ["pink"])
        if not c_hover:
            logger.warning("Container servo failed, using scan position")
            c_hover = c_scan_j

        # 6. Place
        logger.info("── PLACE ──")
        above = self._lift(c_hover)
        await self.bridge.move(above)
        await asyncio.sleep(0.3)
        drop = self._lower(c_hover, 0.5)
        await self.bridge.move(drop)
        await asyncio.sleep(0.3)
        await self.bridge.grip(0.0)
        await asyncio.sleep(0.5)
        await self.bridge.move(above)
        await asyncio.sleep(0.3)

        # 7. Home
        await self.bridge.home()
        logger.info("=== Pick & Place complete ===")
        after = await self.bridge.image("workspace")
        cv2.imwrite("/tmp/pnp_after.jpg", after)
        return True


# ── CLI ────────────────────────────────────────────────────────────
async def cmd_detect(args):
    bridge = BridgeClient(args.url)
    await bridge.connect()
    try:
        await bridge.home()
        await asyncio.sleep(0.5)
        frame = await bridge.image("workspace")
        det = Detector()
        container = det.find_container(frame)
        objects = det.find_objects(frame)
        vis = frame.copy()
        if container:
            x, y, w, h = container.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(
                vis, "CONTAINER", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
            )
            print(f"Container: ({container.pixel_x},{container.pixel_y}) area={container.area:.0f}")
        for o in objects:
            x, y, w, h = o.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{o.name}({o.area:.0f})",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            print(f"Object: {o.name} ({o.pixel_x},{o.pixel_y}) area={o.area:.0f}")
        cv2.imwrite("/tmp/pnp_detect.jpg", vis)
        print("\nSaved → /tmp/pnp_detect.jpg")
    finally:
        await bridge.close()


async def cmd_scan(args):
    bridge = BridgeClient(args.url)
    await bridge.connect()
    try:
        det = Detector()
        grid = make_scan_grid()
        await bridge.grip(0.0)
        all_colors = ["black", "red", "green", "blue", "yellow", "orange", "pink"]
        for i, joints in enumerate(grid):
            await bridge.move(joints)
            await asyncio.sleep(0.5)
            frame = await bridge.image("wrist")
            dets = det.detect(frame, all_colors)
            tag = ", ".join(f"{d.name}({d.area:.0f})" for d in dets) or "—"
            print(f"Pos {i:2d}  pan={joints[0]:+.2f} sl={joints[1]:+.2f}: {tag}")
            cv2.imwrite(f"/tmp/scan_{i:02d}.jpg", frame)
        await bridge.home()
        print("\nSaved wrist images → /tmp/scan_*.jpg")
    finally:
        await bridge.close()


async def cmd_run(args):
    bridge = BridgeClient(args.url)
    await bridge.connect()
    try:
        ctrl = PickAndPlace(bridge)
        await ctrl.run(target_color=args.color)
    finally:
        await bridge.close()


def main():
    ap = argparse.ArgumentParser(description="SO101 Pick & Place")
    ap.add_argument("--url", default="ws://localhost:8765")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("detect", help="Detect objects via workspace camera")
    sub.add_parser("scan", help="Scan workspace with wrist camera")
    p = sub.add_parser("run", help="Execute pick & place")
    p.add_argument(
        "--color", default=None, help="Target color: black, red, green, blue, yellow, orange"
    )
    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    asyncio.run({"detect": cmd_detect, "scan": cmd_scan, "run": cmd_run}[args.cmd](args))


if __name__ == "__main__":
    main()
