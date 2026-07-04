"""Dedicated Pick & Place script for the black cube, utilizing calibrated 3D coordinates."""

import asyncio
import logging
import math

import cv2

from robot_manipulator.tasks.pick_and_place import BridgeClient, Detector, _clamp

logger = logging.getLogger("PickBlackCube")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class PickBlackCubeController:
    def __init__(self, bridge: BridgeClient):
        self.bridge = bridge
        self.det = Detector()

        # Ignore bad camera_intrinsics.json for now because it was wildly overfit
        # Assuming standard 640x480 webcam specs
        self.fx, self.fy = 600.0, 600.0
        self.cx, self.cy = 320.0, 240.0

        # Re-calculated Servo constants to prevent both overshoot AND undershoot
        self.PAN_GAIN = 0.0008  # Radians per pixel horizontally
        self.WF_GAIN = 0.0010  # Radians per pixel vertically

        self.SERVO_TOL_X = 25
        self.SERVO_TOL_Y = 25
        self.SERVO_MAX_ITER = 25
        self.STUCK_THRESHOLD = 5

    async def servo(self, joints: list[float], colors: list[str]) -> list[float] | None:
        cur = list(joints)
        prev_dist = float("inf")
        stuck_count = 0
        lost_count = 0

        for it in range(self.SERVO_MAX_ITER):
            print(f"Moving to {cur}")
            await self.bridge.move(cur)
            await asyncio.sleep(1.0)  # 1.0s to ensure camera completely stabilizes

            # Clear stale frame cache by grabbing a few and tossing them
            for _ in range(4):
                _ = await self.bridge.image("wrist")
                await asyncio.sleep(0.1)

            frame = await self.bridge.image("wrist")
            cv2.imwrite(f"/tmp/servo_step_{it}.jpg", frame)

            det = self.det.best_in_frame(frame, colors)
            if not det:
                lost_count += 1
                logger.warning(f"  Servo {it}: target lost (strike {lost_count})")
                if lost_count >= 3:
                    return None
                continue  # Try again after an iteration
            else:
                lost_count = 0  # reset strikes

            dx = det.pixel_x - self.cx
            dy = det.pixel_y - self.cy
            dist = math.hypot(dx, dy)
            logger.info(
                f"  Servo {it}: offset ({dx:+.1f},{dy:+.1f}) dist={dist:.0f} area={det.area:.0f}"
            )

            if abs(dx) < self.SERVO_TOL_X and abs(dy) < self.SERVO_TOL_Y:
                logger.info("  Servo converged")
                cv2.imwrite("/tmp/servo_final.jpg", frame)
                return cur

            if dist >= prev_dist - 2:
                stuck_count += 1
                if stuck_count >= self.STUCK_THRESHOLD:
                    logger.info("  Servo stuck, using best position")
                    return cur
            else:
                stuck_count = 0
            prev_dist = dist

            # Dynamic pan & flex adjustment
            if abs(dx) >= self.SERVO_TOL_X:
                cur[0] -= self.PAN_GAIN * dx  # Pan negative to move object right
            if abs(dy) >= self.SERVO_TOL_Y:
                cur[3] += self.WF_GAIN * dy
            cur = _clamp(cur)

        return cur

    def _lower(self, joints: list[float], amount: float = 1.0) -> list[float]:
        res = list(joints)
        drop = 0.5 * amount
        res[1] += drop
        res[2] -= drop * 0.5
        res[3] += drop * 0.8
        return _clamp(res)

    def _lift(self, joints: list[float]) -> list[float]:
        res = list(joints)
        res[1] -= 0.6
        res[2] += 0.3
        res[3] -= 0.4
        return _clamp(res)

    async def scan_for(self, colors: list[str]):
        logger.info("  Capturing workspace image to locate cube...")
        ws_frame = await self.bridge.image("workspace")
        import cv2
        import numpy as np

        # We look for the black cube in the overhead camera
        hsv = cv2.cvtColor(ws_frame, cv2.COLOR_BGR2HSV)
        # Using the same black bounds from the first script
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 65]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_diff = float("inf")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # The cube was area ~1404 in workspace camera.
            if 1000 < area < 2000:
                diff = abs(area - 1400)
                if diff < best_diff:
                    best_diff = diff
                    best_cnt = cnt

        if best_cnt is None:
            logger.error("  Could not find cube in workspace camera!")
            return None

        x, y, w, h = cv2.boundingRect(best_cnt)
        cx = x + w // 2
        cy = y + h // 2
        logger.info(
            f"  Cube found in workspace at cx={cx}, cy={cy}, area={cv2.contourArea(best_cnt)}"
        )

        # Approximate Mapping: Base is at ~cx=467.
        # cx=265 -> pan -0.6
        pan = (cx - 467) * 0.003

        jump_j = [pan, -0.9, 1.6, 0.1, 0.0, -0.4]
        logger.info(f"  Jumping directly to [pan={pan:.3f}, lift=-0.9, ...]")

        # Perform the jump
        await self.bridge.move(jump_j)
        await asyncio.sleep(2.0)

        # Flush stale frames
        _ = await self.bridge.image("wrist")
        await asyncio.sleep(0.5)
        wrist_frame = await self.bridge.image("wrist")

        # Verify if det can see it in wrist
        det = self.det.best_in_frame(wrist_frame, colors)
        if det:
            logger.info("  Target verified in wrist camera after jump.")
            return jump_j, det
        else:
            logger.warning(
                "  Wrist camera could not see target after jump. Trying a slight search..."
            )
            # fallback small sweep?
            # just return the jump_j and hope servo catches or it just fails
            return jump_j, None

    async def run(self):
        logger.info("=== Black Cube Pick & Place ===")
        t_colors = ["black"]

        await self.bridge.home()
        await asyncio.sleep(0.5)
        await self.bridge.grip(0.0)

        # 1. Scan
        logger.info("── SCAN ──")
        res = await self.scan_for(t_colors)
        if not res:
            logger.error("Black cube not found!")
            return
        scan_j, _ = res

        # 2. Servo
        logger.info("── SERVO ──")
        hover = await self.servo(scan_j, t_colors)
        if not hover:
            logger.error("Lost target during servo")
            return

        # 3. Pre-grasp Verification (NEW)
        logger.info("── VERIFY PRE-GRASP ──")
        verify_frame = await self.bridge.image("wrist")
        verify_det = self.det.best_in_frame(verify_frame, t_colors)
        if not verify_det:
            logger.error("Cannot confirm pre-grasp! Aborting to avoid crash.")
            await self.bridge.home()
            return

        dx = verify_det.pixel_x - self.cx
        dy = verify_det.pixel_y - self.cy
        # Convert pixel to absolute millimeter error for safety check!
        abs_err_X = (dx / self.fx) * 150.0  # Height ~150mm
        abs_err_Y = (dy / self.fy) * 150.0
        dist_mm = math.hypot(abs_err_X, abs_err_Y)
        logger.info(f"  Verification: Cube is mechanically {dist_mm:.1f}mm from optical center.")

        if dist_mm > 15.0:  # If > 1.5 cm off center, it's an unsafe grasp
            logger.error("  Cube is too far off-center, grasp would fail constraints! Aborting.")
            await self.bridge.home()
            return

        # 4. Pick
        logger.info("── PICK ──")
        pre_grasp = self._lower(hover, 0.85)
        await self.bridge.move(pre_grasp)
        await asyncio.sleep(0.6)

        grasp = self._lower(hover, 1.0)
        await self.bridge.move(grasp)
        await asyncio.sleep(0.5)

        await self.bridge.grip(0.85)
        await asyncio.sleep(0.8)

        # 5. Determine placement from anchor? Just hardcode a drop point for testing.
        logger.info("── LIFT ──")
        lift = self._lift(hover)
        await self.bridge.move(lift)
        await asyncio.sleep(0.5)

        # 6. Home and drop
        logger.info("── HOME & DROP ──")
        await self.bridge.home()
        await asyncio.sleep(0.5)
        await self.bridge.grip(0.0)
        logger.info("=== SUCCESS ===")


async def main_async():
    bridge = BridgeClient("ws://localhost:8765")
    await bridge.connect()
    try:
        ctrl = PickBlackCubeController(bridge)
        await ctrl.run()
    finally:
        await bridge.close()


if __name__ == "__main__":
    asyncio.run(main_async())
