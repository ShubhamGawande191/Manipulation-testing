"""PincOpen gripper control — direct Feetech STS serial, no LeRobot/ROS dependency.

The PincOpen (Pollen Robotics open design, driver: CNURobotics/pinc_open_driver)
is mounted as the follower arm's last servo: **ID 6 on /dev/so101_follower**,
replacing the stock SO101 gripper.  It is driven over the shared arm bus with
plain Dynamixel-v1-style packets, so this module must not be used while LeRobot
or the ROS2 bridge holds the same serial port.

The mechanism is NOT back-drivable — never force it by hand; always move it
under power.  Range measured 2026-07-26 with a torque-limited creep against
both ends of the servo's EEPROM window:

    OPEN_POS   = 2005 steps   (low = open)
    CLOSED_POS = 3200 steps   (high = closed; hard preload lives at ~3311,
                               beyond the EEPROM max of 3218)

``grip()`` closes until either the position target is reached or the load /
current spike of finger-object contact is detected, then holds torque so the
object stays clamped.  ``open_()``/``release()`` returns to OPEN_POS and drops
torque.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import serial

PORT = "/dev/so101_follower"
SERVO_ID = 6
BAUD = 1_000_000

OPEN_POS = 2005
CLOSED_POS = 3200

# registers (Feetech STS3215)
REG_TORQUE_ENABLE = 0x28
REG_ACCELERATION = 0x29
REG_GOAL_POSITION = 0x2A
REG_GOAL_SPEED = 0x2E
REG_TORQUE_LIMIT = 0x30
REG_PRESENT_POSITION = 0x38
REG_PRESENT_LOAD = 0x3C
REG_PRESENT_CURRENT = 0x45
REG_PRESENT_TEMP = 0x3F


@dataclass
class GripperStatus:
    position: int
    load: int  # signed, /1000 of stall torque
    current_ma: float
    temp_c: int

    @property
    def openness(self) -> float:
        """0.0 = fully closed, 1.0 = fully open."""
        span = CLOSED_POS - OPEN_POS
        return max(0.0, min(1.0, (CLOSED_POS - self.position) / span))


class PincGripper:
    """Torque-limited position control of the PincOpen gripper servo.

    Usage::

        with PincGripper() as g:
            g.open_()
            result = g.grip()          # close on an object, hold it
            g.release()                # let go, torque off
    """

    def __init__(self, port: str = PORT, servo_id: int = SERVO_ID):
        self.port = port
        self.servo_id = servo_id
        self.ser: serial.Serial | None = None

    def __enter__(self) -> PincGripper:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        self.ser = serial.Serial(self.port, baudrate=BAUD, timeout=0.05)

    def close(self) -> None:
        """Close the serial port (leaves torque in whatever state it was)."""
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    # --- low-level protocol -------------------------------------------------

    @staticmethod
    def _checksum(body: list[int]) -> int:
        return (~sum(body)) & 0xFF

    def _txn(self, body: list[int]) -> None:
        pkt = [0xFF, 0xFF] + body + [self._checksum(body)]
        self.ser.reset_input_buffer()
        self.ser.write(bytes(pkt))
        self.ser.flush()

    def _write_reg(self, addr: int, value: int, width: int) -> None:
        params = [addr, *value.to_bytes(width, "little")]
        self._txn([self.servo_id, len(params) + 2, 0x03, *params])
        time.sleep(0.005)
        self.ser.read(6)  # consume status packet

    def _read_reg(self, addr: int, width: int) -> int | None:
        self._txn([self.servo_id, 0x04, 0x02, addr, width])
        r = self.ser.read(6 + width)
        if len(r) < 6 + width or r[0] != 0xFF or r[1] != 0xFF or r[2] != self.servo_id:
            return None
        return int.from_bytes(r[5 : 5 + width], "little")

    # --- state --------------------------------------------------------------

    def status(self) -> GripperStatus:
        pos = self._read_reg(REG_PRESENT_POSITION, 2)
        load_raw = self._read_reg(REG_PRESENT_LOAD, 2) or 0
        load = (load_raw & 0x3FF) * (-1 if load_raw & 0x400 else 1)
        cur = (self._read_reg(REG_PRESENT_CURRENT, 2) or 0) * 6.5
        temp = self._read_reg(REG_PRESENT_TEMP, 1) or 0
        if pos is None:
            raise OSError(f"gripper servo id={self.servo_id} not responding on {self.port}")
        return GripperStatus(pos, load, cur, temp)

    # --- motion -------------------------------------------------------------

    def _arm_motion(self, torque_permille: int, speed: int, accel: int = 50) -> None:
        self._write_reg(REG_TORQUE_LIMIT, torque_permille, 2)
        self._write_reg(REG_ACCELERATION, accel, 1)
        self._write_reg(REG_GOAL_SPEED, speed, 2)
        self._write_reg(REG_TORQUE_ENABLE, 1, 1)

    def torque_off(self) -> None:
        self._write_reg(REG_TORQUE_ENABLE, 0, 1)

    def move_to(
        self,
        target: int,
        torque_permille: int = 250,
        speed: int = 800,
        load_abort: int = 350,
        current_abort_ma: float = 500.0,
        timeout_s: float = 5.0,
    ) -> GripperStatus:
        """Move to *target* steps; stop early on load/current spike (contact).

        Returns the final status.  Torque stays ON (holding) — call
        ``torque_off()`` or ``release()`` to let go.
        """
        target = max(OPEN_POS, min(CLOSED_POS, target))
        self._arm_motion(torque_permille, speed)
        self._write_reg(REG_GOAL_POSITION, target, 2)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            time.sleep(0.05)
            st = self.status()
            if abs(st.load) > load_abort or st.current_ma > current_abort_ma:
                # contact / stall: hold just short of the spike point
                back = st.position + (30 if target < st.position else -30)
                self._write_reg(REG_GOAL_POSITION, back, 2)
                time.sleep(0.2)
                return self.status()
            if abs(st.position - target) < 10:
                return st
        return self.status()

    def open_(self, **kw) -> GripperStatus:
        st = self.move_to(OPEN_POS, **kw)
        self.torque_off()
        return st

    def close_fingers(self, **kw) -> GripperStatus:
        """Close fully (or onto an object); torque stays on to hold."""
        return self.move_to(CLOSED_POS, **kw)

    def grip(
        self,
        torque_permille: int = 300,
        speed: int = 500,
        contact_load: int = 280,
    ) -> GripperStatus:
        """Close on an object and keep holding it.

        Returns final status; ``status().openness`` > 0 with sustained load
        means an object is held (empty closure ends at openness ≈ 0).
        """
        return self.move_to(
            CLOSED_POS,
            torque_permille=torque_permille,
            speed=speed,
            load_abort=contact_load,
        )

    def release(self, **kw) -> GripperStatus:
        return self.open_(**kw)


def main() -> None:
    """CLI: python -m robot_manipulator.gripper {status|open|close|grip|release|cycle}."""
    import argparse

    p = argparse.ArgumentParser(description="PincOpen gripper control (follower bus ID 6)")
    p.add_argument("cmd", choices=["status", "open", "close", "grip", "release", "cycle"])
    p.add_argument("--port", default=PORT)
    args = p.parse_args()

    with PincGripper(port=args.port) as g:
        if args.cmd == "status":
            st = g.status()
            print(
                f"pos={st.position} openness={st.openness:.2f} load={st.load} "
                f"current={st.current_ma:.0f}mA temp={st.temp_c}C"
            )
            return
        if args.cmd == "open" or args.cmd == "release":
            st = g.open_()
        elif args.cmd == "close":
            st = g.close_fingers()
        elif args.cmd == "grip":
            st = g.grip()
            print(f"holding: pos={st.position} openness={st.openness:.2f} load={st.load}")
            return  # keep torque on
        elif args.cmd == "cycle":
            for _ in range(2):
                g.close_fingers()
                g.move_to(OPEN_POS)
            st = g.open_()
        print(f"done: pos={st.position} openness={st.openness:.2f} temp={st.temp_c}C")


if __name__ == "__main__":
    main()
