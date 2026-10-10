"""Robot backends behind one protocol: fake (tests), MuJoCo sim, and LeRobot (real arm).

Every backend speaks radians in CMD order (`CMD_JOINTS`). `read()` returns
`{"t", "q"[6], "dq"[6]}`, and `command(q_target)` takes the same layout. Only
`LeRobotBackend` converts units, through `JointMap` in `harness/units.py`.

MuJoCo and LeRobot are imported lazily, so the fake backend and the unit tests
run without either. `LeRobotBackend` never talks to hardware when a `robot` object
is injected, which is how the tests exercise it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np

from robot_manipulator.core.policy_iface import Obs
from robot_manipulator.exec.scheduler import WallClock
from robot_manipulator.harness.clock import Clock
from robot_manipulator.harness.units import CMD_JOINTS, JointMap

log = logging.getLogger(__name__)
N = len(CMD_JOINTS)


@runtime_checkable
class RobotBackend(Protocol):
    joint_names: tuple[str, ...]

    def read(self) -> Obs: ...

    def command(self, q_target: np.ndarray) -> None: ...

    def close(self) -> None: ...


def _check_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape != (N,):
        raise ValueError(f"expected {N} joint targets, got shape {q.shape}")
    return q


class FakeBackend:
    """First-order lag toward each command: q += alpha * (target - q).

    Time comes from the injected clock, so tests can drive it with a SimClock.
    `dq` is a finite difference between consecutive reads.
    """

    joint_names = CMD_JOINTS

    def __init__(
        self,
        q0: np.ndarray | None = None,
        alpha: float = 0.3,
        clock: Clock | None = None,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.clock = clock or WallClock()
        self.q = np.zeros(N) if q0 is None else _check_q(q0).copy()
        self._last_t = self.clock.now()
        self._last_q = self.q.copy()

    def read(self) -> Obs:
        t = self.clock.now()
        dt = t - self._last_t
        dq = (self.q - self._last_q) / dt if dt > 0 else np.zeros(N)
        self._last_t, self._last_q = t, self.q.copy()
        return {"t": t, "q": self.q.copy(), "dq": dq}

    def command(self, q_target: np.ndarray) -> None:
        target = _check_q(q_target)
        self.q = self.q + self.alpha * (target - self.q)

    def close(self) -> None:
        pass


class SimBackend:
    """MuJoCo SO101 with position servos. `command` steps physics to the next control tick.

    `hz` is the control rate. Each `command` runs `round(1 / (hz * timestep))` physics
    substeps. `t` is simulated time.
    """

    joint_names = CMD_JOINTS

    def __init__(
        self,
        model: Any = None,
        hz: float = 50.0,
        viewer: bool = False,
        randomizer: Callable[[Any], None] | None = None,
    ) -> None:
        import mujoco

        from robot_manipulator.sim import model as sim_model

        self._mj = mujoco
        self._sm = sim_model
        self.model = model if model is not None else sim_model.load_model()
        self.data = mujoco.MjData(self.model)
        self.hz = hz
        self.n_substeps = max(1, round(1.0 / (hz * self.model.opt.timestep)))
        self.randomizer = randomizer
        self._qpos = sim_model.joint_qpos_adr(self.model)
        self._dof = sim_model.joint_dof_adr(self.model)
        self._act = sim_model.actuator_ids(self.model)
        self._site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sim_model.EE_SITE)
        self._viewer = None
        if viewer:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.reset()

    def reset(self, q0: np.ndarray | None = None) -> None:
        """Reset state to q0 (default zeros), hold it there, and re-apply the randomizer."""
        if self.randomizer is not None:
            self.randomizer(self.model)
        self._mj.mj_resetData(self.model, self.data)
        q = np.zeros(N) if q0 is None else _check_q(q0)
        self.data.qpos[self._qpos] = q
        self.data.ctrl[self._act] = q
        self._mj.mj_forward(self.model, self.data)

    def read(self) -> Obs:
        return {
            "t": float(self.data.time),
            "q": self.data.qpos[self._qpos].copy(),
            "dq": self.data.qvel[self._dof].copy(),
        }

    def command(self, q_target: np.ndarray) -> None:
        q = _check_q(q_target)
        self.data.ctrl[self._act] = q
        for _ in range(self.n_substeps):
            self._mj.mj_step(self.model, self.data)
        if self._viewer is not None:
            self._viewer.sync()

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._site].copy()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def _connect(port: str, robot_id: str):
    from lerobot.robots.so_follower import SOFollower

    from robot_manipulator.arm import build_follower_config

    robot = SOFollower(build_follower_config(port=port, robot_id=robot_id, cameras=False))
    robot.connect(calibrate=False)
    return robot


class LeRobotBackend:
    """Real SO101 follower through LeRobot. Refuses live commands until the joint map is calibrated.

    With `dry_run=True` (the default), `command` only records what it would send, in
    `self.last_command` and the debug log. `read` still queries the arm. Pass `robot=` to
    inject an object with `get_observation`, `send_action` and `disconnect`. Tests use this,
    so they never open a serial port.
    """

    joint_names = CMD_JOINTS

    def __init__(
        self,
        port: str,
        joint_map: JointMap,
        robot_id: str = "my_so101_follower",
        dry_run: bool = True,
        robot: Any = None,
    ) -> None:
        if not dry_run and not joint_map.calibrated:
            raise RuntimeError(
                "refusing live LeRobot backend: config/sim2real.yaml has calibrated: false "
                "(complete hardware ticket H2 first, or run with dry_run=True)"
            )
        self.joint_map = joint_map
        self.dry_run = dry_run
        self.last_command: dict[str, float] | None = None
        self._robot = robot if robot is not None else _connect(port, robot_id)
        self._last_t: float | None = None
        self._last_q: np.ndarray | None = None

    def read(self) -> Obs:
        import time

        obs = self._robot.get_observation()
        q = self.joint_map.lerobot_to_sim(obs)
        t = time.monotonic()
        dq = np.zeros(N)
        if self._last_t is not None and t > self._last_t:
            dq = (q - self._last_q) / (t - self._last_t)
        self._last_t, self._last_q = t, q
        return {"t": t, "q": q, "dq": dq}

    def command(self, q_target: np.ndarray) -> None:
        vals = self.joint_map.sim_to_lerobot(_check_q(q_target))
        self.last_command = vals
        if self.dry_run:
            log.debug("dry-run: would send %s", vals)
            return
        self._robot.send_action(vals)

    def close(self) -> None:
        disconnect = getattr(self._robot, "disconnect", None)
        if disconnect is not None:
            disconnect()
