"""Controlled peg-in-hole on the SO-101 twin: which error sources move the 1 mm / 2 mm line?

    uv run insertion-study smoke                    # one perfect insertion per clearance
    uv run insertion-study sweep --n 40             # success vs clearance, per error factor

A square peg is fixed at the fingertip along the tool axis (no grasp: only the arm's accuracy
matters), and a square hole with an exact diametral clearance (+ optional chamfer) sits on the
table. A scripted straight-down insertion uses the closed-form IK. Error sources, each with the
value measured on the real arm (2026-10-18..20), are switched on one at a time and together:

- ``deadband``: each joint stops short of its command on the side it came from (play operator,
  real: ~2 deg on lift/elbow; P/D servo loop with I = 0 against gearbox friction)
- ``lag``: commands reach the joints late (real: 68 ms vs 17 ms nominal twin)
- ``soft``: servo stiffness instead of the stiff Menagerie actuator (real: ~18 N.m/rad lift)
- ``perception``: the hole is located with a random error (real overhead camera: ~1 mm/px,
  board pose vs the arm: 2-9 mm)
- ``trim``: the settle trim (software integral action, bounded +-4 deg) on top, kept on
  through the slow descent

Success = the peg reaches ``DEPTH_M`` into the hole within the time limit.
"""

from __future__ import annotations

import json
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import typer

from robot_manipulator.sim.model import ASSET_DIR, JOINTS, joint_qpos_adr, joint_ranges

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs" / "insertion"
TABLE_Z = -0.0275
PEG_SIDE = 0.012  # m, square peg
PEG_LEN = 0.025  # m beyond the fingertip (straight down needs wrist_flex <= 95 deg: keep it low)
HOLE_H = 0.02  # fixture height above the table
DEPTH_M = 0.012  # success: peg bottom this far below the hole top
# Stiff contacts: with MuJoCo's defaults (5 ms, solref 0.02) a position-controlled arm pushes the
# peg 2-4 mm into the walls and "inserts" through them. 1 ms + solref 2 ms keeps penetration
# under 0.1 mm and agrees with 0.5 ms (checked 2026-10-20). Every episode reports its penetration.
TIMESTEP = 0.001
CONTACT_SOLREF = [0.002, 1.0]
CONTACT_SOLIMP = [0.99, 0.999, 0.0005, 0.5, 2.0]
MAX_PEN_MM = 0.2  # an episode deeper than this is a physics failure, not a result
FRICTION = 0.35  # printed PLA on PLA; MuJoCo's default 1.0 jams a 45 deg chamfer
HOLE_XY = np.array([0.22, 0.0])
HZ = 50.0
ROLL = np.radians(-80.0)

app = typer.Typer(add_completion=False, help=__doc__)


@dataclass
class Factors:
    deadband_deg: float = 0.0  # per arm joint (lift/elbow/wflex), play-operator half width
    lag_ticks: int = 0  # extra command delay at 50 Hz (68 ms ~ 3 ticks beyond the twin's own)
    soft: bool = False  # servo-like stiffness (kp scaled to the measured ~18 N.m/rad)
    perception_mm: float = 0.0  # std of the error in the estimated hole position (x, y)
    trim: bool = False  # settle trim before the final descent
    trim_descent: bool = True  # keep the trim running during the descent (else frozen)
    chamfer_mm: float = 0.0
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- scene
def build(
    clearance_mm: float,
    chamfer_mm: float = 0.0,
    hole_xy=HOLE_XY,
    yaw_deg: float | None = None,
    friction: float = FRICTION,
    cameras: dict[str, tuple] | None = None,
) -> mujoco.MjModel:
    """Scene with the hole turned to the peg's yaw at the insertion pose (``peg_yaw``): the wrist
    roll and pan turn the square peg ~7 deg off the table axes, which alone needs ~1.4 mm of
    clearance. ``yaw_deg`` overrides it (e.g. to study misalignment)."""
    if yaw_deg is None:
        yaw_deg = peg_yaw(hole_xy)
    spec = mujoco.MjSpec.from_file(str(ASSET_DIR / "scene_pinc.xml"))
    spec.option.timestep = TIMESTEP
    # peg on the fingertip, colliding only with the fixture (contype/conaffinity bit 2)
    pinc = spec.body("pinc_base")
    peg = pinc.add_geom()
    peg.name = "peg"
    peg.type = mujoco.mjtGeom.mjGEOM_BOX
    tip_x = 0.1241  # closed fingertip along the tool axis (calib.extrinsics.TipFK)
    peg.pos = [tip_x + PEG_LEN / 2, 0.0, -0.005]
    peg.size = [PEG_LEN / 2, PEG_SIDE / 2, PEG_SIDE / 2]
    peg.contype, peg.conaffinity = 2, 0
    peg.rgba = [0.9, 0.5, 0.1, 1]
    peg.mass = 0.005
    peg.friction = [friction, 0.005, 0.0001]
    peg.solref, peg.solimp = CONTACT_SOLREF, CONTACT_SOLIMP
    # square hole from four walls; side = peg + clearance
    half_hole = (PEG_SIDE + clearance_mm / 1000) / 2
    wall_t, outer = 0.02, half_hole + 0.02
    ch = chamfer_mm / 1000
    body = spec.worldbody.add_body()
    body.name = "fixture"
    body.pos = [hole_xy[0], hole_xy[1], 0]
    qz = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qz, np.array([0.0, 0.0, 1.0]), np.radians(yaw_deg))
    body.quat = qz.tolist()
    for k, (dx, dy, sx, sy) in enumerate(
        (
            (half_hole + wall_t / 2, 0, wall_t / 2, outer),
            (-half_hole - wall_t / 2, 0, wall_t / 2, outer),
            (0, half_hole + wall_t / 2, half_hole, wall_t / 2),
            (0, -half_hole - wall_t / 2, half_hole, wall_t / 2),
        )
    ):
        h = HOLE_H - ch  # the chamfer ramp takes the top `ch` of the inner edge
        g = body.add_geom()
        g.name = f"wall{k}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.pos = [dx, dy, TABLE_Z + h / 2]
        g.size = [sx, sy, h / 2]
        g.contype, g.conaffinity = 0, 2
        g.friction = [friction, 0.005, 0.0001]
        g.solref, g.solimp = CONTACT_SOLREF, CONTACT_SOLIMP
        g.rgba = [0.3, 0.3, 0.35, 1]
        if ch > 0:  # 45 deg ramp from the hole edge (height h) out to `ch` away (height H)
            r = body.add_geom()
            r.name = f"chamfer{k}"
            r.friction = [friction, 0.005, 0.0001]
            r.solref, r.solimp = CONTACT_SOLREF, CONTACT_SOLIMP
            r.type = mujoco.mjtGeom.mjGEOM_BOX
            out_dir = np.sign([dx, dy]) * (np.abs([dx, dy]) > 1e-9)
            edge = out_dir * half_hole
            mid = np.r_[edge + out_dir * ch / 2, TABLE_Z + h + ch / 2]
            t = 0.004
            n = np.r_[-out_dir / np.sqrt(2), 1 / np.sqrt(2)]  # ramp normal: into the hole, up
            r.pos = (mid - n * t).tolist()
            along = 0.5 * np.hypot(ch, ch)
            length = outer if abs(dx) > 1e-9 else half_hole
            r.size = [along, length, t] if abs(dx) > 1e-9 else [length, along, t]
            axis = [0, 1, 0] if abs(dx) > 1e-9 else [1, 0, 0]
            ang = np.radians(45) * (-np.sign(dx) if abs(dx) > 1e-9 else np.sign(dy))
            q = np.zeros(4)
            mujoco.mju_axisAngle2Quat(q, np.array(axis, float), ang)
            r.quat = q.tolist()
            r.contype, r.conaffinity = 0, 2
            r.rgba = [0.4, 0.4, 0.45, 1]
    for name, (pos, target, fovy) in (cameras or {}).items():  # fixed render cameras
        pos, target = np.asarray(pos, float), np.asarray(target, float)
        fwd = (target - pos) / np.linalg.norm(target - pos)
        x = np.cross(fwd, [0.0, 0.0, 1.0])
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
        y = np.cross(x, fwd)  # MuJoCo cameras look along -z with +y up
        R = np.column_stack([x, y, -fwd])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.flatten())
        spec.worldbody.add_camera(name=name, pos=pos.tolist(), quat=q.tolist(), fovy=float(fovy))
    return spec.compile()


_YAW_CACHE: dict[tuple, float] = {}


def peg_yaw(hole_xy=HOLE_XY) -> float:
    """The peg's yaw about the vertical (deg) when it sits straight down over ``hole_xy``."""
    key = tuple(np.round(hole_xy, 5))
    if key not in _YAW_CACHE:
        from robot_manipulator.ik.analytic import AnalyticIK
        from robot_manipulator.sim.model import apply_joint_couplings

        m = build(1.0, hole_xy=hole_xy, yaw_deg=0.0)
        d = mujoco.MjData(m)
        q = AnalyticIK().solve(np.r_[hole_xy, TABLE_Z + HOLE_H + PEG_LEN], np.pi / 2, ROLL)
        d.qpos[joint_qpos_adr(m, JOINTS)] = q
        apply_joint_couplings(m, d)
        mujoco.mj_forward(m, d)
        R = d.geom_xmat[m.geom("peg").id].reshape(3, 3)
        _YAW_CACHE[key] = float(np.degrees(np.arctan2(R[1, 1], R[0, 1])) - 90.0)
    return _YAW_CACHE[key]


# ---------------------------------------------------------------- plant + controller
class Plant:
    """MuJoCo arm + the measured imperfections between command and joint."""

    def __init__(self, model: mujoco.MjModel, f: Factors, rng) -> None:
        from robot_manipulator.sim.model import actuator_ids

        self.m, self.f, self.rng = model, f, rng
        self.d = mujoco.MjData(model)
        self.qadr = joint_qpos_adr(model, JOINTS)
        self.act = actuator_ids(model, JOINTS)
        self.n_sub = max(1, round(1 / (HZ * model.opt.timestep)))
        if f.soft:  # measured holding stiffness ~18 N.m/rad on the lift; P-scaled elsewhere
            for j, k in zip(range(5), (12.0, 18.0, 18.0, 12.0, 12.0)):
                a = self.act[j]
                scale = k / model.actuator_gainprm[a, 0]
                model.actuator_gainprm[a, 0] *= scale
                model.actuator_biasprm[a, 1] *= scale
        self.engaged = None  # play-operator output per joint
        self.pending: deque = deque()
        self.peg = model.geom("peg").id
        self.max_pen = 0.0  # deepest peg contact penetration seen (m)

    def reset(self, q0: np.ndarray) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.qadr] = q0
        self.d.ctrl[self.act] = q0
        mujoco.mj_forward(self.m, self.d)
        self.engaged = q0.copy()
        self.pending = deque(q0.copy() for _ in range(self.f.lag_ticks))

    def q(self) -> np.ndarray:
        return self.d.qpos[self.qadr].copy()

    def step(self, cmd: np.ndarray) -> None:
        cmd = np.asarray(cmd, float)
        if self.f.lag_ticks:
            self.pending.append(cmd)
            cmd = self.pending.popleft()
        b = np.radians(self.f.deadband_deg)
        out = cmd.copy()
        if b > 0:  # play operator on lift/elbow/wflex: output lags the command by up to b
            for j in (1, 2, 3):
                lo, hi = cmd[j] - b, cmd[j] + b
                self.engaged[j] = min(max(self.engaged[j], lo), hi)
                out[j] = self.engaged[j]
        self.d.ctrl[self.act] = out
        for _ in range(self.n_sub):
            mujoco.mj_step(self.m, self.d)
            for c in self.d.contact[: self.d.ncon]:
                if self.peg in (c.geom1, c.geom2):
                    self.max_pen = max(self.max_pen, -c.dist)

    def peg_bottom(self) -> np.ndarray:
        """Centre of the peg's far face (world)."""
        R = self.d.geom_xmat[self.peg].reshape(3, 3)
        return self.d.geom_xpos[self.peg] + R[:, 0] * PEG_LEN / 2


def run_episode(model, f: Factors, rng, hole_xy=HOLE_XY) -> dict:
    from robot_manipulator.ik.analytic import AnalyticIK

    ik = AnalyticIK()
    plant = Plant(model, f, rng)
    lo, hi = joint_ranges(model, JOINTS)
    est = hole_xy + rng.normal(0, f.perception_mm / 1000, 2)  # where the arm *thinks* it is
    top = TABLE_Z + HOLE_H

    def tip_target(z_peg_bottom: float, xy) -> np.ndarray | None:
        # the IK point is the fingertip; the peg extends PEG_LEN below it (tool straight down)
        return ik.solve(np.r_[xy, z_peg_bottom + PEG_LEN], np.pi / 2, ROLL)

    start = tip_target(top + 0.02, est)
    if start is None:
        return {"ok": False, "reason": "unreachable"}
    plant.reset(np.clip(start, lo, hi))
    # settle above the hole (with trim: integral action until the joints sit on target)
    above = tip_target(top + 0.005, est)
    trim = np.zeros(6)

    def trim_step(target: np.ndarray) -> None:
        """Settle trim (software integral action, as deployed): bounded, with a deadband."""
        nonlocal trim
        err = target - plant.q()
        act = np.abs(err) > np.radians(0.15)
        trim[act] += 0.08 * err[act]
        trim = np.clip(trim, -np.radians(4), np.radians(4))

    # settle above the hole until the arm stops moving (and, with trim, sits on target)
    q_prev = plant.q()
    for k in range(int(3.0 * HZ)):
        if f.trim:
            trim_step(above)
        plant.step(above + trim)
        q_now = plant.q()
        still = np.abs(q_now - q_prev)[:5].max() < np.radians(0.02)
        on_target = not f.trim or np.abs(above - q_now)[:5].max() < np.radians(0.2)
        if k > 10 and still and on_target:
            break
        q_prev = q_now
    lateral = float(np.linalg.norm(plant.peg_bottom()[:2] - hole_xy))
    # straight-down descent at 1 cm/s, Cartesian via per-tick IK; the trim stays on (bounded)
    z = top + 0.005
    for k in range(int(4.0 * HZ)):
        z = max(z - 0.01 / HZ, top - DEPTH_M - 0.005)
        q = tip_target(z, est)
        if q is None:
            break
        if f.trim and f.trim_descent:
            trim_step(q)
        plant.step(q + trim)
        if plant.peg_bottom()[2] < top - DEPTH_M:
            pen = 1000 * plant.max_pen
            return {"ok": True, "lateral_mm": 1000 * lateral, "t": k / HZ, "pen_mm": pen}
    return {
        "ok": False,
        "reason": "stuck",
        "lateral_mm": 1000 * lateral,
        "depth_mm": float(1000 * (top - plant.peg_bottom()[2])),
        "pen_mm": 1000 * plant.max_pen,
    }


FACTOR_SETS = {
    "ideal": Factors(),
    "deadband 2deg": Factors(deadband_deg=2.0),
    "deadband 2deg + trim": Factors(deadband_deg=2.0, trim=True),
    "lag 68ms": Factors(lag_ticks=3),
    "soft servo": Factors(soft=True),
    "soft + trim": Factors(soft=True, trim=True),
    "perception 1mm": Factors(perception_mm=1.0),
    "perception 2mm": Factors(perception_mm=2.0),
    "all (as measured)": Factors(deadband_deg=2.0, lag_ticks=3, soft=True, perception_mm=1.0),
    "all + trim": Factors(deadband_deg=2.0, lag_ticks=3, soft=True, perception_mm=1.0, trim=True),
    # does a chamfer need lateral give, and does the trim cancel it during the descent?
    "perception 1mm + soft + trim": Factors(soft=True, perception_mm=1.0, trim=True),
    "perception 1mm + soft + settle-only trim": Factors(
        soft=True, perception_mm=1.0, trim=True, trim_descent=False
    ),
    "all + settle-only trim": Factors(
        deadband_deg=2.0, lag_ticks=3, soft=True, perception_mm=1.0, trim=True, trim_descent=False
    ),
}


def _cell(job: tuple) -> list[bool]:
    """Episodes ``idx`` of one cell. Episode i always draws from rng([seed, i]): paired across
    rows, chamfers and sweeps, and independent of how the work is split over processes."""
    f, clearance, chamfer, friction, seed, idx = job
    model = build(clearance, chamfer, friction=friction)
    res = [run_episode(model, f, np.random.default_rng([seed, i])) for i in idx]
    return [(bool(r["ok"]), float(r.get("pen_mm", 0.0))) for r in res]


def run_cell(f: Factors, clearance, chamfer, friction, n, seed, pool) -> list[tuple[bool, float]]:
    chunks = [list(range(n))[k :: os.cpu_count() or 1] for k in range(os.cpu_count() or 1)]
    jobs = [(f, clearance, chamfer, friction, seed, c) for c in chunks if c]
    out = [(False, 0.0)] * n
    for c, oks in zip([c for c in chunks if c], pool.map(_cell, jobs), strict=True):
        for i, ok in zip(c, oks, strict=True):
            out[i] = ok
    return out


@app.command()
def smoke() -> None:
    """One ideal insertion per clearance (sanity check of the scene and controller)."""
    rng = np.random.default_rng(0)
    for c in (0.5, 1.0, 2.0, 3.0):
        r = run_episode(build(c), Factors(), rng)
        typer.echo(f"clearance {c} mm: {r}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson interval for k successes out of n."""
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


@app.command()
def sweep(
    n: int = typer.Option(30, help="episodes per cell"),
    clearances: str = typer.Option("0.5,1,2,3", help="diametral clearance, mm"),
    chamfer: float = typer.Option(0.0, help="chamfer size, mm"),
    friction: float = typer.Option(FRICTION, help="peg/fixture sliding friction"),
    only: str = typer.Option("", help="comma list of factor sets (default: all)"),
    seed: int = 0,
) -> None:
    """Success rate vs clearance for each error factor -> runs/insertion/<ts>.json.

    Every cell uses the same seed, so episode i draws the same perception error in every row and
    every sweep: compare variants pair by pair (``outcomes``), not by the rates alone.
    """
    cl = [float(c) for c in clearances.split(",")]
    sets = {k: v for k, v in FACTOR_SETS.items() if not only or k in only.split(",")}
    rows, outcomes, pens = {}, {}, {}
    worst_ok = worst_fail = 0.0
    pool = ProcessPoolExecutor()
    typer.echo(f"n = {n} per cell; 95 % Wilson interval in brackets")
    typer.echo(f"{'factors':22} " + " ".join(f"{c:>14.1f}mm" for c in cl))
    for name, f in sets.items():
        f = Factors(**{**asdict(f), "chamfer_mm": chamfer})
        rates, oks = [], []
        for c in cl:
            res = run_cell(f, c, chamfer, friction, n, seed, pool)
            ok = [r[0] for r in res]
            # a deep contact only fakes a result if the episode succeeded through it
            p_ok = max([r[1] for r in res if r[0]], default=0.0)
            p_fail = max([r[1] for r in res if not r[0]], default=0.0)
            pens.setdefault(name, []).append([p_ok, p_fail])
            worst_ok, worst_fail = max(worst_ok, p_ok), max(worst_fail, p_fail)
            oks.append(ok)
            rates.append(float(np.mean(ok)))
        rows[name], outcomes[name] = rates, oks
        cells = []
        for ok in oks:
            lo, hi = wilson(sum(ok), n)
            cells.append(f"{100 * np.mean(ok):4.0f}% [{100 * lo:2.0f}-{100 * hi:3.0f}]")
        typer.echo(f"{name:22} " + " ".join(f"{c:>16}" for c in cells))
    pool.shutdown()
    flag = "OK" if worst_ok <= MAX_PEN_MM else "INVALID: a success went through the walls"
    typer.echo(
        f"deepest peg penetration: successes {worst_ok:.2f} mm (limit {MAX_PEN_MM}) -> {flag}; "
        f"failures {worst_fail:.2f} mm"
    )
    bad = [k for k, v in pens.items() if any(c[0] > MAX_PEN_MM for c in v)]
    if bad:
        typer.echo(f"  cells with deep successes: {', '.join(bad)}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"sweep_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(
        json.dumps(
            {
                "clearances_mm": cl,
                "chamfer_mm": chamfer,
                "friction": friction,
                "n": n,
                "seed": seed,
                "timestep": TIMESTEP,
                "max_penetration_mm": {"successes": worst_ok, "failures": worst_fail},
                "penetration_mm": pens,
                "success": rows,
                "outcomes": outcomes,
            },
            indent=1,
        )
    )
    typer.echo(f"-> {path}")


if __name__ == "__main__":
    app()


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs (b: only A ok, c: only B ok)."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


@app.command()
def compare(
    a: Path = typer.Argument(..., help="sweep json (A)"),
    row_a: str = typer.Argument(..., help="factor set in A"),
    b: Path = typer.Argument(..., help="sweep json (B); may be the same file"),
    row_b: str = typer.Argument(..., help="factor set in B"),
) -> None:
    """Paired comparison of two rows, episode by episode (exact McNemar test per clearance)."""
    da, db = json.loads(a.read_text()), json.loads(b.read_text())
    if da.get("seed") != db.get("seed") or da["clearances_mm"] != db["clearances_mm"]:
        raise typer.BadParameter("sweeps differ in seed or clearances: not paired")
    typer.echo(f"A = {row_a} ({a.name}), B = {row_b} ({b.name})")
    for k, cl in enumerate(da["clearances_mm"]):
        oa = np.array(da["outcomes"][row_a][k])
        ob = np.array(db["outcomes"][row_b][k])
        only_a, only_b = int(np.sum(oa & ~ob)), int(np.sum(~oa & ob))
        typer.echo(
            f"{cl:4.1f} mm: A {oa.mean():4.0%}  B {ob.mean():4.0%}  "
            f"only A {only_a:3d}  only B {only_b:3d}  p = {mcnemar_p(only_a, only_b):.3f}"
        )


# ---------------------------------------------------------------- sim commands on the real arm
def scripted_commands(lift_m: float = 0.03, hole_xy=HOLE_XY, hold_s: float = 1.0) -> np.ndarray:
    """The ideal insertion's joint commands (CMD order, rad, 50 Hz): hover, hold, descend at
    1 cm/s to insertion depth, hold. ``lift_m`` raises everything so the real arm, without a peg
    or fixture, replays it in free air above the table."""
    from robot_manipulator.ik.analytic import AnalyticIK

    ik = AnalyticIK()
    top = TABLE_Z + HOLE_H + lift_m

    def q_at(z):
        q = ik.solve(np.r_[hole_xy, z + PEG_LEN], np.pi / 2, ROLL)
        if q is None:
            raise ValueError(f"IK failed at z = {z:.3f} m")
        return q

    rows = [q_at(top + 0.005)] * int(hold_s * HZ)
    z, z_end = top + 0.005, top - DEPTH_M - 0.005
    while z > z_end:
        z = max(z - 0.01 / HZ, z_end)
        rows.append(q_at(z))
    rows += [rows[-1]] * int(hold_s * HZ)
    return np.array(rows)


def peg_tips(model: mujoco.MjModel, qs: np.ndarray) -> np.ndarray:
    """Peg-bottom position (world, m) for each joint vector (forward kinematics only)."""
    d = mujoco.MjData(model)
    adr = joint_qpos_adr(model, JOINTS)
    peg = model.geom("peg").id
    out = []
    for q in qs:
        d.qpos[adr] = q
        mujoco.mj_kinematics(model, d)
        R = d.geom_xmat[peg].reshape(3, 3)
        out.append(d.geom_xpos[peg] + R[:, 0] * PEG_LEN / 2)
    return np.array(out)


@app.command("export-motion")
def export_motion(
    lift_mm: float = typer.Option(30.0, help="raise the whole motion (free-air replay, no peg)"),
    name: str = typer.Option("insertion_replay", help="config/gap_motions/<name>.csv"),
) -> None:
    """Write the scripted insertion as a gap motion, for `gap run --motion <name>` on the arm."""
    from robot_manipulator.sim.gap.motions import write_csv

    m = scripted_commands(lift_m=lift_mm / 1000)
    tips = peg_tips(build(2.0), m)
    fingertip_above_table = 1000 * (tips[:, 2].min() + PEG_LEN - TABLE_Z)
    path = write_csv(REPO / "config" / "gap_motions" / f"{name}.csv", m)
    typer.echo(
        f"{len(m)} ticks ({len(m) / HZ:.1f} s); lowest fingertip {fingertip_above_table:.0f} mm "
        f"above the table (no peg) -> {path}"
    )
    typer.echo(f"replay: uv run gap run --backend real --live --motion {name} --max-vel 0.5")


@app.command("replay-check")
def replay_check(
    run_dir: Path = typer.Argument(..., help="runs/gap/<ts>_real with the replayed motion"),
    name: str = typer.Option("insertion_replay"),
) -> None:
    """Where the sim's insertion commands put the peg tip on the real arm (encoder FK) vs where
    the commands say and where the nominal twin goes. Encoders only: link flex and the tool
    model are not in this number (tip-accuracy measures those)."""
    from robot_manipulator.sim.gap.cli import _load, _sim_from_commands

    real = _load(run_dir / f"{name}.npz")
    sim = _sim_from_commands(real)
    model = build(2.0)
    want, got, twin = (peg_tips(model, x) for x in (real.q_cmd, real.q, sim.q))
    n = min(len(want), len(got), len(twin))
    want, got, twin = want[:n], got[:n], twin[:n]
    hold = slice(n - int(0.5 * HZ), n)  # last 0.5 s of the final hold
    for label, a in (
        ("real vs command", got - want),
        ("real vs twin", got - twin),
        ("twin vs command", twin - want),
    ):
        lat = 1000 * np.linalg.norm(a[:, :2], axis=1)
        z = 1000 * a[:, 2]
        typer.echo(
            f"{label:16}: final lateral {np.median(lat[hold]):5.2f} mm, "
            f"z {np.median(z[hold]):+6.2f} mm | whole motion lateral max {lat.max():5.2f} mm"
        )
    typer.echo("clearance needed to insert without search ~ 2 x final lateral (diametral)")
