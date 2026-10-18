"""Workspace atlas: what the arm can do at each place in its workspace.

Layer 1 here is the **physics prior** from the twin. It samples joint configurations of the
default model (PincOpen), drops those in contact with the table or the arm, and for each one
computes:

- ``margin``: ``1 - max_j |tau_gravity_j| / tau_max_j`` over the arm joints. Holding torque
  headroom: 1 = no load, <= 0 = cannot hold the pose.
- ``payload_kg``: largest extra mass at the EE site that keeps every joint within ``tau_max``.
- ``push_n``: largest downward force the fingertips can press into a surface. Static balance
  ``tau = tau_g - J_z^T F``.
- ``down``: whether the tool axis (palm -> fingertips) points within ``DOWN_DEG`` of straight
  down. Picking from above and pressing keys need this.

Configurations are binned by EE position into ``VOXEL`` cubes. Each voxel keeps the best value
over its configurations, overall and for tool-down ones. Real telemetry (``hw/telemetry.py``)
is layer 2: measured load and tracking error that correct this prior.

    uv run atlas prior [--n 200000]          # -> runs/atlas/sim_prior.parquet + heatmaps
    uv run atlas query 0.25 0.0 0.05 [--down]

Static estimates only: no dynamics, friction or servo current limits. tau_max comes from the
actuator forcerange (STS3215 at 12 V: 2.94 N·m). Self-collision near the folded pose is
unreliable (coarse Menagerie geoms). See docs/DIGITAL-TWIN-RUNBOOK.md §4.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
import typer

from robot_manipulator.sim.model import (
    EE_SITE,
    JOINTS,
    apply_joint_couplings,
    joint_dof_adr,
    joint_qpos_adr,
    joint_ranges,
    load_model,
)

REPO = Path(__file__).resolve().parents[3]
OUT_DIR = REPO / "runs" / "atlas"
VOXEL = 0.025  # m
DOWN_DEG = 25.0
GRAVITY = 9.81
N_ARM = 5
# grid bounds (base frame, m); the table is at z = -0.0275
LO = np.array([-0.15, -0.45, -0.03])
HI = np.array([0.50, 0.45, 0.50])

app = typer.Typer(add_completion=False, help=__doc__)


def max_scalar(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> float:
    """Largest s >= 0 with |a_j + s*b_j| <= t_j for all j (0 if s = 0 already violates)."""
    if np.any(np.abs(a) > t):
        return 0.0
    hi = np.inf
    for aj, bj, tj in zip(a, b, t):
        if abs(bj) < 1e-12:
            continue
        s1, s2 = (tj - aj) / bj, (-tj - aj) / bj
        hi = min(hi, max(s1, s2))
    return float(max(hi, 0.0)) if np.isfinite(hi) else float("inf")


def evaluate(model: mujoco.MjModel, qs: np.ndarray) -> dict[str, np.ndarray]:
    """Per configuration: ee, tool_down_cos, margin, payload_kg, push_n, contact (bool)."""
    d = mujoco.MjData(model)
    qadr = joint_qpos_adr(model, JOINTS)
    dadr = joint_dof_adr(model, JOINTS)[:N_ARM]
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS[:N_ARM]]
    tau_max = np.array([model.actuator_forcerange[a][1] for a in act])
    jacp = np.zeros((3, model.nv))
    n = len(qs)
    out = {
        "ee": np.zeros((n, 3)),
        "tool_down_cos": np.zeros(n),
        "margin": np.zeros(n),
        "payload_kg": np.zeros(n),
        "push_n": np.zeros(n),
        "contact": np.zeros(n, dtype=bool),
    }
    for i, q in enumerate(qs):
        d.qpos[:] = model.qpos0
        d.qpos[qadr] = q
        apply_joint_couplings(model, d)
        d.qvel[:] = 0
        mujoco.mj_forward(model, d)
        out["contact"][i] = d.ncon > 0
        out["ee"][i] = d.site_xpos[site]
        # tool axis = x axis of the PincOpen base frame (palm -> fingertips)
        x_axis = d.body("pinc_base").xmat.reshape(3, 3)[:, 0]
        out["tool_down_cos"][i] = -x_axis[2]
        tau_g = d.qfrc_bias[dadr].copy()
        mujoco.mj_jacSite(model, d, jacp, None, site)
        jz = jacp[2, dadr]
        out["margin"][i] = 1.0 - float(np.max(np.abs(tau_g) / tau_max))
        out["payload_kg"][i] = max_scalar(tau_g, GRAVITY * jz, tau_max) / GRAVITY
        out["push_n"][i] = max_scalar(tau_g, -jz, tau_max)
    return out


def build_prior(n: int = 200_000, seed: int = 0, model: mujoco.MjModel | None = None):
    """Sample, evaluate, voxelize. Returns a list of per-voxel dict rows."""
    model = model if model is not None else load_model()
    rng = np.random.default_rng(seed)
    lo, hi = joint_ranges(model, JOINTS)
    qs = np.zeros((n, len(JOINTS)))
    qs[:, :N_ARM] = rng.uniform(lo[:N_ARM] + 0.05, hi[:N_ARM] - 0.05, size=(n, N_ARM))
    qs[:, N_ARM] = 0.0  # gripper closed (fingertips together, EE site meaningful)
    ev = evaluate(model, qs)
    ok = ~ev["contact"] & np.all((ev["ee"] >= LO) & (ev["ee"] < HI), axis=1)
    down = ev["tool_down_cos"] >= np.cos(np.radians(DOWN_DEG))
    keys = np.floor((ev["ee"] - LO) / VOXEL).astype(int)
    rows: dict[tuple, dict] = {}
    for i in np.flatnonzero(ok):
        k = tuple(keys[i])
        r = rows.setdefault(
            k,
            {
                "x": LO[0] + (k[0] + 0.5) * VOXEL,
                "y": LO[1] + (k[1] + 0.5) * VOXEL,
                "z": LO[2] + (k[2] + 0.5) * VOXEL,
                "n": 0,
                "margin": -np.inf,
                "payload_kg": 0.0,
                "push_n": 0.0,
                "n_down": 0,
                "margin_down": -np.inf,
                "payload_kg_down": 0.0,
                "push_n_down": 0.0,
            },
        )
        r["n"] += 1
        r["margin"] = max(r["margin"], ev["margin"][i])
        r["payload_kg"] = max(r["payload_kg"], ev["payload_kg"][i])
        r["push_n"] = max(r["push_n"], ev["push_n"][i])
        if down[i]:
            r["n_down"] += 1
            r["margin_down"] = max(r["margin_down"], ev["margin"][i])
            r["payload_kg_down"] = max(r["payload_kg_down"], ev["payload_kg"][i])
            r["push_n_down"] = max(r["push_n_down"], ev["push_n"][i])
    out = []
    for r in rows.values():
        for k in ("margin_down",):
            if not np.isfinite(r[k]):
                r[k] = None
        out.append(r)
    stats = {
        "n": n,
        "valid": int(ok.sum()),
        "contact": int(ev["contact"].sum()),
        "voxels": len(out),
    }
    return out, stats


def query(rows: list[dict], xyz, down: bool = False) -> dict | None:
    """Voxel row containing ``xyz`` (None if never reached in the sample)."""
    k = np.floor((np.asarray(xyz, dtype=float) - LO) / VOXEL).astype(int)
    c = LO + (k + 0.5) * VOXEL
    for r in rows:
        if np.allclose([r["x"], r["y"], r["z"]], c, atol=1e-9):
            if down and not r["n_down"]:
                return None
            return r
    return None


def _heatmaps(rows: list[dict], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heights = [0.0, 0.05, 0.10, 0.20]  # base frame; table at -0.0275
    fig, axes = plt.subplots(2, len(heights), figsize=(4.2 * len(heights), 8), squeeze=False)
    nx, ny = int(round((HI[0] - LO[0]) / VOXEL)), int(round((HI[1] - LO[1]) / VOXEL))
    for col, h in enumerate(heights):
        kz = int(np.floor((h - LO[2]) / VOXEL))
        for row, (key, label, vmax) in enumerate(
            (
                ("push_n_down", "max downward push, tool down [N]", 15.0),
                ("payload_kg", "max payload, any orientation [kg]", 1.0),
            )
        ):
            grid = np.full((ny, nx), np.nan)
            for r in rows:
                if int(np.floor((r["z"] - LO[2]) / VOXEL)) != kz or not r.get(key):
                    continue
                ix = int(np.floor((r["x"] - LO[0]) / VOXEL))
                iy = int(np.floor((r["y"] - LO[1]) / VOXEL))
                grid[iy, ix] = r[key]
            ax = axes[row][col]
            im = ax.imshow(
                grid,
                origin="lower",
                extent=[LO[0], HI[0], LO[1], HI[1]],
                vmin=0,
                vmax=vmax,
                cmap="viridis",
            )
            ax.plot(0, 0, "r+", ms=12)  # robot base
            ax.set_title(f"{label}\nz = {h * 100:.0f} cm above base")
            ax.set_xlabel("x forward [m]")
            ax.set_ylabel("y left [m]")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = out / "sim_prior_maps.png"
    fig.savefig(path, dpi=90)
    return path


@app.command()
def prior(n: int = 200_000, seed: int = 0) -> None:
    """Build the physics prior and write runs/atlas/sim_prior.parquet + heatmaps."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.monotonic()
    rows, stats = build_prior(n=n, seed=seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in keys}), OUT_DIR / "sim_prior.parquet")
    png = _heatmaps(rows, OUT_DIR)
    typer.echo(f"{stats} in {time.monotonic() - t0:.1f}s -> {OUT_DIR / 'sim_prior.parquet'}, {png}")


@app.command("query")
def query_cmd(x: float, y: float, z: float, down: bool = False) -> None:
    """Look up the voxel at (x, y, z) in the base frame [m]."""
    import pyarrow.parquet as pq

    rows = pq.read_table(OUT_DIR / "sim_prior.parquet").to_pylist()
    r = query(rows, (x, y, z), down=down)
    typer.echo("not reached in the sample" if r is None else r)


@app.command()
def real() -> None:
    """Aggregate real telemetry into runs/atlas/real.parquet (+ load calibration vs the twin)."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from robot_manipulator.sim.atlas_real import build_real

    rows, calib, paths = build_real()
    if not rows:
        typer.echo("no telemetry in runs/telemetry yet")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    pq.write_table(pa.table({k: [r.get(k) for r in rows] for k in keys}), OUT_DIR / "real.parquet")
    (OUT_DIR / "real_calibration.json").write_text(json.dumps(calib, indent=2) + "\n")
    typer.echo(f"{len(paths)} telemetry files, {len(rows)} voxels -> {OUT_DIR / 'real.parquet'}")
    typer.echo("measured servo load vs twin gravity load (measured ≈ gain x predicted):")
    for j, v in calib.items():
        typer.echo(f"  {j:14} {v}")


if __name__ == "__main__":
    app()
