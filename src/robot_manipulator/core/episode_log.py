"""Episode log schema -> parquet (PLAN item 7).

One row per episode. Every blog table in Phases 1-6 is generated from these
parquet files, so the schema here is the single source of truth. wandb is an
optional mirror, never the primary store.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class EpisodeRecord:
    run_id: str
    arm_id: str
    policy: str
    executor: str
    chunk_len: int
    precision: str  # fp32 | bf16 | int8
    policy_latency_ms: float
    e2e_latency_ms: float
    jitter_ms: float
    # scorer outputs (eval/scorer.py ScoreResult, flattened)
    success: bool
    key_press_f1: float | None = None
    velocity_error: float | None = None
    cc_error: float | None = None
    time_to_success_s: float | None = None
    task: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    # sim2real harness additions (defaults keep older records loadable)
    backend: str = ""  # sim | fake | real
    n_steps: int = 0  # control ticks in the episode
    return_: float | None = None  # sum of reward, when the env provides one
    reset_ok: bool | None = None  # real-rig auto-reset succeeded (Phase D)


def write_records(records: list[EpisodeRecord], path: str | Path) -> Path:
    """Write one parquet file (typically runs/<run_id>.parquet)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([asdict(r) for r in records])
    pq.write_table(table, path)
    return path


def load_records(path_or_dir: str | Path) -> list[dict]:
    """Load one parquet file, or concatenate every *.parquet under a directory."""
    import pyarrow.parquet as pq

    p = Path(path_or_dir)
    files = sorted(p.glob("*.parquet")) if p.is_dir() else [p]
    rows: list[dict] = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    return rows
