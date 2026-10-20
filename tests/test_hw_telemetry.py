import json

import numpy as np
import pyarrow.parquet as pq

from robot_manipulator.harness.units import JointMap
from robot_manipulator.hw.telemetry import JOINTS, REPO, TelemetryLogger


class FakeBus:
    def __init__(self):
        self.reads = 0

    def sync_read(self, reg, normalize=True):
        self.reads += 1
        val = {
            "Present_Load": -120,
            "Present_Current": 40,
            "Present_Temperature": 27,
            "Present_Voltage": 120,
        }[reg]
        return {j: val for j in JOINTS}


def _zero_offset_map(tmp_path):
    """The repo's joint map with all encoder offsets zeroed: this test checks the unit conversion,
    not the calibration values (config/sim2real.yaml has real offsets since 2026-10-20)."""
    import yaml

    cfg = yaml.safe_load((REPO / "config" / "sim2real.yaml").read_text())
    for j in cfg["joints"].values():
        j["offset_deg"] = 0.0
    path = tmp_path / "sim2real_zero.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return JointMap.from_yaml(path)


def test_records_both_unit_systems_and_extras_every_n(tmp_path):
    jm = _zero_offset_map(tmp_path)
    tl = TelemetryLogger("test", "fake_arm", joint_map=jm, extra_every=3, out_dir=tmp_path)
    bus = FakeBus()
    for k in range(7):
        q = {f"{j}.pos": float(k) for j in JOINTS}
        tl.record(q, q_cmd=q, bus=bus)
    path = tl.save()
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == 7
    with_extras = [r for r in rows if r.get("load") is not None]
    assert len(with_extras) == 3  # rows 0, 3, 6
    assert with_extras[0]["load"] == [-120.0] * 6
    assert with_extras[0]["current_ma"] == [260.0] * 6 and with_extras[0]["voltage_v"] == [12.0] * 6
    assert np.allclose(rows[2]["q_meas_rad"][:5], np.deg2rad(2.0))  # sign +1, offset 0
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["n_rows"] == 7 and meta["joint_map"]["calibrated"] in (True, False)


def test_extras_failure_never_raises(tmp_path):
    class BadBus:
        def sync_read(self, *a, **k):
            raise ConnectionError("no status packet")

    tl = TelemetryLogger("test", "fake_arm", out_dir=tmp_path)
    tl.record(np.zeros(6), bus=BadBus())
    assert tl.rows[0]["event"].startswith("extras_failed")
