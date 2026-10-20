import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mjcf_urdf_simple_converter")

from robot_manipulator.sim.urdf_export import export  # noqa: E402


@pytest.mark.parametrize("mimic", [True, False])
def test_export_adds_fingertip_limits_and_optional_mimic(tmp_path, mimic):
    urdf = export(out=tmp_path, mimic=mimic)
    root = ET.parse(urdf).getroot()
    assert any(link.get("name") == "fingertip" for link in root.findall("link"))
    joints = {j.get("name"): j for j in root.findall("joint")}
    for finger in ("pinc_right_arm", "pinc_left_finger"):
        lim = joints[finger].find("limit")
        assert float(lim.get("lower")) < float(lim.get("upper"))  # not the converter's [0, 0]
        assert (joints[finger].find("mimic") is not None) == mimic
    assert all(m.get("filename") for m in root.iter("mesh"))
