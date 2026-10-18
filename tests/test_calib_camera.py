import glob

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from robot_manipulator.calib.camera import calibrate, is_new_view, make_board  # noqa: E402

OLD = sorted(f for f in glob.glob("calibration_images/calib_*.jpg") if "debug" not in f)


def test_new_view_gate():
    pts = np.array([[[100.0, 100.0]], [[200.0, 100.0]], [[100.0, 200.0]], [[200.0, 200.0]]])
    prev = [(pts.reshape(-1, 2).mean(0), float(pts.reshape(-1, 2).std(0).mean()))]
    assert not is_new_view(prev, pts + 5)  # barely moved
    assert is_new_view(prev, pts + 100)  # moved
    assert is_new_view(prev, pts * 1.5)  # much bigger on screen


@pytest.mark.skipif(len(OLD) < 10, reason="needs the archived calibration_images/")
def test_calibrates_archived_board_images_to_sane_intrinsics():
    board, _ = make_board()
    views = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in OLD]
    res = calibrate(board, views, (640, 480))
    K = np.array(res["camera_matrix"])
    assert res["rms_px"] < 1.5 and res["views_used"] >= 15
    assert K[0, 0] == pytest.approx(K[1, 1], rel=0.03)  # square pixels
    assert 450 < K[0, 0] < 650 and 280 < K[0, 2] < 360 and 200 < K[1, 2] < 280


def test_region_counts_and_missing():
    from robot_manipulator.calib.camera import missing_regions, region_counts

    pts = np.array([[10, 10], [630, 10], [320, 240], [320, 240], [630, 470]], dtype=float)
    c = region_counts(pts, (480, 640))
    assert c[0, 0] == 1 and c[0, 2] == 1 and c[1, 1] == 2 and c[2, 2] == 1 and c.sum() == 5
    assert missing_regions(c, 2) == [
        "top-left",
        "top-centre",
        "top-right",
        "middle-left",
        "middle-right",
        "bottom-left",
        "bottom-centre",
        "bottom-right",
    ]
