"""Camera intrinsics from a ChArUco board (``uv run camcal capture|intrinsics``).

    uv run camcal capture workspace     # preview window; move the board around; auto-saves views
    uv run camcal intrinsics workspace  # fit K + distortion, reject outlier views, write yaml

Board spec: ``config/charuco_board.yaml``. Cameras: ``config/cameras.yaml`` (stable by-path).
Output: ``config/camera_intrinsics/<name>.yaml``. Views: ``runs/camcal/<name>/view_XX.png``.

Capture tips: fill different parts of the image (corners too), tilt the board up to ~45° in
both axes, vary distance. A view is saved only when >= MIN_CORNERS corners are seen and the
board moved noticeably since the last saved view.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import numpy as np
import typer
import yaml

from robot_manipulator.cameras import REPO, apply_controls, load_camera_config

BOARD_CFG = REPO / "config" / "charuco_board.yaml"
OUT_DIR = REPO / "config" / "camera_intrinsics"
VIEWS_DIR = REPO / "runs" / "camcal"
MIN_CORNERS = 12
MIN_MOVE_PX = 40.0  # centroid shift or scale change needed between saved views
OUTLIER_FACTOR = 2.5  # drop views whose error exceeds this x the median, then refit

app = typer.Typer(add_completion=False, help=__doc__)


def make_board(cfg_path: Path = BOARD_CFG):
    import cv2

    cfg = yaml.safe_load(Path(cfg_path).read_text())
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (cfg["squares_x"], cfg["squares_y"]),
        cfg["square_length_m"],
        cfg["marker_length_m"],
        dic,
    )
    return board, cfg


def detect(board, gray: np.ndarray):
    """ChArUco corners/ids, or (None, None) when the board is not usable."""
    import cv2

    cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ci is None or len(ci) < MIN_CORNERS:
        return None, None
    return cc, ci


def is_new_view(prev: list[tuple[np.ndarray, float]], corners: np.ndarray) -> bool:
    """True if the board centroid moved >= MIN_MOVE_PX or its spread changed >= 20 %."""
    pts = corners.reshape(-1, 2)
    c, spread = pts.mean(0), float(pts.std(0).mean())
    for pc, ps in prev:
        if np.linalg.norm(c - pc) < MIN_MOVE_PX and abs(spread - ps) / max(ps, 1e-6) < 0.2:
            return False
    return True


def calibrate(board, views: list[np.ndarray], size: tuple[int, int]) -> dict:
    """Fit K/dist (k3 fixed: few views overfit it), drop outlier views once, refit."""
    import cv2

    obj, img = [], []
    for gray in views:
        cc, ci = detect(board, gray)
        if cc is not None:
            o, p = board.matchImagePoints(cc, ci)
            obj.append(o)
            img.append(p)
    if len(obj) < 6:
        raise RuntimeError(f"only {len(obj)} usable views; capture more (>= 15 recommended)")
    flags = cv2.CALIB_FIX_K3

    def fit(o, p):
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(o, p, size, None, None, flags=flags)
        errs = []
        for oi, pi, r, t in zip(o, p, rvecs, tvecs):
            proj, _ = cv2.projectPoints(oi, r, t, K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((proj - pi) ** 2, axis=-1)))))
        return rms, K, dist, np.array(errs)

    rms, K, dist, errs = fit(obj, img)
    keep = errs <= OUTLIER_FACTOR * np.median(errs)
    if keep.sum() >= 6 and not keep.all():
        rms, K, dist, errs = fit(
            [o for o, k in zip(obj, keep) if k], [p for p, k in zip(img, keep) if k]
        )
    return {
        "image_size": list(size),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.ravel().tolist(),
        "rms_px": float(rms),
        "views_used": int(len(errs)),
        "views_dropped": int((~keep).sum()),
        "per_view_rms_px": [round(float(e), 3) for e in errs],
        "date": date.today().isoformat(),
        "flags": "CALIB_FIX_K3",
    }


@app.command()
def capture(
    name: str,
    target_views: int = 25,
    max_seconds: float = 300.0,
    append: bool = typer.Option(False, help="keep existing views and add target_views more"),
) -> None:
    """Preview + auto-save board views for camera ``name`` (q / Esc to stop early).

    With a headless OpenCV build (no GUI), progress goes to the terminal and the annotated live
    frame to ``runs/camcal/<name>/preview.jpg`` (open it in an image viewer that auto-reloads);
    stop early with Ctrl+C.
    """
    import cv2

    cam = load_camera_config()[name]
    apply_controls(cam["path"], cam.get("controls", {}))
    board, _ = make_board()
    out = VIEWS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    start = 0
    if append:
        start = len(list(out.glob("view_*.png")))
    else:
        for old in out.glob("view_*.png"):
            old.unlink()
    cap = cv2.VideoCapture(cam["path"])
    saved: list[tuple[np.ndarray, float]] = []
    t0, last_save, last_preview = time.monotonic(), 0.0, 0.0
    gui = True
    cells: set[tuple[int, int]] = set()  # 3x3 image regions the saved views covered
    try:
        while len(saved) < target_views and time.monotonic() - t0 < max_seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cc, ci = detect(board, gray)
            show = frame.copy()
            if cc is not None:
                cv2.aruco.drawDetectedCornersCharuco(show, cc, ci)
                now = time.monotonic()
                if now - last_save > 0.7 and is_new_view(saved, cc):
                    cv2.imwrite(str(out / f"view_{start + len(saved):02d}.png"), gray)
                    pts = cc.reshape(-1, 2)
                    saved.append((pts.mean(0), float(pts.std(0).mean())))
                    last_save = now
                    h, w = gray.shape
                    cx, cy = pts.mean(0)
                    cells.add((min(int(3 * cy / h), 2), min(int(3 * cx / w), 2)))
                    if not gui:
                        typer.echo(
                            f"\rsaved view {len(saved)}/{target_views} "
                            f"({len(ci)} corners; image regions covered {len(cells)}/9)   "
                        )
            n = 0 if ci is None else len(ci)
            cv2.putText(
                show,
                f"saved {len(saved)}/{target_views}  corners {n}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            if gui:
                try:
                    cv2.imshow(f"camcal {name}", show)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                except cv2.error:
                    gui = False
                    typer.echo(
                        f"no GUI in this OpenCV build: live preview -> {out / 'preview.jpg'}; "
                        "Ctrl+C to stop early"
                    )
            if not gui and time.monotonic() - last_preview > 0.5:
                cv2.imwrite(str(out / "preview.jpg"), show)
                last_preview = time.monotonic()
                typer.echo(f"\rcorners {n:2d}  saved {len(saved)}/{target_views}", nl=False)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if gui:
            cv2.destroyAllWindows()
    typer.echo(("" if gui else "\n") + f"saved {len(saved)} views -> {out}")


@app.command()
def intrinsics(name: str) -> None:
    """Calibrate from ``runs/camcal/<name>/`` and write ``config/camera_intrinsics/<name>.yaml``."""
    import cv2

    board, _ = make_board()
    files = sorted((VIEWS_DIR / name).glob("view_*.png"))
    views = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
    if not views:
        raise typer.BadParameter(f"no views in {VIEWS_DIR / name}; run `camcal capture {name}`")
    h, w = views[0].shape
    res = calibrate(board, views, (w, h))
    res["camera"] = name
    res["device"] = load_camera_config()[name]["path"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(res, sort_keys=False))
    K = np.array(res["camera_matrix"])
    hfov = 2 * np.degrees(np.arctan(w / 2 / K[0, 0]))
    typer.echo(
        f"{name}: rms {res['rms_px']:.3f} px over {res['views_used']} views "
        f"(dropped {res['views_dropped']}), fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} "
        f"cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}, HFOV {hfov:.1f} deg -> {path}"
    )


if __name__ == "__main__":
    app()
