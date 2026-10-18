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


def detect(board, gray: np.ndarray, min_corners: int = MIN_CORNERS):
    """ChArUco corners/ids, or (None, None) when fewer than ``min_corners`` are seen."""
    import cv2

    cc, ci, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ci is None or len(ci) < min_corners:
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


REGION_NAMES = (
    ("top-left", "top-centre", "top-right"),
    ("middle-left", "centre", "middle-right"),
    ("bottom-left", "bottom-centre", "bottom-right"),
)


def region_counts(pts: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(3, 3) number of board corners in each third of the image (rows top -> bottom)."""
    h, w = shape[:2]
    counts = np.zeros((3, 3), dtype=int)
    for x, y in np.asarray(pts).reshape(-1, 2):
        counts[min(int(3 * y / h), 2), min(int(3 * x / w), 2)] += 1
    return counts


def missing_regions(counts: np.ndarray, min_corners: int) -> list[str]:
    return [REGION_NAMES[r][c] for r in range(3) for c in range(3) if counts[r, c] < min_corners]


def _draw_regions(img: np.ndarray, counts: np.ndarray, min_corners: int) -> None:
    import cv2

    h, w = img.shape[:2]
    for r in range(3):
        for c in range(3):
            done = counts[r, c] >= min_corners
            x0, y0 = c * w // 3, r * h // 3
            color = (0, 200, 0) if done else (0, 0, 255)
            cv2.rectangle(img, (x0, y0), (x0 + w // 3 - 1, y0 + h // 3 - 1), color, 1)
            cv2.putText(img, str(counts[r, c]), (x0 + 6, y0 + h // 3 - 8), 0, 0.6, color, 2)


@app.command()
def capture(
    name: str,
    target_views: int = typer.Option(25, help="stop after this many new views"),
    max_seconds: float | None = typer.Option(None, help="default 300 s, 900 s --until-covered"),
    append: bool = typer.Option(False, help="keep existing views and add more"),
    until_covered: bool = typer.Option(
        False,
        help="keep going until every image third has --min-corners (overrides --target-views); "
        "only views that add to a missing region are saved",
    ),
    min_corners: int = typer.Option(40, help="with --until-covered: corners needed per region"),
) -> None:
    """Preview + auto-save board views for camera ``name`` (q / Esc to stop early).

    With a headless OpenCV build (no GUI), progress goes to the terminal and the annotated live
    frame to ``runs/camcal/<name>/preview.jpg`` (open it in an image viewer that auto-reloads);
    stop early with Ctrl+C. The preview shows a 3x3 grid with corner counts per region (red =
    still missing with --until-covered).
    """
    import cv2

    cam = load_camera_config()[name]
    apply_controls(cam["path"], cam.get("controls", {}))
    board, _ = make_board()
    out = VIEWS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    counts = np.zeros((3, 3), dtype=int)
    existing = sorted(out.glob("view_*.png"))
    if append:
        for f in existing:  # coverage so far
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            cc, _ = detect(board, g)
            if cc is not None:
                counts += region_counts(cc, g.shape)
        start = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
    else:
        for old in existing:
            old.unlink()
        start = 0
    if until_covered:
        target_views = 10_000
        todo = missing_regions(counts, min_corners)
        typer.echo(f"regions still needed ({min_corners}+ corners each): {', '.join(todo)}")
        if not todo:
            typer.echo("all regions covered already")
            return

    def done() -> bool:
        if until_covered:
            return not missing_regions(counts, min_corners)
        return len(saved) >= target_views

    if max_seconds is None:
        max_seconds = 900.0 if until_covered else 300.0
    cap = cv2.VideoCapture(cam["path"])
    saved: list[tuple[np.ndarray, float]] = []
    t0, last_save, last_preview = time.monotonic(), 0.0, 0.0
    gui = True
    try:
        while not done() and time.monotonic() - t0 < max_seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cc, ci = detect(board, gray)
            show = frame.copy()
            if cc is not None:
                cv2.aruco.drawDetectedCornersCharuco(show, cc, ci)
                now = time.monotonic()
                add = region_counts(cc, gray.shape)
                useful = not until_covered or bool(np.any((add > 0) & (counts < min_corners)))
                if useful and now - last_save > 0.7 and is_new_view(saved, cc):
                    cv2.imwrite(str(out / f"view_{start + len(saved):02d}.png"), gray)
                    pts = cc.reshape(-1, 2)
                    saved.append((pts.mean(0), float(pts.std(0).mean())))
                    counts += add
                    last_save = now
                    if not gui:
                        todo = missing_regions(counts, min_corners)
                        tail = f"still need: {', '.join(todo)}" if todo else "all regions covered"
                        typer.echo(f"\rsaved view {len(saved)} ({len(ci)} corners); {tail}   ")
            n = 0 if ci is None else len(ci)
            _draw_regions(show, counts, min_corners)
            cv2.putText(show, f"saved {len(saved)}  corners {n}", (10, 25), 0, 0.7, (0, 255, 0), 2)
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
                typer.echo(f"\rcorners {n:2d}  saved {len(saved)}", nl=False)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if gui:
            cv2.destroyAllWindows()
    typer.echo(("" if gui else "\n") + f"saved {len(saved)} views -> {out}")
    typer.echo("corners per region (top -> bottom):\n" + str(counts))


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
