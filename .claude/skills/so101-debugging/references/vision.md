# Cameras, intrinsics and extrinsics (layer 4)

## Device identity and image stability

| Symptom | Check | Action |
|---|---|---|
| Wrong camera / swapped views | `config/cameras.yaml` uses `/dev/v4l/by-path/...` | Both eMeet C950 share serial `SN0001`, so `by-id` can't separate them; keep each camera in its USB port (workspace 2.2, wrist 2.1.4 via hub). Never hardcode `/dev/videoN` (it shuffles on replug). |
| Colours/brightness drift between frames | auto-exposure + dynamic framerate | `uv run cam-setup` locks exposure 300, gain 10, WB 3800 K; **settings reset on replug** - rerun. |
| Camera missing after replug | `uv run cam-setup --list-paths` | re-plug into the same port; update `config/cameras.yaml` only if the port changed |

## Intrinsics (`uv run camcal capture|intrinsics <name>`)

- Pass: RMS < ~0.5 px with >= 15 views covering all 9 regions, fx ~ fy, principal point near centre.
- A wild fit (old `camera_intrinsics.json`: fx 9697 / fy 2109, RMS 5 px) is a broken fit, not a bad camera:
  the same images calibrate to fx ~ 554.
- Board: ChArUco 5x7 `DICT_4X4_50`, 33.2 mm squares / 21.6 mm markers (`config/charuco_board.yaml`).
  Units for any tool expecting meters: 0.0332 / 0.0216.

## Extrinsics (`uv run camcal-wrist plan|run|solve|solve-joint`)

| Symptom | Check | Action |
|---|---|---|
| Two calibration sessions disagree by ~1.5 deg on a joint | board scale | the printed scale is a hidden parameter: `solve-joint --fit-scale` (old board was 2.8 % large) |
| Few usable views | corner threshold | closed fingers hide the near rows; BA uses views with >= 8 corners |
| Residuals small per corner but offset per view (3-15 px) | per-pose error | ~0.5 deg / 2 mm joint play/flex per pose: the arm's accuracy floor, not a calibration bug |
| Planner finds no viewpoints | clearance / roll | planar board guess, targeted aiming, wrist roll within +-60 deg of natural |
| Touch-based extrinsics inconsistent (7-34 mm) | method | prefer `camcal-wrist`; touches need every corner within comfortable reach and saved raw joints |

## Validating the chain end to end

`uv run tip-accuracy run --backend real --live` then `analyze`: raw vs trimmed in-plane error per corner,
play between approach directions. A bias with the same sign at every corner is a model/calibration error
(tool length, board estimate), not the servo; scatter between approaches is play.

Acceptance thresholds worth copying from industrial practice: commit a calibration only when 2D RMS < 1 px
on held-out views (and 3D error under the task tolerance); keep the previous calibration until then.
