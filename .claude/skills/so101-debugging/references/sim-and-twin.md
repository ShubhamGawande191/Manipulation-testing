# MuJoCo twin, actuator gap and simulation validity (layer 6)

## A. Simulation validity (results that are too good, or changes that do nothing)

Four bugs in one afternoon of the insertion study (`sim/insertion.py`, notebook 2026-10-20). Each produced a
clean-looking table.

| Symptom | Root cause | Check | Fix |
|---|---|---|---|
| Soft servo 0 % at every clearance | descent started while the arm was still moving; trim frozen during descent | trace joint error at descent start | settle until still (and on target with trim); keep the trim on, bounded |
| Chamfer can't help at mu = 1.0 | MuJoCo default friction 1.0 = 45 deg friction angle = chamfer slope | compare mu with tan(chamfer angle) | set realistic friction (PLA on PLA ~0.35) on both geoms |
| Success above what geometry allows (69 % vs ~15 %) | default 5 ms step + soft contacts: peg pushed 2-4 mm into the walls | log deepest peg contact per episode | 1 ms timestep, solref 2 ms, solimp 0.99/0.999; flag successes > 0.2 mm |
| A 1 mm chamfer changes 1-6 of 100 outcomes | ramp boxes rotated the wrong way (normals facing out) | print each ramp's surface normal | flip the rotation; known-input test (1.2 mm offset jams plain, inserts chamfered) |

Generic checks before trusting any contact sim:
- Analytic bound (geometry/physics by hand) vs the sim number.
- Penetration / conservation logged per episode, with a limit tied to the tolerance under study.
- Convergence: halve the timestep; the answer must not move (35 % vs 33 % here).
- Inspect the geometry you vary (poses, normals, collision masks `contype/conaffinity`).
- Defaults you never chose: friction, solref/solimp, timestep, iterations, `n_sub` = round(1/(hz*dt)).

## B. Twin vs arm: the actuator gap

| Symptom | Check | Action |
|---|---|---|
| Real lags the nominal twin by 80-100 ms | `gap compare` lag per joint | delay grid must cover 0..6 ticks; apply the fitted per-joint delay wherever the fit is used (`apply_actuator_fit` only sets model params) |
| Fit predicts open-loop replay well but its policy hunts on the arm | `gap closed-loop` real vs sim runs | judge twins closed-loop; train on the broad nominal twin + DR, not the fit (Tune to Learn: best sysID fit, worst closed-loop transfer) |
| Pinning kp to the measured sag breaks the fit | params on their bounds | static stiffness (sag, ~18 N.m/rad) is not the P loop's dynamic stiffness; model sag/deadband as separate elements |
| Fit params sit on bounds | | the fit is compensating, not identifying; judge on held-out motions (steps, sine) |
| Calm policy trips the oscillation guard in sim only | sim dithers (nominal friction 8-18x too low) | guard is real-only by default (`--oscillation-guard` forces it) |
| No twin reproduces "stops ~2 deg short by approach" | | needs a play-operator / deadband model (board task `twin-closed-loop-fix`) |

## C. Model and assets

- Vendored Menagerie SO101 @ `3abca2e` + PincOpen variant (`scene_pinc.xml`); tool length may be ~6 mm off
  (fingertip accuracy bias); collision near the fold is untrusted.
- URDF export for cuRobo: `uv run urdf-export` (fixes the converter's [0, 0] limits on unlimited hinges).
- Policy metadata pins scene, gripper hold and action scale; the loader refuses mismatches.

## Circuit breaker

If a sweep's table looks clean but one cell contradicts a bound, stop the sweep and trace one episode
(positions, contacts, geometry) before running more cells. Never "fix" by tuning the sim until the number
looks right - find the mechanism.
