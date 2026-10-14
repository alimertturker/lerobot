# AGENTS.md: context for agents and developers

What was built in the first working session (2026-10-11 to 2026-10-14), why, and where things stand. Read this
before changing the leader-arm code or this project. User-facing instructions are in `README.md`.

This project lives in `examples/hilserl_so101/` of the lerobot branch `hilserl-fix`; paths to lerobot code
below are relative to the repository root. It started as a separate folder (`~/playground/hilserl-so101`) and
was moved here at the end of the session.

## Goal

HIL-SERL ([paper](https://arxiv.org/abs/2412.09858), [lerobot docs](https://huggingface.co/docs/lerobot/main/en/hilserl))
on a real SO-101 follower, with the **SO-101 leader arm** for demonstrations and human interventions instead
of a gamepad or keyboard. First task: pick up a cube (`"Pick up the cube"`).

## Ground rules from the user

- **All robot motion comes from the leader arm.** The keyboard only switches modes and ends episodes; it never
  moves the robot. Don't propose keyboard/gamepad teleop or motion-based "grab detection" (see Incidents).
- **Ask before any step that moves the robot.** The user runs robot steps from their own terminal.
- Keep follower speed limits during interventions.
- Both arms go to a home pose when a script ends or is interrupted.
- Prefer **short commands** (long ones break when copied from tmux); put long invocations in `scripts/`.
- Commit or push only when asked.

## Code changes in lerobot (branch `hilserl-fix`)

| File                                          | Change                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lerobot/rl/leader_intervention.py` (new) | `LeaderArmInterventionStep`, `LeaderJointActionOverrideStep`, `KeyboardEpisodeEvents`, `ee_delta_action_from_joints`, `LeaderPhase`                                                                                                                                                                                       |
| `src/lerobot/envs/configs.py`                 | `LeaderInterventionConfig` (keys, handover, speed cap, `start_paused`); `ResetConfig.home_joint_positions`, `ResetConfig.release_wait_s`                                                                                                                                                                                  |
| `src/lerobot/rl/gym_manipulator.py`           | `control_mode: "leader"` wiring in `make_processors`; home pose on close; processors reset before `env.reset()`; recording starts at the first take-over; `release_wait_s` hold and gripper-last reset; info merge keeps `leader_phase`; dataset always finalized; `close_env_and_processors`; `init_logging()` in `main` |
| `src/lerobot/rl/actor.py`                     | `atexit` cleanup of env and processors (arms go home on exit)                                                                                                                                                                                                                                                             |
| `tests/rl/test_leader_intervention.py` (new)  | 15 test functions (some parametrized): phases, labels, rate limit, keys, reset/close, start-paused, info merge                                                                                                                                                                                                            |
| `docs/source/hilserl.mdx`                     | Leader-arm section (space/tab handover, config example, home on exit)                                                                                                                                                                                                                                                     |

Test status at the end: `tests/rl tests/envs tests/processor` → 645 passed, 62 skipped; pre-commit clean.

### How leader interventions work

Modelled on the DAgger rollout strategy (`src/lerobot/rollout/strategies/dagger.py`):

```
AUTONOMOUS --space--> PAUSED --tab--> CORRECTING --tab--> PAUSED --space--> AUTONOMOUS
leader parked,        follower holds,  leader torque off,  leader holds        leader parked
torque off            leader smoothly  follower copies it                      at home
                      moved onto it    (rate-limited)
```

- **AUTONOMOUS**: the policy acts; the leader rests at `home_joint_positions` with torque off.
- **PAUSED**: the follower holds its pose; the leader is driven onto the follower (`handover_duration_s`, 1.5 s)
  and holds it with torque, so releasing it later causes no jump.
- **CORRECTING**: leader torque off; the follower target follows the leader, rate-limited to
  `max_joint_speed` (90 deg/s per joint).
- `s` / `q`,`esc` / `r`: success / failure / re-record. An episode end while correcting first makes the leader
  hold. `reset()` (called **before** `env.reset()`) makes the leader safe before the follower moves.
- `start_paused` (recording only): every episode starts PAUSED, so `tab` starts the demo straight away.

### Action labels (important)

The policy action is `[dx, dy, dz, gripper]`: the end-effector move divided by `end_effector_step_sizes`
(0.02 m), clipped to [-1, 1], gripper discrete 0 / 1 / 2 (0 = increase the joint = open on the SO-101).

- `EEReferenceAndDelta(use_ik_solution=True)` takes the **previous commanded joints** as reference, so labels
  are **FK(new target) - FK(previous target)**, not relative to the measured pose. The follower sags about
  5 mm under gravity; measured-pose labels were biased (paused steps labelled 0.24 instead of 0).
- The gripper label compares the leader gripper with the **measured** follower gripper (`gripper_threshold`).
- During an intervention `LeaderJointActionOverrideStep` sends the rate-limited joint target directly and
  resets the IK reference, so the follower tracks the leader exactly while the dataset stores the EE delta.

## Incidents and their fixes

1. **Dangerous drop (2026-10-12).** The first design detected a "grab" when the leader, which was following
   the follower, deviated from it, and then released its torque. With softened gains (P 8, torque limit 300)
   the leader sagged, the sag was read as a grab, torque was released, and the follower copied the falling
   leader. Fix: the DAgger-style keys above, the 90 deg/s rate limit, default leader gains (P 16, D 32, torque
   limit 1000). **No automatic grab detection.**
2. **Episode timeout mid-correction.** The 20 s limit expired while the user corrected, and the follower reset
   while the leader was released. Fix: processors reset before `env.reset()`, longer episodes when recording
   (30 s), episode timer reset after each reset.
3. **Nothing recorded.** `leader_phase` was dropped when the env and action-processor infos were merged, so
   every demo looked like "no take-over" and was discarded. Fix in `step_env_and_process_transition` plus a
   test of that path.
4. **Cube thrown on reset.** The reset pose has the gripper open and all joints moved together. Fix: 3 s hold
   after the episode (`release_wait_s`) to take the cube out, then the arm moves first and the gripper opens
   last.
5. **Keys leaking into bash** (tab completion after exit): `common.sh` flushes the terminal input on exit.
6. **Learner froze the Mac (2026-10-14).** M1, 8 GB: batch 256 with the actor also on MPS; the learner never
   finished a step in 8 min and the actor ran at 3-6 fps. Benchmarked afterwards (batch 64, no robot): CPU
   3.4 s/step, MPS 1.3 s/step. The M1 is fine for the actor, too slow for the learner.

## Project state at the end of the session

- **Configs** (`configs/`): `env_record_leader.json` (15 demos, `start_paused`), `env_test_leader.json` (no
  dataset), `train_hilserl_leader.json` (SAC, demos as offline buffer). Machine values come from `local.env`.
- **Bounds and poses** (original setup): EE box min [0.12, -0.16, -0.012], max [0.32, 0.02, 0.13] m; reset pose
  tip at (0.187, -0.039, 0.076); home pose = folded rest.
- **Demos**: 15 successful episodes, 1420 frames, `outputs/datasets/so101_leader_demos`, pushed (private) to
  the Hub as `alimerido/so101_leader_demos`. Images were stored at 128x128 **uncropped**.
- **Crop step removed**: cropping after recording only cuts the already-shrunk 128x128 image (blurry). If
  cropping is wanted, set `crop_params_dict` from a full-resolution frame **before** recording and re-record.
- **Reward classifier**: skipped; the user presses `s` (the lerobot guide recommends starting this way).
- **First actor/learner run**: connection, interventions and rewards worked (6 episodes, 3 successes after
  take-overs), but the learner never completed a training step on the M1 (incident 6).
- **Portability**: `local.env` (git-ignored) + `local.env.example`; `scripts/common.sh` turns it into
  command-line overrides; `scripts/check_setup.sh` validates a setup without moving anything. Verified that
  every config resolves identically to the earlier hard-coded version on the original Mac.
- **URDF** is not committed (lerobot limits files to 1 MB and keeps `.stl` in LFS); `scripts/get_urdf.sh`
  downloads `so101_new_calib.urdf` and its meshes from TheRobotStudio's SO-ARM100 repo (identical to the copy
  used in this session).

## Open items

1. **Remote learner** (planned, not started; see the collapsible section in `README.md`): learner on an NVIDIA
   GPU (AWS EC2 `g6.xlarge` or similar), actor on the robot machine, SSH tunnel on port 50051.
2. **Smaller transfer format** (needed for a remote learner): each step is sent as about 0.8 MB (float32
   images for state and next state, ~157 MB per episode). Plan: uint8 images, no duplicated next state.
3. **Actor on CPU on Apple Silicon** (`ACTOR_DEVICE=cpu`, about 11 ms per step for the encoder), not yet tried
   on the robot.
4. The lerobot docs (`docs/source/hilserl.mdx`) don't yet mention `start_paused`, `release_wait_s` or
   recording from the first take-over.

## Useful facts

- The learner's "Push parameters to the Actor" log line appears every ~2 s but only sends when new weights are
  queued (every `policy_parameters_push_frequency` = 50 s). gRPC `recvmsg ... Message too long` lines are
  harmless.
- The actor runs the same output-directory check as the learner, so `run_actor.sh` gives it its own
  `outputs/actor/<timestamp>/` folder.
- On macOS OpenCV camera indices can shift when devices are added; `lerobot-find-cameras opencv` or a frame
  grab shows which is which. A one-off "Timed out waiting for frame" at startup usually goes away on retry.
- Grant the terminal **Input Monitoring** on macOS for the keyboard listener, or keep it focused (terminal
  fallback).
