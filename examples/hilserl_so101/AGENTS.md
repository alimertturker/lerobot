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

| File                                          | Change                                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lerobot/rl/leader_intervention.py` (new) | `LeaderArmInterventionStep`, `LeaderJointActionOverrideStep`, `KeyboardEpisodeEvents`, `ee_delta_action_from_joints`, `LeaderPhase`                                                                                                                                                                                              |
| `src/lerobot/envs/configs.py`                 | `LeaderInterventionConfig` (keys, handover, speed cap, `start_paused`, `follow_follower`); `ResetConfig.home_joint_positions`, `ResetConfig.release_wait_s`                                                                                                                                                                      |
| `src/lerobot/rl/gym_manipulator.py`           | `control_mode: "leader"` wiring in `make_processors`; home pose on close; processors reset before `env.reset()`; recording starts at the first take-over; `release_wait_s` hold and gripper-last reset; info merge keeps `leader_phase`; dataset always finalized; `close_env_and_processors`; `init_logging()` in `main`        |
| `src/lerobot/rl/actor.py`                     | `atexit` cleanup of env and processors (arms go home on exit)                                                                                                                                                                                                                                                                    |
| `tests/rl/test_leader_intervention.py` (new)  | 19 test functions (some parametrized): phases, labels, rate limit, keys, reset/close, start-paused, info merge                                                                                                                                                                                                                   |
| `src/lerobot/rl/buffer.py`                    | Scale uint8 dataset images to [0, 1] in `from_lerobot_dataset` (offline buffer)                                                                                                                                                                                                                                                  |
| `src/lerobot/transport/utils.py`              | Compact transition format (uint8 images, shared next state); tests in `tests/transport/test_transport_utils.py`                                                                                                                                                                                                                  |
| Wrist joint actions                           | `InverseKinematicsConfig.joint_step_sizes`/`joint_bounds`; `RobotKinematics.inverse_kinematics(locked_joints=)`; joint deltas in `MapTensorToDeltaActionDictStep`, `MapDeltaActionToRobotActionStep`, `InverseKinematicsRLStep`, `ee_delta_action_from_joints`; env action size; tests in `tests/rl/test_wrist_joint_actions.py` |
| `docs/source/hilserl.mdx`                     | Leader-arm section (space/tab handover, config example, home on exit)                                                                                                                                                                                                                                                            |

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

**One-key mode (`follow_follower`, added 2026-10-19, used by the train and test configs).** The user asked for
the leader to always follow the follower, as in the first design, but without grab detection:

```
AUTONOMOUS --space--> CORRECTING --space--> AUTONOMOUS
leader torque on,     leader torque off,    leader holds, then
tracks the follower   follower copies it    rejoins the follower
```

- The take-over stays on a key, so nothing is inferred from leader motion (see incident 1). Default motor
  gains are used (the softened gains were removed), so the holding leader does not sag.
- When the leader is away from the follower (episode start, end of a correction) it first moves onto it at
  `max_joint_speed`, then gets the follower pose as its goal every step.
- An episode end while correcting holds the leader (phase PAUSED until `reset()`); the leader is not parked
  between episodes. `correction_key` and `start_paused` are ignored.

### Action labels (important)

The policy action is `[dx, dy, dz, d_wrist_flex, d_wrist_roll, gripper]` (since 2026-10-21; before it was
`[dx, dy, dz, gripper]`): the end-effector move divided by `end_effector_step_sizes` (0.02 m), the wrist joint
changes divided by `inverse_kinematics.joint_step_sizes` (9°), each clipped to [-1, 1], gripper discrete
0 / 1 / 2 (0 = increase the joint = open on the SO-101).

- **Why the wrist joints**: with xyz only, the follower copied the leader's wrist during take-overs but the
  label had no wrist part, so wrist corrections were lost and the policy could not rotate the gripper. The
  user asked for wrist motion "with real joint angles". The SO-101 has 5 arm joints, so the paper's 6D pose
  action is not reachable; the user chose xyz + direct wrist joint deltas over all-joint deltas or IK pitch/roll.
- **How it is executed**: `InverseKinematicsRLStep(joint_step_sizes=...)` adds the wrist deltas to the
  previous IK solution, clips them to `joint_bounds` (optional, `WRIST_FLEX_RANGE`/`WRIST_ROLL_RANGE` in
  `local.env`), and solves IK for the tip position with the wrist joints **locked**
  (`RobotKinematics.inverse_kinematics(locked_joints=...)`, placo `mask_dof`, orientation ignored). The URDF
  limit of a locked joint is widened to its value for the solve: the real wrist_flex goes past the URDF's 95°
  (every demo reaches about 102°), and placo reports an infeasible QP otherwise; clamping instead moved the
  wrist and made IK compensate with the elbow (7.7° median replay error). A wrist-only action turns the wrist
  with the tip held in place.
- **Labels replay exactly**: the xyz label is FK(target) - FK(previous target) with all joints, which the
  locked-wrist IK reproduces; the 9° wrist step equals the 90°/s take-over speed limit at 10 Hz, so wrist
  labels are not clipped (`tests/rl/test_wrist_joint_actions.py` replays a take-over through the pipeline). Checked
  offline on the 15 xyz demos' joint paths with the real URDF: 1404 steps replay with 0.000° error.
- Without `joint_step_sizes` everything is xyz-only as before (gamepad/keyboard paths unchanged).

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

7. **Leader "Input voltage error" killed the actor (2026-10-19).** In one-key mode the leader holds torque
   all the time; at an episode start its six motors moving together pulled its 5 V supply under the motors'
   4.0 V limit, motor 5 reported it and the exception ended the actor. Fix: leader bus errors are warnings
   and retried next step (`_leader_command`); a failed release on `space` keeps the policy in control. Root
   cause is the leader's power supply (measured 5.0 V at rest).
8. **Demo images on the wrong scale in the learner (2026-10-19).** Upstream `make_dataset` now returns
   uint8 images (`a8b72d96`); `lerobot-train` divides them by 255, but `ReplayBuffer.from_lerobot_dataset`
   did not, so the offline buffer held demos in [0, 255] while the actor's images are in [0, 1]. Training
   mixed two scales, and the first checkpoint (step 5000) crashed when writing the offline buffer as images.
   Fix: `_uint8_image_to_float` in `rl/buffer.py`, test `test_from_lerobot_dataset_scales_uint8_images`.
   The 5000-step run is kept on EC2 as `outputs/train/hilserl_so101_leader_bad_image_scale_20261019`.
9. **Aggressive policy motion and `EE jump 0.084m > 0.05m` crash (2026-10-21, first wrist run).** The user saw
   moves that "should not be possible within this workspace". Causes, found by running random policy actions
   through the action steps offline (real URDF, local box): (a) no joint-speed limit on the policy path (the
   90°/s cap was only for take-overs; even xyz-only reached 39°/step); (b) a wrist step with the tip held makes
   shoulder/elbow compensate ~13° per 9° of flex; (c) IK failures were sent as they were: with the wrist locked
   many targets were missed, the tip left the box and the next target, relative to that pose, jumped. Before
   the fix 26 of 30 simulated episodes crashed and the tip left the box on a third of the steps. Fix:
   `InverseKinematicsRLStep(max_joint_step_deg=9, max_ik_error_m=0.005)` shrinks the whole step (1/2, 1/4,
   1/8) until it fits, else holds; `EEBoundsAndSafety(max_ee_step_m=None)` then (its target-vs-target check
   misfires after a shrunk step). After: max 9°/step, tip never > 5 mm outside the box, no crashes; the demo
   joint paths still replay (one shrunk step). Moving the wrist without arm compensation was tried and
   rejected: it pushed the tip out of the box.
   Follow-up the same evening: after a take-over that left the wrist outside `joint_bounds`, the first fix
   clipped it back in one step (> 9°), so every step was held for the rest of the episode ("IK could not
   reach the target … error 0.0 mm, joint step 10–11°"). Now a joint outside its bounds may stay or move
   back towards them, not snap; the warning names the joint.
10. **Learner killed on resume, out of memory (2026-10-22).** `ec2_learner.sh start --resume=true` stopped
    after "Resume training load the online dataset" with no error in the log; `dmesg` showed the kernel's OOM
    kill at 15 GB (g5.xlarge, 16 GB, no swap). `ReplayBuffer.from_lerobot_dataset` first built a list of all
    14,410 transitions, each with its image and next image as float32 (~11 GB), then copied them into the
    buffer (another 5.7 GB). The live buffer also stored images as float32, so even without resume it would
    have run out at about 25–30k frames. Fix (`rl/buffer.py`): transitions are added one by one (each frame
    decoded once), and `ReplayBuffer(store_images_as_uint8=True)` (set by the learner for the online and
    offline buffers) stores images as uint8 and turns them back into [0, 1] floats when sampling; exact for
    camera images. 30k online frames now take about 3 GB.
11. **Success swinging between policy versions (2026-10-22).** The user saw some weight versions do much
    better than others. From 393 episodes plus the step-20000 buffer: (a) every actor start ran the
    untrained policy until the first weights arrived (the actor also loads weights only at episode ends): 3
    episodes in one run, a whole 10-episode run in another; (b) each checkpoint saved both replay buffers
    synchronously, about 5 min at step 20000, with no training and no weight pushes meanwhile (that was
    the 10-episode run); (c) cube start position matters: next to the arm 0% policy-alone success (9
    episodes), middle 42% (89), far 21% (82) for episodes ≥ 150; (d) the rest is noise (sampled actions,
    ~4 episodes per weight version). Temperature fell steadily (0.0029 → 0.0005), so not more randomness.
    Fixes: the actor waits for and loads the learner's weights before its first reset
    (`wait_for_policy_parameters`); `buffer_save_freq` (20000 in the train config) saves the buffers only
    every that many steps and once more on stop (Ctrl+C; `ec2_learner.sh stop` now waits up to 10 min),
    policy checkpoints stay every 5000; buffers are written to `dataset.tmp` and swapped in, so an
    interrupted save keeps the previous copy. A crash loses the online data since the last buffer save.
    (These settings only reached resumed runs after incident 12's fix; before it, a resume used the
    checkpoint's config.)
12. **Every resume restarted the policy from scratch (2026-10-22).** The user saw the policy turn "terrible"
    after a resume. `handle_resume_logic` replaces the config with the checkpoint's `train_config.json`,
    whose `policy.pretrained_path` is None, so `make_policy` built a fresh policy: checked on the step-20000
    checkpoint, 0 of its 36 trained tensors (actor, image spatial embeddings and post-encoders, state encoder,
    discrete critic) were restored, while the critic heads, the Adam state and the temperature (0.0005)
    were. The run resumed at step 15000 (21:53) relearned from the 19k-step buffer: mostly failures for
    ~2,700 updates, 70% at steps ~17.7k–19.5k, then 13 failures in a row; the 20000 checkpoint is that
    relearned policy. The resume at 20000 (22:50) was random again. The same resume also ignored config
    changes (push frequency, `buffer_save_freq`). Fixes: `make_learner_policy` loads
    `checkpoints/last/pretrained_model` on resume (test fails on the old code); `RESUME_RUNTIME_SETTINGS`
    (saving, logging, wandb, `actor_learner_config`) come from the given config; the learner logs averaged
    losses, Q and temperature every `log_freq`. Compared with rail-berkeley/hil-serl (which restores the
    whole agent): weights now go out every 10 s (theirs: every 50 learner steps) and the actor uses them
    from the next step (theirs: on arrival), not at the episode end; the frozen ResNet-10 is no longer
    pushed (23 MB → ~4 MB). Rolled back to the 15000 weights (end of the 21 Oct run) with all buffers;
    `checkpoints_set_aside/0020000_policy_reset_on_resume` keeps the old 20000. Data: 197 of 330 online
    episodes ended with `q` (terminal, value 0), 97 of them within 3 s; take over instead. A 10 s episode
    limit (theirs: 100 steps) would cut 19 of 61 corrected successes, so it stays 20 s.

## Project state at the end of the session

- **Configs** (`configs/`): `env_record_leader.json` (15 demos, `start_paused`), `env_test_leader.json` (no
  dataset), `train_hilserl_leader.json` (SAC, demos as offline buffer). Machine values come from `local.env`.
- **Bounds and poses** (original setup): EE box min [0.12, -0.16, -0.012], max [0.32, 0.02, 0.13] m; reset pose
  tip at (0.187, -0.039, 0.076); home pose = folded rest.
- **Demos**: 15 successful episodes, 1420 frames, `outputs/datasets/so101_leader_demos`, pushed (private) to
  the Hub as `alimerido/so101_leader_demos`. Images were stored at 128x128 **uncropped**. These are xyz-only
  (4-number actions). `scripts/convert_demos_wrist.sh` turned them into `outputs/datasets/so101_leader_demos_wrist`
  (2026-10-21): xyz and gripper kept, wrist deltas = measured wrist change to the next frame / 9° (no step was
  clipped; largest 7.3°), 731 of 1420 steps move the wrist. Episodes 7 and 10 have no success reward (already
  so in the original).
- **xyz-only run (2026-10-19/20)**: 74 min of robot time, success rose from 20% to 52% in the last session,
  14 of the last 31 episodes without take-over. Kept on EC2 as `outputs/train/hilserl_so101_leader`; rename it
  before starting the wrist run.
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

1. **Remote learner: done (2026-10-19).** `scripts/ec2_setup.sh`, `ec2_learner.sh`, `run_actor_remote.sh`,
   settings `EC2_HOST/EC2_KEY/EC2_DIR` in `local.env`. Tested without the robot against the user's A10G
   instance (`g5.2xlarge`, eu-west-2): tunnel round trip 101 ms, 3 fake episodes uploaded at 24.5 Mbit/s,
   weights (45 MB) received, learner at 3.66 updates/s at batch 256 with the GPU at 94%. The fake run's output
   was deleted. Not yet run with the robot.
2. **Compact transfer format: done.** `transport/utils.py` `transitions_to_bytes` sends images in [0, 1] as
   uint8 and a `next_state` entry equal to the next transition's `state` once (10 MB instead of ~80 MB per
   100 steps); `bytes_to_transitions` still reads plain lists.
3. **Actor on CPU on Apple Silicon** (`ACTOR_DEVICE=cpu`, about 11 ms per step for the encoder), not yet tried
   on the robot.
4. **Wrist actions: code done (2026-10-21).** `test_leader.sh` ran cleanly with the 6-number action; wrist ranges
   set from calibration + demos (flex [60, 104], roll [20, 110]); demos converted. Next: `ec2_setup.sh`, remove
   the empty `outputs/train/hilserl_so101_leader` left by a failed start on EC2, fresh learner run.
5. The lerobot docs (`docs/source/hilserl.mdx`) don't yet mention `start_paused`, `release_wait_s` or
   recording from the first take-over.

## Useful facts

- The learner's "Push parameters to the Actor" log line appears every ~2 s but only sends when new weights are
  queued (every `policy_parameters_push_frequency` = 10 s). gRPC `recvmsg ... Message too long` lines are
  harmless.
- The actor runs the same output-directory check as the learner, so `run_actor.sh` gives it its own
  `outputs/actor/<timestamp>/` folder.
- On macOS OpenCV camera indices can shift when devices are added; `lerobot-find-cameras opencv` or a frame
  grab shows which is which. A one-off "Timed out waiting for frame" at startup usually goes away on retry.
- Grant the terminal **Input Monitoring** on macOS for the keyboard listener, or keep it focused (terminal
  fallback).
