# HIL-SERL on SO-101 with leader-arm interventions

Real-robot HIL-SERL ([docs](https://huggingface.co/docs/lerobot/main/en/hilserl)) on an SO-101 follower,
using the SO-101 **leader arm** for demonstrations and interventions (`control_mode: "leader"`). This folder
lives in `examples/hilserl_so101/` of the lerobot branch `hilserl-fix`, which also holds the leader-arm code;
see `AGENTS.md` for what was built and why. Run every command from this folder.

Task: pick up a cube. Two cameras (`wrist`, `top`), images resized to 128x128, no crop.

Action (6 numbers per step at 10 Hz): `[dx, dy, dz, d_wrist_flex, d_wrist_roll, gripper]`. `dx, dy, dz` move
the gripper tip by up to 2 cm (`end_effector_step_sizes`); IK solves them on shoulder pan, shoulder lift and
elbow with the wrist held. The two wrist joints are commanded directly, up to 9° per step
(`inverse_kinematics.joint_step_sizes`), so a wrist-only action turns the wrist in place. `gripper` is
open / stay / close.

The policy never moves a joint more than 9° per step (`inverse_kinematics.max_joint_step_deg`, the same 90°/s
as take-overs). A step that would need more, or that IK can't reach within 5 mm (`max_ik_error_m`), is shrunk
to 1/2, 1/4 or 1/8 along the same straight line, or skipped (the arm holds, and the log says "IK could not
reach the target"). So the tip stays inside the `EE_BOUNDS` box.

## Project layout

| Path                | What                                                                     | In git               |
| ------------------- | ------------------------------------------------------------------------ | -------------------- |
| `configs/*.json`    | lerobot configs, shared by every setup                                   | yes                  |
| `local.env.example` | template for the per-machine settings                                    | yes                  |
| `local.env`         | **your** machine and setup settings                                      | **no** (git-ignored) |
| `scripts/`          | one script per step; all load `local.env` through `scripts/common.sh`    | yes                  |
| `assets/SO101/`     | SO-101 URDF and meshes (kinematics), downloaded by `scripts/get_urdf.sh` | **no**               |
| `outputs/`          | datasets, logs, checkpoints                                              | **no**               |

## Setting up on a new machine

1. **Install lerobot from the `hilserl-fix` branch** in a virtual environment (Python 3.12+), then get the
   SO-101 URDF (from TheRobotStudio's SO-ARM100 repo, as the lerobot docs recommend):
   ```bash
   git clone <your lerobot fork> lerobot && cd lerobot && git checkout hilserl-fix
   python -m venv ~/.lrobot && ~/.lrobot/bin/pip install -e ".[hilserl,feetech]"   # hilserl includes placo
   cd examples/hilserl_so101 && scripts/get_urdf.sh
   ```
2. **Calibrate both arms** with the usual lerobot commands (`lerobot-calibrate`), using the ids you will put in
   `local.env` (`FOLLOWER_ID`, `LEADER_ID`). Calibration is stored per machine in
   `~/.cache/huggingface/lerobot/calibration/`.
3. **Create your settings**: `cp local.env.example local.env`, then fill it in (table below).
4. **Check**: `scripts/check_setup.sh`. It resolves every config with your settings, checks the Python, the
   arm ports and the URDF, and moves nothing.
5. Follow the [Runbook](#runbook): measure your bounds and reset pose, record your own demos, train.

### What to change in `local.env`

Everything that differs between setups lives here; the scripts pass it to lerobot as command-line overrides,
so the JSON configs never need per-machine edits.

| Setting                                | What                                                                       | How to find it                                                           |
| -------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `LEROBOT_PYTHON`                       | Python of the venv with lerobot `hilserl-fix` (`[hilserl,feetech]` extras) | `which python` inside the venv                                           |
| `HF_USER`                              | Hugging Face user for the repos; empty = never push                        | `hf auth whoami`                                                         |
| `FOLLOWER_PORT`, `LEADER_PORT`         | Serial ports of the arms                                                   | `lerobot-find-port` (Linux: `/dev/ttyACM*`, macOS: `/dev/tty.usbmodem*`) |
| `FOLLOWER_ID`, `LEADER_ID`             | Calibration names                                                          | the ids used with `lerobot-calibrate`                                    |
| `WRIST_CAMERA`, `TOP_CAMERA`           | OpenCV camera index (or `/dev/videoN`)                                     | `lerobot-find-cameras opencv`; check the saved images                    |
| `CAMERA_WIDTH/HEIGHT/FPS`              | Capture mode (default 640x480 at 30)                                       | a mode your cameras support                                              |
| `EE_BOUNDS_MIN`, `EE_BOUNDS_MAX`       | Gripper-tip workspace box [x, y, z] in metres                              | `scripts/find_bounds.sh`                                                 |
| `WRIST_FLEX_RANGE`, `WRIST_ROLL_RANGE` | Optional [min, max] degrees the policy may move each wrist joint to        | hold the leader at the extremes, `scripts/read_pose.sh`                  |
| `RESET_POSE`                           | Joint pose every episode starts from (degrees); tip must be inside the box | hold the leader there, `scripts/read_pose.sh`                            |
| `HOME_POSE`                            | Folded rest pose both arms go to on exit                                   | hold the leader there, `scripts/read_pose.sh`                            |
| `LEARNER_DEVICE`, `ACTOR_DEVICE`       | `cuda`, `mps` or `cpu`                                                     | `cuda` on NVIDIA; on Apple Silicon see _Device_ below                    |
| `EC2_HOST`, `EC2_KEY`, `EC2_DIR`       | Optional remote learner: `user@address`, SSH key, remote folder            | see [Remote learner](#remote-learner-on-a-gpu-machine-aws-ec2)           |

Bounds and poses depend on where the robot, cameras and table are, so measure them on every new setup; the
values in `local.env.example` are from the original setup and are only a starting point.

### What lives in `configs/` (shared)

Change these only to change the experiment for everyone:

| Field                                                  | File(s)                      | Value                                                               |
| ------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------------- |
| `env.fps`                                              | all                          | 10                                                                  |
| `processor.reset.control_time_s`                       | test 60, record 30, train 20 | episode length (s)                                                  |
| `processor.reset.release_wait_s`                       | all                          | 3 s hold after an episode ends                                      |
| `processor.inverse_kinematics.end_effector_step_sizes` | all                          | 0.02 m per action unit                                              |
| `processor.image_preprocessing`                        | all                          | `resize_size` 128x128, no crop                                      |
| `processor.leader_intervention`                        | all                          | keys and speeds, see [Tuning](#tuning-processorleader_intervention) |
| `dataset.num_episodes_to_record`, `dataset.task`       | record                       | 15, "Pick up the cube"                                              |
| `policy.*`, `algorithm.*`, `batch_size`                | train                        | SAC settings                                                        |

Fields that `local.env` overrides hold placeholders in the JSON (`"port": "set-in-local.env"`, `null` bounds
and poses), so running lerobot on a config directly, without the scripts, fails early instead of using
another machine's values.

## How control works

Keys only switch modes; all motion comes from the leader. There are two modes, set by
`processor.leader_intervention.follow_follower`.

**Training and testing (`follow_follower: true`, one key).** The leader follows the follower while the policy
runs, so you can take over at any moment:

| Key                   | From                           | What happens                                                            |
| --------------------- | ------------------------------ | ----------------------------------------------------------------------- |
| `space`               | policy running, **hand on it** | Leader torque off, the follower copies it at once (max 90°/s per joint) |
| `space`               | correcting                     | Leader holds, rejoins the follower, policy continues from there         |
| `s` / `q`,`esc` / `r` | any                            | Episode success / failure / re-record                                   |

Hold the leader before pressing `space`: with torque off it falls if nobody holds it. At the start of each
episode the leader moves onto the follower's reset pose (at 90°/s) before following it.

**Recording (`follow_follower: false`, DAgger-style).** Each episode starts paused with the leader on the
follower; the table below is the full two-key flow (also usable in training by setting `follow_follower` to
false):

| Key                   | From                           | What happens                                                                                  |
| --------------------- | ------------------------------ | --------------------------------------------------------------------------------------------- |
| `space`               | policy running                 | Policy pauses, follower holds still, **the leader moves onto the follower** (1.5 s) and holds |
| `tab`                 | paused, **hand on the leader** | Leader torque off, the follower copies it (max 90°/s per joint)                               |
| `tab`                 | correcting                     | Leader holds where you left it, follower holds still                                          |
| `space`               | paused                         | Leader parks at home (torque off), policy resumes                                             |
| `s` / `q`,`esc` / `r` | any                            | Episode success / failure / re-record                                                         |

In this mode the leader rests at the home pose with torque off while the policy runs. Paused and correcting steps are
recorded as interventions in the policy's action space: the gripper-tip displacement of the commanded move
divided by `end_effector_step_sizes`, then the wrist flex and roll change divided by `joint_step_sizes`, each
clipped to [-1, 1], then the gripper command. The 9° wrist step equals the 90°/s take-over speed limit at
10 Hz, so wrist moves are recorded without clipping.

On exit (end or Ctrl+C) the leader and then the follower move to `reset.home_joint_positions` (the folded
rest pose) before torque is released. A hard kill (`kill -9`, closing the terminal) skips this.

On macOS, grant the terminal **Input Monitoring** (System Settings → Privacy & Security) for the keys, or keep
the terminal focused (terminal fallback).

## Runbook

0. **Check**: `scripts/check_setup.sh` (nothing moves).
1. **Bounds** (moves the follower): `scripts/find_bounds.sh` mirrors the leader on the follower for 40 s; move
   the leader over the whole task workspace. Copy the printed end-effector min/max into `local.env`
   (`EE_BOUNDS_MIN` / `EE_BOUNDS_MAX`). Then hold the leader at a start pose above the work area and run
   `scripts/read_pose.sh` (nothing moves): copy the pose into `RESET_POSE` if it says INSIDE bounds. Do the same
   for `HOME_POSE` with the arm folded at rest.
2. **Demos**: `scripts/record_demos.sh` → `outputs/datasets/so101_leader_demos_wrist` (15 episodes). Each episode
   starts paused with the leader already on the follower: hold the leader, press `tab`, pick up the cube,
   press `s`. Frames are saved only from the first `tab` on; an episode ended before any `tab` is re-recorded.
   Demos recorded before the wrist action (`outputs/datasets/so101_leader_demos`, 4 numbers per step) can be
   converted instead of re-recorded: `scripts/convert_demos_wrist.sh` (nothing moves) adds the wrist deltas from
   the recorded joint positions and writes `so101_leader_demos_wrist`. Start a fresh learner run either way.
3. _(optional)_ Reward classifier, see the docs.
4. **Train**: `scripts/train_learner.sh` in one terminal, then `scripts/run_actor.sh` in another.
5. **Push the policy**: `scripts/push_hub.sh policy` uploads the latest checkpoint to `$HF_USER/hilserl_so101_leader`.

Extra arguments to `record_demos.sh`, `test_leader.sh`, `train_learner.sh` and `run_actor.sh` are passed to
lerobot, e.g. `scripts/record_demos.sh --dataset.num_episodes_to_record=5`. Logs go to `outputs/*.log`.

## How training works

HIL-SERL ([paper](https://arxiv.org/abs/2412.09858), Luo et al. 2024) trains with two processes that talk over
gRPC on `127.0.0.1:50051`:

```
 Terminal 1: LEARNER (no robot)               Terminal 2: ACTOR (drives the robot)
 ┌───────────────────────────────┐            ┌────────────────────────────────┐
 │ offline buffer: demos +       │◄───────────│ runs the policy at 10 fps      │
 │   all interventions           │ transitions│ you intervene (space)          │
 │ online buffer: every step     │            │ s / q end each episode         │
 │ trains SAC on the Mac GPU     │───────────►│                                │
 └───────────────────────────────┘ weights    └────────────────────────────────┘
                                   (~10 s)
```

- **Learner** (`scripts/train_learner.sh`, start first): loads the demos into the offline buffer and trains SAC
  continuously. Each batch (256) is half offline, half online data (RLPD-style), so demos and interventions
  keep weight as online data grows. A separate discrete critic handles the gripper (the paper's grasp critic).
  Images go through a frozen pretrained ResNet-10 (`lerobot/resnet10`); only the small heads train.
  Policy checkpoints every 5000 updates in `outputs/train/hilserl_so101_leader/checkpoints/`; the replay buffers
  (slow to write) every 20000 updates and when the learner is stopped (`buffer_save_freq`). The log shows the
  losses, the mean Q value and the temperature every 500 updates (`[LEARNER] step ...`).
- **Resume** (`--resume=true`) restores the last checkpoint (policy, critics, optimizers, temperature) and the
  saved buffers. Saving, logging and weight-push settings come from `configs/train_hilserl_leader.json`; all
  other settings stay as in the checkpoint.
- **Actor** (`scripts/run_actor.sh`, start second, keep its terminal focused): runs the latest policy on the
  follower, streams every transition to the learner and receives new weights about every 10 s, used from the
  next step (it waits for the first ones before the first episode). Episodes last
  up to 20 s; after one ends the follower holds 3 s (`release_wait_s`) and resets.
- **Device**: `LEARNER_DEVICE` / `ACTOR_DEVICE` in `local.env`; buffers live in CPU memory. On an M1 with
  8 GB the learner is too slow (1.3 s per step at batch 64); run it on a GPU machine instead, see
  [Remote learner](#remote-learner-on-a-gpu-machine-aws-ec2) below. An NVIDIA GPU (`cuda`) is the intended learner device.

**Your role**: watch every episode, `s` when the cube is lifted (you are the reward), `q` on a clear failure.
Intervene with a hand on the leader: `space`, guide, `space`. Corrections go into both buffers. When to
intervene (the paper's advice):

- **Early on** (first ~20-30 episodes): whenever the policy clearly heads the wrong way (drifts away from the
  cube, closes on air, gets stuck). Correct just past the mistake, then give it back.
- **Keep corrections short**: a few seconds, not the whole task. The policy has to explore to learn.
- **Don't intervene** for small wobbles or slow progress while it is safe.
- **Always intervene** before it hits the table or leaves the workspace.
- **Over time**: less and less; the paper's intervention rate falls toward zero as success approaches 100%.
  Expect roughly 30-60 min of training before consistent grasps (the paper: 1-2.5 h on harder tasks).

**Stopping**: Ctrl+C the actor, then the learner (both arms go home). `scripts/push_hub.sh policy` uploads the
latest checkpoint.

|               | Paper              | This setup                                     |
| ------------- | ------------------ | ---------------------------------------------- |
| Robot         | Franka             | SO-101                                         |
| Interventions | SpaceMouse         | Leader arm following the follower, one key     |
| Reward        | Trained classifier | `s` key (classifier optional, skipped for now) |
| Demos         | 20-30              | 15                                             |
| Image crop    | Yes                | No (full scene at 128x128)                     |

## Remote learner on a GPU machine (AWS EC2)

The learner needs a GPU; the actor needs the robot. With `EC2_HOST` and `EC2_KEY` set in `local.env`, the
learner runs on a remote NVIDIA machine and the actor stays here.

```
┌──────────── EC2 (GPU) ────────────────┐          ┌────────── robot machine ─────┐
│ LEARNER (in tmux)                     │          │ ACTOR                        │
│ • demos (offline buffer)              │◄─────────│ • follower, leader, cameras  │
│ • every robot step (online buffer)    │ steps    │ • keys (space/s/q)           │
│ • trains SAC on the GPU (cuda)        │          │ • runs the policy            │
│ • saves checkpoints                   │─────────►│                              │
└───────────────────────────────────────┘ weights  └──────────────────────────────┘
                  ▲                       (~10 s)                │
                  └──────────── SSH tunnel (port 50051) ─────────┘
```

The actor still connects to `127.0.0.1:50051`; `ssh -L` forwards that port to the learner, so only SSH has to
be open on the instance (restrict it to your IP) and the configs are the same as for local training.

**Measured** (A10G `g5.2xlarge`, London, home upload): learner 0.28 s per update at batch 256 (Mac M1 GPU:
1.3 s at batch 64), 3.5-4.8 GB GPU memory. Transitions travel in a compact format (uint8 images, each
observation sent once): 10 MB per 100 steps instead of ~80 MB, about 8 Mbit/s of upload at 10 fps. Weights
come back as ~4 MB every ~10 s (the frozen ResNet-10 is not sent; it was 23 MB with it).

### Instance

| Instance      | GPU         | Approx. on-demand price |
| ------------- | ----------- | ----------------------- |
| `g4dn.xlarge` | T4, 16 GB   | ~$0.5/h                 |
| `g6.xlarge`   | L4, 24 GB   | ~$0.8/h                 |
| `g5.xlarge`   | A10G, 24 GB | ~$1/h                   |

- Ubuntu with an NVIDIA driver (e.g. the **Deep Learning AMI**), Python 3.12+, `tmux`, `rsync`; region close
  to the robot.
- New accounts often have a GPU quota of 0: _Service Quotas → EC2 → "Running On-Demand G and VT instances"_,
  needs at least 4 vCPUs.
- **Stop the instance after each session**; a stopped instance only costs its disk.

### Commands (from this folder on the robot machine)

| Step | Command                        | What                                                                                                                                                                                 |
| ---- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1    | `scripts/ec2_setup.sh`         | Copies this checkout, the URDF, the demos and `local.env` (with the remote Python, `cuda`), installs lerobot in `~/$EC2_DIR/.venv`, checks the GPU. Again after code or demo changes |
| 2    | `scripts/ec2_learner.sh start` | Starts the learner in tmux on the instance                                                                                                                                           |
| 3    | `scripts/ec2_learner.sh log`   | Follows its log (Ctrl+C stops following only); wait for `Optimization frequency loop`                                                                                                |
| 4    | `scripts/run_actor_remote.sh`  | **Moves the robot.** Opens the tunnel and starts the actor; Ctrl+C stops it, arms go home                                                                                            |
| 5    | `scripts/ec2_learner.sh stop`  | Stops the learner (`status` shows whether it runs)                                                                                                                                   |
| 6    | `scripts/ec2_learner.sh pull`  | Copies the checkpoints to `outputs/train/`, then `scripts/push_hub.sh policy` can upload one                                                                                         |

Then stop the instance in the AWS console. The remote copy lives in `~/$EC2_DIR/` (code, venv, outputs) and
leaves anything else on the machine alone.

## Hugging Face Hub

Repos go to `HF_USER` from `local.env` (log in once with `hf auth login`; the token stays in
`~/.cache/huggingface/`, never in this project). Repos are created **private**; add `--public` to make one
public. With `HF_USER` empty nothing is pushed and the dataset is named `local/so101_leader_demos_wrist`.

| Data   | Local                                                 | Hub                                 | Pushed by                    |
| ------ | ----------------------------------------------------- | ----------------------------------- | ---------------------------- |
| Demos  | `outputs/datasets/so101_leader_demos_wrist`           | `$HF_USER/so101_leader_demos_wrist` | end of `record_demos.sh`     |
| Policy | `outputs/train/hilserl_so101_leader/checkpoints/last` | `$HF_USER/hilserl_so101_leader`     | `scripts/push_hub.sh policy` |

Set `NO_PUSH=1` to skip the automatic upload, e.g. `NO_PUSH=1 scripts/record_demos.sh`. A failed upload can be
retried with `scripts/push_hub.sh demos`.

## Tuning (`processor.leader_intervention`)

| Field                 | Default                     | Meaning                                                 |
| --------------------- | --------------------------- | ------------------------------------------------------- |
| `pause_key`           | space                       | Pause / resume the policy                               |
| `correction_key`      | tab                         | Start / stop a correction                               |
| `handover_duration_s` | 1.5                         | Leader → follower move when pausing                     |
| `max_joint_speed`     | 90                          | Follower joint speed cap while copying the leader (°/s) |
| `gripper_threshold`   | 3.0                         | Gripper difference for open/close label                 |
| `start_paused`        | false (true for recording)  | Start each episode paused, leader on the follower       |
| `follow_follower`     | false (true for train/test) | Leader follows the follower; `space` alone takes over   |
