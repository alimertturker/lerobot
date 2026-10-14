# HIL-SERL on SO-101 with leader-arm interventions

Real-robot HIL-SERL ([docs](https://huggingface.co/docs/lerobot/main/en/hilserl)) on an SO-101 follower,
using the SO-101 **leader arm** for demonstrations and interventions (`control_mode: "leader"`). This folder
lives in `examples/hilserl_so101/` of the lerobot branch `hilserl-fix`, which also holds the leader-arm code;
see `AGENTS.md` for what was built and why. Run every command from this folder.

Task: pick up a cube. Two cameras (`wrist`, `top`), images resized to 128x128, no crop.

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

| Setting                          | What                                                                       | How to find it                                                           |
| -------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `LEROBOT_PYTHON`                 | Python of the venv with lerobot `hilserl-fix` (`[hilserl,feetech]` extras) | `which python` inside the venv                                           |
| `HF_USER`                        | Hugging Face user for the repos; empty = never push                        | `hf auth whoami`                                                         |
| `FOLLOWER_PORT`, `LEADER_PORT`   | Serial ports of the arms                                                   | `lerobot-find-port` (Linux: `/dev/ttyACM*`, macOS: `/dev/tty.usbmodem*`) |
| `FOLLOWER_ID`, `LEADER_ID`       | Calibration names                                                          | the ids used with `lerobot-calibrate`                                    |
| `WRIST_CAMERA`, `TOP_CAMERA`     | OpenCV camera index (or `/dev/videoN`)                                     | `lerobot-find-cameras opencv`; check the saved images                    |
| `CAMERA_WIDTH/HEIGHT/FPS`        | Capture mode (default 640x480 at 30)                                       | a mode your cameras support                                              |
| `EE_BOUNDS_MIN`, `EE_BOUNDS_MAX` | Gripper-tip workspace box [x, y, z] in metres                              | `scripts/find_bounds.sh`                                                 |
| `RESET_POSE`                     | Joint pose every episode starts from (degrees); tip must be inside the box | hold the leader there, `scripts/read_pose.sh`                            |
| `HOME_POSE`                      | Folded rest pose both arms go to on exit                                   | hold the leader there, `scripts/read_pose.sh`                            |
| `LEARNER_DEVICE`, `ACTOR_DEVICE` | `cuda`, `mps` or `cpu`                                                     | `cuda` on NVIDIA; on Apple Silicon see _Device_ below                    |

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

Like the DAgger rollout strategy: keys only switch modes, all motion comes from the leader.

| Key                   | From                           | What happens                                                                                  |
| --------------------- | ------------------------------ | --------------------------------------------------------------------------------------------- |
| `space`               | policy running                 | Policy pauses, follower holds still, **the leader moves onto the follower** (1.5 s) and holds |
| `tab`                 | paused, **hand on the leader** | Leader torque off, the follower copies it (max 90°/s per joint)                               |
| `tab`                 | correcting                     | Leader holds where you left it, follower holds still                                          |
| `space`               | paused                         | Leader parks at home (torque off), policy resumes                                             |
| `s` / `q`,`esc` / `r` | any                            | Episode success / failure / re-record                                                         |

While the policy runs the leader rests at the home pose with torque off. Paused and correcting steps are
recorded as interventions: `[delta_x, delta_y, delta_z, gripper]`, the end-effector displacement of the
commanded move divided by `end_effector_step_sizes`, clipped to [-1, 1]. Wrist rotation is not part of the
action.

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
2. **Demos**: `scripts/record_demos.sh` → `outputs/datasets/so101_leader_demos` (15 episodes). Each episode
   starts paused with the leader already on the follower: hold the leader, press `tab`, pick up the cube,
   press `s`. Frames are saved only from the first `tab` on; an episode ended before any `tab` is re-recorded.
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
 │   all interventions           │ transitions│ you intervene (space / tab)    │
 │ online buffer: every step     │            │ s / q end each episode         │
 │ trains SAC on the Mac GPU     │───────────►│                                │
 └───────────────────────────────┘ weights    └────────────────────────────────┘
                                   (~50 s)
```

- **Learner** (`scripts/train_learner.sh`, start first): loads the demos into the offline buffer and trains SAC
  continuously. Each batch (256) is half offline, half online data (RLPD-style), so demos and interventions
  keep weight as online data grows. A separate discrete critic handles the gripper (the paper's grasp critic).
  Images go through a frozen pretrained ResNet-10 (`lerobot/resnet10`); only the small heads train.
  Checkpoints every 5000 updates in `outputs/train/hilserl_so101_leader/checkpoints/`.
- **Actor** (`scripts/run_actor.sh`, start second, keep its terminal focused): runs the latest policy on the
  follower, streams every transition to the learner and receives new weights about every 50 s. Episodes last
  up to 20 s; after one ends the follower holds 3 s (`release_wait_s`) and resets.
- **Device**: `LEARNER_DEVICE` / `ACTOR_DEVICE` in `local.env`; buffers live in CPU memory. On an M1 with
  8 GB the learner is too slow (1.3 s per step at batch 64) and the actor should run on the CPU; see _Remote
  learner on AWS EC2_ below. An NVIDIA GPU (`cuda`) is the intended learner device.

**Your role**: watch every episode, `s` when the cube is lifted (you are the reward), `q` on a clear failure.
Intervene with `space` → `tab` → guide → `tab` → `space` when the policy heads somewhere wrong or is stuck.
Corrections go into both buffers. Intervene often at the start, keep corrections short (nudge, then hand
back), and less over time; the paper's intervention rate falls toward zero as success approaches 100%.
Expect roughly 30-60 min of training before consistent grasps (the paper: 1-2.5 h on harder tasks).

**Stopping**: Ctrl+C the actor, then the learner (both arms go home). `scripts/push_hub.sh policy` uploads the
latest checkpoint.

|               | Paper              | This setup                                     |
| ------------- | ------------------ | ---------------------------------------------- |
| Robot         | Franka             | SO-101                                         |
| Interventions | SpaceMouse         | Leader arm, key-switched                       |
| Reward        | Trained classifier | `s` key (classifier optional, skipped for now) |
| Demos         | 20-30              | 15                                             |
| Image crop    | Yes                | No (full scene at 128x128)                     |

<details>
<summary><b>Remote learner on AWS EC2 (planned, the M1 is too slow to train)</b></summary>

### Why

Measured on this Mac (M1, 8 GB) with the real demos, batch 64, no robot:

| Learner on    | First step | Then                       | Peak memory |
| ------------- | ---------- | -------------------------- | ----------- |
| CPU           | 9.0 s      | 3.4 s / step               | 1.4 GB      |
| Mac GPU (MPS) | 7.3 s      | 1.3 s / step (0.8 steps/s) | 1.1 GB      |

The robot produces 10 steps/s, so an hour at the robot gives only ~3,000 training steps. Most of the time goes
into the frozen ResNet-10 image encoder. The first full run (batch 256, actor also on MPS) froze the Mac: the
learner never finished a step in 8 min and the actor dropped to 3-6 fps. The Mac is fine for the robot side,
not for the learner.

### Layout

```
┌──────────── AWS EC2 (GPU) ────────────┐          ┌────────── Mac ───────────────┐
│ LEARNER                               │          │ ACTOR                        │
│ • demos (offline buffer)              │◄─────────│ • follower, leader, cameras  │
│ • every robot step (online buffer)    │ steps    │ • keys (space/tab/s/q)       │
│ • trains SAC on the GPU (cuda)        │          │ • policy on the CPU (~11 ms) │
│ • saves checkpoints                   │─────────►│                              │
└───────────────────────────────────────┘ weights  └──────────────────────────────┘
                  ▲                       (~50 s)                │
                  └──────────── SSH tunnel (port 50051) ─────────┘
```

- EC2 runs only the learner: no robot, cameras or USB, just the GPU, the demos and the code.
- The Mac runs only the actor; everything at the robot stays the same.
- `ssh -N -L 50051:localhost:50051 ubuntu@<ec2-ip>` forwards the port, so the actor keeps
  `learner_host: 127.0.0.1`. Only SSH is open on the instance (restrict it to your IP).
- The learner runs in `tmux` on EC2, so it survives a dropped SSH session.

### Instance

| Instance      | GPU         | Approx. on-demand price |
| ------------- | ----------- | ----------------------- |
| `g4dn.xlarge` | T4, 16 GB   | ~$0.5/h                 |
| `g6.xlarge`   | L4, 24 GB   | ~$0.8/h (preferred)     |
| `g5.xlarge`   | A10G, 24 GB | ~$1/h                   |

- AMI: **Deep Learning AMI (Ubuntu)** (CUDA and PyTorch preinstalled), region closest to the robot.
- New accounts often have a GPU quota of 0: _Service Quotas → EC2 → "Running On-Demand G and VT instances"_,
  needs at least 4 vCPUs.
- Stop the instance after each session; a stopped instance only costs its disk.

### Before it works (to do)

1. **Smaller transfer format.** Each step is currently sent as ~0.8 MB (both images as float32, for state and
   next state): ~157 MB per episode, ~60 Mbit/s of sustained upload at 10 fps. Plan: send images as uint8 and
   drop the duplicated next state (~8x smaller, ~20 MB per episode); the learner converts back.
2. `scripts/ec2_setup.sh`: rsync the lerobot branch and this project to the instance, install, `hf auth login`,
   download `$HF_USER/so101_leader_demos`.
3. `scripts/ec2_learner.sh`: start the learner in tmux on EC2 (`device: cuda`, batch 256).
4. `scripts/run_actor_remote.sh`: open the SSH tunnel and start the actor (`--policy.device=cpu`).
5. Connection test without the robot.

Needs: the instance's IP and the path to its `.pem` key.

### A session

1. Start the instance in the AWS console.
2. Mac: `scripts/ec2_learner.sh` (learner starts on EC2).
3. Mac: `scripts/run_actor_remote.sh` (tunnel + actor; the robot starts moving).
4. Train as usual: watch, intervene, `s` / `q`.
5. Ctrl+C the actor (arms go home), stop the learner, **stop the instance**.
6. Checkpoints stay on EC2; `push_hub.sh policy` can upload one from there.

</details>

## Hugging Face Hub

Repos go to `HF_USER` from `local.env` (log in once with `hf auth login`; the token stays in
`~/.cache/huggingface/`, never in this project). Repos are created **private**; add `--public` to make one
public. With `HF_USER` empty nothing is pushed and the dataset is named `local/so101_leader_demos`.

| Data   | Local                                                 | Hub                             | Pushed by                    |
| ------ | ----------------------------------------------------- | ------------------------------- | ---------------------------- |
| Demos  | `outputs/datasets/so101_leader_demos`                 | `$HF_USER/so101_leader_demos`   | end of `record_demos.sh`     |
| Policy | `outputs/train/hilserl_so101_leader/checkpoints/last` | `$HF_USER/hilserl_so101_leader` | `scripts/push_hub.sh policy` |

Set `NO_PUSH=1` to skip the automatic upload, e.g. `NO_PUSH=1 scripts/record_demos.sh`. A failed upload can be
retried with `scripts/push_hub.sh demos`.

## Tuning (`processor.leader_intervention`)

| Field                 | Default                    | Meaning                                                 |
| --------------------- | -------------------------- | ------------------------------------------------------- |
| `pause_key`           | space                      | Pause / resume the policy                               |
| `correction_key`      | tab                        | Start / stop a correction                               |
| `handover_duration_s` | 1.5                        | Leader → follower move when pausing                     |
| `max_joint_speed`     | 90                         | Follower joint speed cap while copying the leader (°/s) |
| `gripper_threshold`   | 3.0                        | Gripper difference for open/close label                 |
| `start_paused`        | false (true for recording) | Start each episode paused, leader on the follower       |
