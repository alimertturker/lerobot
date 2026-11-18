# EXPO-FT and Real-Time EXPO-FT for SO arms

Online reinforcement-learning fine-tuning for an existing LeRobot VLA, using one SO-101 or SO-100
follower, one leader, and task cameras on a single machine. Collect demonstrations and train the base
VLA with the standard LeRobot tools first. This example supplies online interaction, human corrections,
manual rewards, replay, learning, checkpoints, and recorded evaluation.

Two modes share the same configuration and launcher:

| Mode | Policies | Execution |
| --- | --- | --- |
| Ordinary EXPO-FT (`real_time=false`) | SmolVLA or compatible PI0.5 | Generate candidates when the current chunk ends; hold the last target while planning. |
| Real-Time EXPO-FT (`real_time=true`) | SmolVLA | Generate prefix-conditioned candidates while the current chunk executes; edit and select using a fresh observation at the boundary. |

Both modes compare original VLA chunks and bounded residual edits using learned critics. These are
PyTorch adaptations of [EXPO-FT](https://arxiv.org/abs/2605.25477) and
[Real-Time EXPO-FT](https://arxiv.org/abs/2609.18207), with the
[authors' implementation](https://github.com/pd-perry/expo-ft) as a reference. The SO-arm action space,
SmolVLA backbone, and single-machine execution differ from the reference experiments. The settings below
are reproducible starting points, not a guarantee of improvement on another task.

## Installation and prerequisites

Run all commands from the **LeRobot repository root**, which contains `pyproject.toml` and `examples/`.
Install the dataset, hardware, and policy dependencies in your environment. For SmolVLA:

```bash
uv sync --extra dataset --extra hardware --extra smolvla --extra feetech
```

For ordinary EXPO-FT with PI0.5, install the `pi` policy extra instead. The VLA, critics, optimizers, and
sampled batches must fit in GPU memory; replay also needs substantial system RAM and disk space.

Prepare:

- A task-trained VLA checkpoint with its saved policy processors and normalization statistics.
- A matching LeRobot dataset of successful joint-position demonstrations.
- Calibrated follower and leader arms, with working cameras and consistent camera identities.
- The task description, dataset FPS, and joint units used for the base VLA.

This example uses five arm joint targets plus the gripper; it does not use inverse kinematics or Cartesian
EE-delta actions. With `use_degrees=true`, arm targets are degrees and the gripper uses its calibrated
0–100 range. Robot, leader, and `dataset_use_degrees` must agree. State/action joint order comes from the
dataset. Single-observation checkpoints without temporal memory are required. Disable the policy's
separate built-in RTC processor; `real_time` selects this example's Real-Time EXPO-FT implementation.

## Configure your task

Copy the template and edit your own file:

```bash
cp examples/expo_ft_so101/configs/train.json /tmp/my_expo_ft.json
```

The launcher accepts `--config-path /tmp/my_expo_ft.json` and forwards `--field=value` overrides to the
training entry point. `--config_path` is also accepted. The config file path is resolved from the caller's
working directory; paths **inside the config and overrides** are resolved from the repository root.

Replace these template values:

| Field | What to supply |
| --- | --- |
| `policy_path` | Local `pretrained_model` directory or Hub model ID for your task-trained VLA. Template: `outputs/train/base_smolvla/checkpoints/last/pretrained_model`. |
| `dataset_repo_id`, `dataset_root` | Dataset ID and local directory. Set `dataset_root=null` to use normal Hub/cache resolution. `local/...` is a local placeholder, not an upload destination. |
| `task`, `fps`, `dataset_use_degrees` | Match the demonstrations and checkpoint. The template uses 10 Hz. |
| `robot`, `teleop` | Actual serial ports, calibration IDs, and optional `calibration_dir`. Template ports/IDs are examples. |
| `robot.cameras` | Camera names, devices, capture resolution and FPS. Verify wrist/top identity; device indices can change after reconnecting. |
| `rename_map` | Dataset-to-policy image key mapping, if needed. An empty map reuses the checkpoint's saved mapping. |
| `output_dir` | A fresh run directory; never reuse another experiment's checkpoints for a fresh run. |
| `demo_episodes` | Up to 50 successful demonstration episodes by default. Loaded demonstrations are assigned success rewards. |

The template sets `use_startup_poses=true`, with `reset_pose` and `home_pose` set to `null`. After connecting,
it captures the follower's starting pose for reset/home and the leader's starting pose for parking. Place
both arms in suitable starting poses before launching. Alternatively, set `use_startup_poses=false` and
provide verified six-value reset/home poses in dataset action order. Check the captured poses in the log.

`joint_bounds={}` disables additional configured absolute bounds. To constrain a joint, add an entry such
as `"shoulder_lift.pos": [minimum, maximum]`, using verified limits in your joint units. Reset/home targets
must satisfy those bounds. The speed limit and follower `max_relative_target` remain active. A measured
joint-bound violation discards the episode for rerecording. Keep an operator and physical stop available.

## Starting parameters

These are the values in `configs/train.json`; dataclass defaults may differ.

| Parameters | Template values | Meaning |
| --- | --- | --- |
| `real_time` | `false` | Ordinary mode by default. |
| `chunk_steps` | `20` | Actions executed per selected chunk; 2 seconds at 10 Hz. Must not exceed the checkpoint's `chunk_size`. |
| `num_candidates`, `num_edits` | `4`, `4` | Four independent VLA proposals and four residual edits; select among eight chunks. Edits cannot exceed the number of base candidates. |
| `candidate_batch_size` | `4` | VLA proposals processed together; reduce to lower inference VRAM. |
| `edit_scale` | `0.03` | Maximum per-coordinate residual magnitude in the checkpoint's normalized action space, not degrees. Does not constrain changes between consecutive residuals. |
| `critic_encoder`, `critic_image_size` | `resnet`, `224` | Reference-style critic encoder, initialized from scratch. |
| `num_critics`, `num_min_critics` | `10`, `2` | Critic ensemble size and randomly selected subset used for conservative candidate scoring/backups. |
| `hidden_dim` | `256` | Hidden width for learner networks. |
| `batch_size`, `vla_batch_size` | `4`, `8` | Replay samples per critic update and successful chunks per VLA update. |
| `critic_lr`, `edit_lr`, `vla_lr` | `3e-4`, `3e-4`, `2.5e-6` | Learning rates for fresh optimizer states. |
| `train_vla` | `true` | Update the VLA using its supervised flow objective on successful chunks. `false` freezes VLA weights while RL components continue learning. |
| `temperature`, `adjust_target_entropy` | `0.01`, `true` | Initial learned entropy coefficient; account for the residual scale when tuning entropy. Temperature is not a direct joint-noise amplitude. |
| `discount`, `tau` | `0.99`, `0.005` | Discount per recorded action step and critic target-network update rate. |
| `grad_clip_norm` | `10` | Training gradient norm limit. |
| `warmup_updates` | `20` | Outer updates using demonstrations before the first robot connection. Skipped when resuming. |
| `updates_per_episode`, `critic_updates_per_update` | `10`, `5` | Ten outer updates between episodes; each has five critic updates, one edit/temperature update, and one VLA update when enabled and eligible data exists. Real-time adds noise-filter learning. |
| `episode_time_s`, `num_episodes` | `30`, `100` | Episode timeout and total training episode target. |
| `reset_time_s`, `release_wait_s` | `3`, `10` | Nominal reset motion duration and wait after each rollout; learning/encoding can add more time between episodes. |
| `replay_capacity`, `demo_episodes` | `10000`, `50` | Replay capacity in action frames and requested demonstration count. Demonstrations are protected from online eviction. |
| `storage_image_size` | `0` | Preserve native camera pixels for VLA inference/replay. Critics resize their own inputs. |
| `save_every`, `checkpoint_keep_last` | `5`, `2` | Save every five completed episodes and retain the latest two managed checkpoints. |
| `shared_replay`, `cpu_threads`, `seed` | `true`, `1`, `1000` | Share immutable episode files across checkpoints; limit PyTorch CPU threads; initialize RNGs for a fresh run. |
| `eval_record` | `true` | Record datasets only when `eval_only=true`; training still uses replay. |

The `resnet` encoder ports the released preactivation `(3,4,6,3)` architecture with group normalization,
channel-stacked cameras, 512 image features and 64 state features. Although called ResNet-50 in the paper,
the released blocks differ from torchvision's bottleneck ResNet-50. SmolVLA keeps its own vision encoder;
critic resizing does not downscale the VLA's stored inputs. `critic_encoder=compact` with
`critic_image_size=96` remains available for a fresh run.

For `D = chunk_steps * action_dim`, entropy adjustment uses `-D/2 + D*log(edit_scale)` in the scaled
residual coordinates. Changing chunk length changes this target and the critic/edit architecture.

## Ordinary EXPO-FT

If the template batches exceed available VRAM, add `--batch_size=1 --vla_batch_size=1
--candidate_batch_size=1` to a training command (on one line). These settings trade throughput for memory;
they do not reduce the native-resolution CPU replay footprint. Validate with `check_only` first.

For a checkpoint with a horizon of at least 20 actions:

```bash
bash examples/expo_ft_so101/scripts/run_online.sh \
  --config-path /tmp/my_expo_ft.json \
  --resume=null --real_time=false --chunk_steps=20 \
  --output_dir=outputs/expo_ft_so101
```

The VLA generates full chunks through its native API; EXPO-FT executes only `chunk_steps` from the selected
chunk. `n_action_steps` is the ordinary policy queue setting and does not determine execution length here.
The checkpoint's `chunk_size` determines the generated horizon; its `num_steps` controls flow denoising
iterations. This example does not change the offline model horizon.

Online VLA updates preserve the checkpoint's trainable/frozen layers. For SmolVLA, inspect the saved
`train_expert_only` and `freeze_vision_encoder` settings; `train_vla=true` does not automatically unfreeze
the backbone. `train_vla=false` disables all VLA optimizer steps.

The leader follows the follower during autonomous execution. Space switches to/from leader corrections;
pending policy output is invalidated on handover. While a new ordinary-mode plan is pending, the follower
holds its last target and the loop continues servicing the leader and keyboard.

| Key | Action |
| --- | --- |
| Space | Toggle policy control and human correction. |
| `s` | End successfully: terminal reward 1. |
| `q` / Esc | End unsuccessfully: terminal reward 0. |
| `r` | Discard the episode and repeat it. |
| Ctrl+C | Stop; training saves completed replay/learner state, evaluation finalizes recorded data. |

Keep the controlling terminal focused. A timeout without `s` counts as failure. Human actions and
intervention flags are stored with the actual joint targets returned by `robot.send_action`, including
clipping. Leader following can be disabled with `--leader_intervention.follow_follower=false`.

## Real-Time EXPO-FT

Start a **fresh RL run** for this mode. For a 20-action SmolVLA horizon, use:

```bash
bash examples/expo_ft_so101/scripts/run_online.sh \
  --config-path /tmp/my_expo_ft.json \
  --resume=null --real_time=true --chunk_steps=10 \
  --real_time_delay_steps=4 --real_time_filter_candidates=8 \
  --num_episodes=30 --output_dir=outputs/expo_ft_so101_realtime
```

Let `H` be the checkpoint horizon, `C=chunk_steps`, and `d=real_time_delay_steps`. Require
`0 < d <= C` and `C+d <= H`. Here `H=20`, `C=10`, `d=4`: execute one second of actions at 10 Hz and start
prefetching when four actions remain, providing approximately 0.4 seconds for generation.

The slow VLA generates a full proposal with the remaining committed actions hard-inpainted as a clean
prefix. At the boundary, fast editing and critic selection use a fresh observation and the proposal window
`[d:d+C]`. The initial request uses no prefix. One GPU worker serializes policy/processor operations while
motor execution overlaps VLA generation. Human handover cancels/drains prefetched output before policy
control resumes. If generation misses its deadline, the arm holds its last target and logs a warning.
This is latency-aware execution, not a hard real-time guarantee; safety clipping can also change the
committed targets.

Training aligns the backup VLA observation and executed prefix with the configured delay, while the critic
and edit policy see the next/current state. A two-head noise-Q filter scores eight Gaussian seeds using
current and delayed observation features, denoises one seed, then compares the unedited proposal with
`num_edits` edits for the backup. The filter regresses the unedited target-Q value once per outer update.
VLA updates use clean prefix tokens and exclude them from the flow loss. The episode's initial execution
window uses no prefix. The same successful demonstrations and successful online episodes supply VLA data.

This port currently supports SmolVLA real-time mode only. The reference uses a different backbone and
prefix-trained SFT initialization. An ordinary SFT checkpoint can initialize this implementation, but
prefix-conditioned behavior must be evaluated on your task. This port uses current VLA weights rather than
the reference backbone's optional EMA. It does not establish numerical or performance equivalence to the
paper. Original experimental real-time filter checkpoints without a format version are incompatible with
the current filter; their VLA `pretrained_model` may initialize a fresh run.

## Resume training

Resume the **whole checkpoint**, with the same mode, execution window, delay, encoder, and replay image
resolution. For example, continue a real-time run from episode 30 to episode 60:

```bash
bash examples/expo_ft_so101/scripts/run_online.sh \
  --config-path /tmp/my_expo_ft.json \
  --resume=outputs/expo_ft_so101_realtime/checkpoints/episode_000030_update_00000320 \
  --eval_only=false --real_time=true --chunk_steps=10 --real_time_delay_steps=4 \
  --train_vla=true --num_episodes=60 --output_dir=outputs/expo_ft_so101_realtime
```

Use the actual checkpoint directory shown in your log. Ordinary runs use `real_time=false` and their
saved `chunk_steps`. `num_episodes` is the **total target**, including saved training episodes. VLA weights,
critic targets, edit policy, temperature, optimizer state, replay sampling state, and RNGs are restored;
real-time also restores the filter and its optimizer. Saved optimizer learning rates are restored too.
`train_vla=false` disables VLA optimizer steps even when resuming.

To switch modes or change an architectural setting, use `resume=null`, a fresh output directory, and
`policy_path` pointing to the desired base VLA or a checkpoint's `pretrained_model`. This initializes new RL
networks and replay; it does not continue the old RL state.

## Evaluate and record a dataset

Evaluate a real-time checkpoint with fixed weights and 30 trials:

```bash
bash examples/expo_ft_so101/scripts/run_online.sh \
  --config-path /tmp/my_expo_ft.json \
  --resume=outputs/expo_ft_so101_realtime/checkpoints/episode_000030_update_00000320 \
  --eval_only=true --eval_record=true \
  --real_time=true --chunk_steps=10 --real_time_delay_steps=4 \
  --num_episodes=30 --output_dir=outputs/expo_ft_so101_realtime_eval
```

For ordinary mode, use its checkpoint, `real_time=false`, its saved chunk length, and a separate output
folder. `num_episodes` counts new evaluation trials from zero. No learner updates, training replay inserts,
or checkpoint saves occur. Candidate generation, residual sampling and critic-subset sampling remain
stochastic, as during training. This is fixed-weight evaluation, not deterministic or exploration-free
inference. Use `s`/`q` to label outcomes; avoid human corrections when measuring autonomous success.

With recording enabled, completed accepted episodes become a LeRobot dataset under
`<output_dir>/evaluation_dataset`; set `eval_dataset_root` to override. All policy cameras become H.264
videos, alongside measured joint states, executed actions, `next.reward`, `next.done`, `next.success`, and
`complementary_info.is_intervention`. Success is true only on the final frame of successful episodes.
`evaluation_context.json` identifies the checkpoint/task/units, and `evaluation_episodes.jsonl` stores
per-episode success, intervention counts and timing summaries.

Frames correspond to executed actions at nominal dataset FPS. Reset/home movements, inference-only hold
ticks, and the extra terminal replay observation without an action are excluded. Video timestamps are
not wall-clock timestamps. Encoding runs between episodes, adding time between trials. Saved episodes
are finalized on normal exit and Ctrl+C; discarded episodes and episodes interrupted during execution
are excluded. Use a fresh dataset destination for each recording session: an existing directory is
rejected before connecting the arms. Already running processes do not reload edited configs.

Set `eval_record=false` to collect metrics only. Evaluation currently restores training replay as part
of checkpoint loading, so retain that replay even if evaluation itself does not update it.

### Upload evaluation data to Hugging Face

Authenticate with an account that can write the intended dataset namespace. To upload automatically after
normal completion, add these options to the evaluation command:

```bash
--eval_dataset_repo_id=YOUR_HF_USERNAME/so101_expo_ft_evaluation \
--eval_dataset_push_to_hub=true
```

Upload is off by default, and the `local/...` placeholder is rejected for upload. Datasets default to
private; use `eval_dataset_private=false` to publish publicly. Upload failure leaves the local dataset.
A finalized local dataset can also be uploaded later:

```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    "YOUR_HF_USERNAME/so101_expo_ft_evaluation",
    root="outputs/expo_ft_so101_realtime_eval/evaluation_dataset",
)
dataset.push_to_hub(private=True)
```

## Monitor learning and preserve useful checkpoints

`train.log` and `run_config_<timestamp>.json` capture metrics and the resolved config, including overrides.
Track success rate over fixed episode windows, intervention counts, successful completion time, planning
wait, deadline warnings and clipping. Compare checkpoints using the same task/object-placement protocol.
Training success with human assistance is a different measurement from autonomous evaluation success.

Losses include `critic_loss`, `edit_loss`, `temperature`, `vla_loss` when successful-chunk VLA updates are
enabled, and `filter_loss` in real-time mode. Lower critic loss measures fit to bootstrapped targets; it
does not prove improved task success. Supervised updates use all eligible chunks in successful episodes,
which can include retries and poor motions. More online training is not guaranteed to improve the VLA.
Use separate fixed-weight evaluations rather than selecting a model solely by its latest training loss.

The default retention keeps only **two latest** checkpoints, not the two best-performing ones. Before
continuing past a checkpoint you want to compare, back up the entire run (checkpoint plus referenced
replay), or set `checkpoint_keep_last=0` to retain all checkpoints if disk space permits. Copying only a
checkpoint folder elsewhere without its shared replay is insufficient for resuming or EXPO-FT evaluation.
A fresh save needs temporary free disk space before older files are pruned.

## Storage, memory and diagnostics

```text
outputs/expo_ft_so101_realtime/
  train.log
  run_config_<timestamp>.json
  replay/                         # Immutable episode .pt files shared by checkpoints
  checkpoints/<checkpoint-name>/
    pretrained_model/             # VLA weights/config/processors
    training_state.pt             # RL networks, optimizers, RNGs and replay reference
    realtime_state.pt             # Filter/optimizer, real-time mode only
    replay_manifest.json          # Replay files required by this snapshot
    expo_ft_config.json
```

Training replay includes both camera pixels, joint states, actions, terminal rewards and intervention
flags. It is an internal learning buffer, not a complete archival LeRobot dataset. With 50 demonstrations
protected in a 10,000-frame buffer, only the remaining capacity is available for recent online episodes.
The demonstrations must fit with enough space left for online data. Whole online episodes are evicted.

Two 640x480 RGB cameras and 10,000 frames require approximately **18.4 GB (17.2 GiB) of RAM for pixels
alone**, plus terminal observations and other memory. Shared replay has a similar uncompressed disk cost.
`storage_image_size=0` preserves native images. A positive value downsizes images **before both VLA
inference and replay storage**, trading visual detail for RAM/disk; it does not affect only the critic.
Changing stored resolution requires fresh replay. Checkpoint retention also cleans up shared replay files
that no remaining checkpoint references. Keep all files named in the manifests of checkpoints you retain.

`pretrained_model` can be used separately for ordinary VLA inference, but does not contain the critics,
residual policy or real-time filter. Evaluation datasets use compressed video and are separate from replay.

Diagnostics are off by default:

| Flag | Output |
| --- | --- |
| `capture_initial_plan` | First ordinary-mode planning input/output, images, RNG and critic diagnostics per episode under `initial_plans/`. |
| `capture_all_plans` | All ordinary captures under `policy_plans/`, or full real-time proposal/selection captures under `realtime_plans/`. |
| `capture_control_trace` | Per-tick commands, phases and planning events under `control_traces/`. |

These captures are not needed for training or resume and can be deleted independently of replay. In
real-time mode, use `capture_all_plans` for planning captures; `capture_initial_plan` alone does not capture
the real-time planner. Ordinary-mode `debug_frozen_training` and `debug_plan_seed` are available for
matched training/evaluation investigations; matched-seed debugging is rejected in real-time mode.

For a hardware-free model sampling/update check, add `--check_only=true --resume=null` and keep the
mode/chunk/delay settings you want to validate. It runs one sampled plan and learner update without
connecting the arms or saving checkpoints. It still requires the base model, dataset, RAM and GPU memory.

The launcher uses `uv run python` by default. For an already installed environment or Docker container,
set `LEROBOT_PYTHON` to its Python executable, for example:

```bash
LEROBOT_PYTHON=/usr/bin/python3 bash examples/expo_ft_so101/scripts/run_online.sh \
  --config-path /tmp/my_expo_ft.json --check_only=true
```
