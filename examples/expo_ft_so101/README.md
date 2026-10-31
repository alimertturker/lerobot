# EXPO-FT for SO-101

This example adds online reinforcement-learning fine-tuning to an existing LeRobot VLA checkpoint
for an SO-101 leader/follower setup. It supports compatible PI0.5 and SmolVLA checkpoints. The
demonstration collection and initial supervised fine-tuning stages use the regular LeRobot tools;
this example covers online interaction, human correction, reward assignment, and learning.

The implementation is inspired by [EXPO-FT](https://arxiv.org/abs/2605.25477). It samples multiple
action chunks from the VLA, generates bounded residual edits, scores the candidates with learned
critics, and executes the highest-scoring candidate. The VLA is fine-tuned with its original
supervised objective while the critics and edit policy learn from online rewards and replay.

## What this example controls

The action space is the SO-101's six joint targets: the five arm joints and the gripper. It does not
use Cartesian end-effector deltas or inverse kinematics. The action order is taken from the dataset
and must match the VLA checkpoint's action features.

When `use_degrees` is enabled, arm joints use degrees and the gripper uses its calibrated 0–100
range. Otherwise, the arm joints use LeRobot's normalized range. Set `dataset_use_degrees` to match
the units used to record the VLA's training dataset. The robot, teleoperator, and dataset unit
settings must agree.

EXPO-FT residual edits are applied in the checkpoint's normalized action representation. The saved
policy processors convert the resulting actions to robot joint targets. Checkpoints using relative
joint actions are supported; offsets are anchored to the observation at the beginning of each
executed chunk. Cartesian and EE-delta demonstration datasets are not supported by this example.

## Requirements

- A compatible, already-trained PI0.5 or SmolVLA checkpoint.
- A LeRobot dataset of SO-101 joint-action demonstrations for the same task and policy.
- One SO-101 follower arm, one SO-101 leader arm, and the cameras expected by the checkpoint.
- A machine with a supported PyTorch device. The VLA, critic, and optimizer must fit in available
  memory; reduce batch sizes or image resolution if needed.
- Calibrated arms and a safe, prepared workspace.

Install LeRobot and the optional dependencies needed by your model, robot, cameras, and dataset.
For example, the extras for SmolVLA and Feetech are:

```bash
uv sync --extra dataset --extra smolvla --extra feetech
```

For PI0.5, install the corresponding policy extra instead of `smolvla`. Add any other extras your
hardware or checkpoint requires. Run the commands below from the LeRobot repository root.

## Prepare the configuration

Edit [`configs/train.json`](configs/train.json) for your setup. Important fields include:

- `policy_path`: local `pretrained_model` directory or a Hugging Face model ID for the initial VLA
  checkpoint.
- `dataset_repo_id` and `dataset_root`: the demonstration dataset identifier and local data path.
- `demo_episodes`: number of demonstration episodes to load. The current implementation treats
  the requested demonstrations as successful because standard recording datasets do not include
  EXPO-FT reward labels. Use successful demonstrations for this bootstrap data.
- `task`, `fps`, and `dataset_use_degrees`: match the base dataset and checkpoint.
- `robot`, `teleop`, and their camera settings: set the follower/leader ports, IDs, calibration,
  camera names, and camera capture properties for your machine. Camera names must map to the image
  inputs expected by the checkpoint. A checkpoint's saved camera rename map is reused automatically;
  `rename_map` can override it.
- `reset_pose` and `home_pose`: safe positions in the configured joint units and dataset action
  order. Verify these values on your robot before running. `use_startup_poses` can capture startup
  positions instead, if enabled in the configuration.
- `joint_bounds`: optional per-joint `[minimum, maximum]` safety limits. Empty means no additional
  configured position bounds; the normal robot target and speed limits still apply.
- `output_dir`: a new directory for logs, replay data, and checkpoints. Use a different directory
  for a fresh run; use `resume` to continue an existing run.

Other useful training settings include `chunk_steps` (actions executed per plan), `num_candidates`,
`num_edits`, `edit_scale`, `batch_size`, `vla_batch_size`, `candidate_batch_size`,
`warmup_updates`, and `updates_per_episode`. `chunk_steps` cannot exceed the VLA's `chunk_size`.
Changing it changes the critic and edit-policy architecture and is not compatible with resuming a
checkpoint created with another value. Smaller batches use less accelerator memory but may change
training behavior.

## Start online training

Review the configuration and confirm both arms, cameras, reset poses, and workspace are ready. The
command below connects to and moves the physical robot:

```bash
bash examples/expo_ft_so101/scripts/run_online.sh
```

The launcher uses `uv run` by default. To use an already activated environment, set
`LEROBOT_PYTHON` to its Python executable. Configuration values can be overridden on the command
line, for example:

```bash
bash examples/expo_ft_so101/scripts/run_online.sh \
  --batch_size=8 \
  --vla_batch_size=1 \
  --candidate_batch_size=1
```

During a rollout, the leader follows the follower while the policy controls the arm. The keyboard
keys are:

| Key | Action |
| --- | --- |
| Space | Toggle between policy control and leader correction |
| `s` | Mark the episode successful (reward 1) |
| `q` or Esc | Mark the episode unsuccessful (reward 0) |
| `r` | Discard the current episode so it can be repeated |
| Ctrl+C | Stop training cleanly and save completed experience |

Leader corrections override policy actions and are recorded in replay. The follower holds its
current target while a new plan is being computed. A takeover invalidates pending policy output; a
fresh plan is generated after returning control to the policy. Episodes that reach `episode_time_s`
without a success label end as failures.

## How learning proceeds

1. The replay buffer is initialized with the selected successful demonstration episodes.
2. EXPO-FT performs `warmup_updates` before connecting to the arms.
3. At each planning boundary, the VLA samples `num_candidates` action chunks from the current
   observation. `candidate_batch_size` limits how many are inferred in one forward pass.
4. The edit policy proposes bounded residuals for `num_edits` candidates. Critics score the
   original and edited candidates, and the highest-scoring chunk is executed for `chunk_steps`.
5. The operator can correct actions at any time. At the end of the episode, the operator assigns a
   binary success/failure reward. The episode and correction labels are added to replay.
6. The critics and edit policy are updated from replay; successful demonstrations and online
   trajectories are also used for VLA supervised fine-tuning. Failed episodes still provide data
   for critic and edit-policy updates.

This is a practical PyTorch adaptation, not a numerical reproduction of the authors' JAX/DROID
experiments. It uses a compact, separately trained image encoder for the critic. The current example
supports single-observation PI0.5/SmolVLA checkpoints; it does not add temporal memory or RTC.

## Monitoring and evaluation

Use behavioral results as the primary measure of progress:

- Episode success rate, including a separate view of episodes completed without human intervention.
- Human intervention rate and the amount of time spent correcting actions.
- Online episodes or robot interaction time needed to improve success.
- Motion quality, safety-limit events, and task completion consistency across repeated trials.

Critic, edit-policy, and VLA losses are useful training diagnostics, but falling losses alone do not
show that the robot is improving. Evaluate checkpoints with a fixed task setup and comparable
starting conditions. Include autonomous trials to measure the policy without human assistance.

## Checkpoints and resume

The run directory contains checkpoints and replay data. With the default `shared_replay` setting,
the structure is similar to:

```text
outputs/expo_ft_so101/
  replay/                          # Immutable episode data shared by checkpoints
  checkpoints/<checkpoint-name>/
    pretrained_model/              # Fine-tuned VLA and its processors
    training_state.pt              # Critics, edit policy, optimizers, and replay state
    replay_manifest.json
    expo_ft_config.json
```

Resume by passing the complete checkpoint directory:

```bash
bash examples/expo_ft_so101/scripts/run_online.sh \
  --resume=outputs/expo_ft_so101/checkpoints/<checkpoint-name>
```

Keep the run's `replay/` directory with the checkpoints when relocating or resuming a run. The
`pretrained_model/` directory can be used separately for ordinary LeRobot inference, but it does
not contain the EXPO-FT critics or edit policy. For continued EXPO-FT training, retain the complete
checkpoint and replay data. `num_episodes` is the total target, including episodes already completed
in a resumed run.

The checkpoint retention settings are `save_every`, `checkpoint_keep_last`, and `shared_replay`.
Shared replay avoids duplicating episode image data in every checkpoint, though model and optimizer
state are still saved per checkpoint. A checkpoint save may temporarily need extra disk space before
older snapshots are removed.

## Data and implementation notes

Replay stores camera frames as uint8 images in system RAM and evicts whole episodes when it reaches
`replay_capacity`. `storage_image_size` can reduce replay memory by resizing images while preserving
aspect ratio; ensure the chosen resolution retains details needed for the task.

The actor stores the joint targets actually returned by `robot.send_action`, including hardware-side
clipping. The configured follower speed limit and optional `joint_bounds` apply during policy
execution and human correction. Verify robot limits and reset behavior on your hardware before
online training. This example does not replace supervision by an operator or a physical emergency
stop.

For algorithm details, see the [EXPO-FT paper](https://arxiv.org/abs/2605.25477) and the
[authors' reference implementation](https://github.com/pd-perry/expo-ft).
