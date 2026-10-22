#!/usr/bin/env bash
# Remote learner, step 2 (no robot): run the learner on the remote GPU machine, inside tmux so it survives a
# dropped connection.
#   scripts/ec2_learner.sh start    start it (extra arguments go to the learner)
#   scripts/ec2_learner.sh log      follow its log (Ctrl+C stops following, not the learner)
#   scripts/ec2_learner.sh status   is it running?
#   scripts/ec2_learner.sh stop     stop it (like Ctrl+C)
#   scripts/ec2_learner.sh pull     copy the checkpoints here (outputs/train/), e.g. for push_hub.sh policy
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_ec2
SESSION="hilserl-learner"
running() { ec2_ssh "tmux has-session -t $SESSION 2>/dev/null"; }

case "${1:-}" in
  start)
    shift
    if running; then
      echo "The learner is already running (scripts/ec2_learner.sh log / stop)." >&2
      exit 1
    fi
    ec2_ssh "cd $REMOTE_PROJECT && tmux new-session -d -s $SESSION bash scripts/train_learner.sh $*"
    echo "Learner started on $EC2_HOST. Follow it with: scripts/ec2_learner.sh log"
    ;;
  log)
    ec2_ssh -t "tail -n 40 -f $REMOTE_PROJECT/outputs/learner.log"
    ;;
  status)
    if running; then echo "running"; else echo "not running"; fi
    ;;
  stop)
    if ! running; then
      echo "The learner is not running."
      exit 0
    fi
    # The learner saves a final checkpoint with the replay buffers first (buffer_save_freq), which takes minutes.
    ec2_ssh "tmux send-keys -t $SESSION C-c"
    echo "Stopping; the learner saves its replay buffers first (a few minutes)..."
    for _ in $(seq 120); do
      running || { echo "Learner stopped."; exit 0; }
      sleep 5
    done
    echo "Still running after 10 min; force it (loses the unsaved buffers) with: ssh ... tmux kill-session -t $SESSION" >&2
    exit 1
    ;;
  pull)
    mkdir -p "$TRAIN_DIR"
    rsync -az -e "ssh -i $EC2_KEY" "$EC2_HOST:$REMOTE_PROJECT/outputs/train/hilserl_so101_leader/" "$TRAIN_DIR/"
    echo "Checkpoints copied to $TRAIN_DIR"
    ;;
  *)
    sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
