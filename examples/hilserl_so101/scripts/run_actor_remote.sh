#!/usr/bin/env bash
# Remote learner, step 3 (moves the robot): open an SSH tunnel to the remote learner and start the actor here.
# The actor keeps learner_host 127.0.0.1; the tunnel forwards that port to the learner. Extra arguments go to
# the actor. Ctrl+C stops the actor (both arms go home) and closes the tunnel.
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_ec2

if ! ec2_ssh "tmux has-session -t hilserl-learner 2>/dev/null"; then
  echo "The remote learner is not running; start it first: scripts/ec2_learner.sh start" >&2
  exit 1
fi

ssh -i "$EC2_KEY" -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L "$LEARNER_PORT:localhost:$LEARNER_PORT" "$EC2_HOST" &
TUNNEL_PID=$!
trap 'kill "$TUNNEL_PID" 2>/dev/null || true; flush_tty_input' EXIT
sleep 2
if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
  echo "Could not open the tunnel on port $LEARNER_PORT (is a local learner still running?)" >&2
  exit 1
fi
echo "Tunnel open: localhost:$LEARNER_PORT -> $EC2_HOST"

bash "$PROJECT_DIR/scripts/run_actor.sh" "$@"
