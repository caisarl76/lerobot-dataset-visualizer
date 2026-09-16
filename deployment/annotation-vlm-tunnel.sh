#!/usr/bin/env bash
# Keep the annotation VLM connection available across temporary network outages.
set -u
while true; do
  ssh -N -o BatchMode=yes -o ConnectTimeout=10 -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L 127.0.0.1:34002:127.0.0.1:34002 h100
  echo "VLM tunnel disconnected; reconnecting in 5 seconds" >&2
  sleep 5
done
