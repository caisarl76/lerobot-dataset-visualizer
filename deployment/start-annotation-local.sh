#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export LEROBOT_ANNOTATE_CONFIG="$PWD/deployment/annotation-h100.json"
export LEROBOT_ANNOTATE_EXPORT=/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations/workspace
export LEROBOT_ANNOTATE_CACHE=/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations/cache
export LEROBOT_ANNOTATE_BROWSER_ORIGIN=http://127.0.0.1:3000
export NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861
logs=/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations/logs
mkdir -p "$LEROBOT_ANNOTATE_EXPORT" "$LEROBOT_ANNOTATE_CACHE" "$logs"
if tmux has-session -t lerobot-annotation 2>/dev/null; then
  echo 'lerobot-annotation already exists; inspect with tmux attach -t lerobot-annotation'
  exit 1
fi
tmux new-session -d -s lerobot-annotation -n tunnel \
  "bash '$PWD/deployment/annotation-vlm-tunnel.sh' > '$logs/tunnel.log' 2>&1"
tmux new-window -t lerobot-annotation -n backend \
  "cd '$PWD'; LEROBOT_ANNOTATE_CONFIG='$LEROBOT_ANNOTATE_CONFIG' LEROBOT_ANNOTATE_EXPORT='$LEROBOT_ANNOTATE_EXPORT' LEROBOT_ANNOTATE_CACHE='$LEROBOT_ANNOTATE_CACHE' LEROBOT_ANNOTATE_BROWSER_ORIGIN='$LEROBOT_ANNOTATE_BROWSER_ORIGIN' backend/.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 7861 > '$logs/backend.log' 2>&1"
tmux new-window -t lerobot-annotation -n ui \
  "cd '$PWD'; NEXT_PUBLIC_ANNOTATE_BACKEND_URL='$NEXT_PUBLIC_ANNOTATE_BACKEND_URL' bun run start --hostname 127.0.0.1 --port 3000 > '$logs/ui.log' 2>&1"
echo 'UI: http://127.0.0.1:3000/annotate'
echo "Logs: $logs"
