#!/usr/bin/env bash
# Run on h100 after downloading Qwen/Qwen3.6-27B into the shared HF cache.
set -euo pipefail
docker stop jihun-cosmos3-nano
if docker container inspect jihun-lerobot-qwen36-gpu0 >/dev/null 2>&1; then
  docker start jihun-lerobot-qwen36-gpu0
else
  docker run -d --name jihun-lerobot-qwen36-gpu0 \
    --gpus device=0 --cpus 8 --memory 96g --shm-size 8g \
    --restart unless-stopped \
    -p 127.0.0.1:34002:8000 \
    -v /mnt/data01/huggingface:/root/.cache/huggingface \
    -v /mnt/data01/jhkim/lerobot-annotation/vllm-cache:/root/.cache/vllm \
    sha256:f37691f675bb82f734f606de8af90e777d3f80a20b120e699fd43fd10e60b8d7 \
    --model Qwen/Qwen3.6-27B --host 0.0.0.0 --port 8000 \
    --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
    --tensor-parallel-size 1 --max-model-len 32768 \
    --gpu-memory-utilization 0.90 --max-num-seqs 4 \
    --max-num-batched-tokens 8192 --reasoning-parser qwen3
fi
