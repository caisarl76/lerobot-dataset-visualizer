# Annotation service on h100 GPU 0

The local visualizer and CPU annotation backend use Qwen3.6-27B through an
SSH tunnel. All three upstream modules remain enabled. Deployment defaults
limit concurrency to four VLM requests and one episode; the UI can override
individual settings without losing the endpoint configuration.

## GPU service

Qwen weights use `/mnt/data01/huggingface` on h100. Download them before the
switch (the existing Cosmos container provides the Hugging Face client):

```bash
ssh h100 'docker exec jihun-cosmos3-nano python3 -c '\''from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.6-27B", ignore_patterns=["*.md", "*.png", "*.jpg"]))'\'''
ssh h100 bash -s < deployment/start-qwen-h100.sh
ssh h100 docker logs --tail 40 jihun-lerobot-qwen36-gpu0
```

The script stops Cosmos and retains its container for rollback. Qwen uses
only GPU 0, 90% GPU memory, a 32,768-token context, eight CPUs, and a 96 GiB
host-memory limit. The existing vLLM 0.23.0 image is pinned by local image ID.
Its API binds only to h100 loopback port 34002; no public endpoint is opened.

## Local services

After installing the backend and building the UI as described in
[the backend guide](../backend/README.md), run from the repository root:

```bash
bash deployment/start-annotation-local.sh
curl --fail http://127.0.0.1:34002/v1/models
curl --fail http://127.0.0.1:7861/api/annotation/config
```

Open <http://127.0.0.1:3000/annotate>. The dedicated `lerobot-annotation`
tmux session contains `tunnel`, `backend`, and `ui` windows. Logs, caches,
and independent dataset drafts live under
`/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations`.
The local services survive closing the terminal; rerun the script after a
workstation reboot. Qwen's container restarts with Docker unless stopped.

`LEROBOT_ANNOTATE_CONFIG` selects the partial JSON defaults in
`annotation-h100.json`. API credentials, if needed, belong only in the
backend's `LEROBOT_VLM_API_KEY` environment variable.

To stop this local stack, use `tmux kill-session -t lerobot-annotation`.
To restore the former Cosmos service on h100:

```bash
ssh h100 docker stop jihun-lerobot-qwen36-gpu0
ssh h100 docker start jihun-cosmos3-nano
```

This rollback restores the former port 34001 service; change the annotation
model/endpoint defaults if the annotation backend should use Cosmos.

## Verified deployment (2026-09-07)

- Model revision: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- GPU 0: NVIDIA H100 80 GB; Qwen serving uses approximately 73,060 MiB.
- Real camera image inference succeeded through the SSH tunnel.
- Original `pnp_trash` converted to an independent 92-episode draft;
  official validation passed with no errors or warnings.
- All three enabled generation modules ran for episode 34 (21.88 seconds).
  Output: 13 persistent rows and 45 event rows; validation passed across
  the resulting 92-episode dataset. Only episode 34 was selected for generation.
- Emitted styles: task augmentation (11), plan (1), subtask (1), speech (1),
  VQA (44). No interjection or memory atoms were emitted for this episode;
  enabled modules do not guarantee every style for every episode.
- Prepared draft: `local/annotation-17ad579896f60259`.
- Generated draft: `local/annotation-74de5f7d90831524`;
  [review episode 34](http://127.0.0.1:3000/local/annotation-74de5f7d90831524/34).
- Generation job: `8c8604a1624048818a93d1a2c5632117`.

Validation checks structure and timing. Generated semantic labels still need
human review in the visualizer before training export.
