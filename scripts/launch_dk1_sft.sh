#!/usr/bin/env bash
# Launch the dk1 LoRA world+action SFT (Cosmos3-Nano) on 2 GPUs.
# DRY RUN by default (TOML max_iter=10). Run only once the FastWAM job frees the GPUs.
set -euo pipefail

COSMOS=/workspace/code/cosmos-framework
DK1=/workspace/code/cosmos-dk1
VENV=$COSMOS/.venv

# torchcodec needs the cu13 NVIDIA libs (libnppicc.so.13); video_reader-rs needs
# its bundled libs (libsharpyuv) — both on LD_LIBRARY_PATH.
export LD_LIBRARY_PATH=$VENV/lib/python3.13/site-packages/nvidia/cu13/lib:$VENV/lib/python3.13/site-packages/video_reader_rs.libs

# Make our dataset + experiment config importable, and register the experiment
# via the COSMOS_EXTRA_EXPERIMENTS hook patched into make_config().
export PYTHONPATH=$COSMOS:$DK1/configs:$DK1/data
export COSMOS_EXTRA_EXPERIMENTS=dk1_action_sft_nano
# Keep the dk1 action projection trainable alongside LoRA (new embodiment must be
# learned, not just adapted). Matches the non-lora_ entries in optimizer.keys_to_select.
export LORA_ALSO_TRAIN=action2llm,llm2action,action_modality_embed

# Paths consumed by the TOML / experiment config.
export BASE_CHECKPOINT_PATH=$DK1/checkpoints/Cosmos3-Nano-dcp
export DK1_DATA_ROOT=${DK1_DATA_ROOT:-/workspace/data/dk1_black_and_white_swan_2026-05-13}
export DK1_ACTION_STATS=$DK1/data/dk1_action_normalization.json
# Wan2.2 VAE for the video tokenizer (same .pth FastWAM uses).
export WAN_VAE_PATH=${WAN_VAE_PATH:-/workspace/code/fastwam/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}

export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$DK1/outputs}
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

# W&B: offline by default (dry run logs locally; no API key needed). To enable
# real-time logging once training is green:
#   WANDB_MODE=online WANDB_API_KEY=<key> bash scripts/launch_dk1_sft.sh
WANDB_MODE=${WANDB_MODE:-online}   # activated; auth via ~/.netrc (or WANDB_API_KEY)
export WANDB_ENTITY=${WANDB_ENTITY:-andreaskoepf}

NPROC=${NPROC:-2}
echo "Launching dk1_action_sft_nano | GPUs=$NPROC | data=$DK1_DATA_ROOT | wandb=$WANDB_MODE"
# Run from the framework root: the model config loads some files via paths
# relative to the repo root (e.g. cosmos_framework/model/.../Qwen3-VL-8B-Instruct.json).
cd "$COSMOS"
# job.wandb_mode override (REMAINDER opts) takes precedence over the TOML.
exec "$VENV/bin/torchrun" --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-12355}" \
    -m cosmos_framework.scripts.train \
    --sft-toml="$DK1/configs/dk1_action_sft_nano.toml" \
    job.wandb_mode="$WANDB_MODE"
