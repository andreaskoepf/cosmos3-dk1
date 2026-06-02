#!/usr/bin/env bash
# Launch the dk1 "Option B" gen-expert SFT (Cosmos3-Nano) on 2 GPUs:
#   gen attention FULL-FT + gen MLP LoRA + action I/O FULL-FT + RTC training.
set -euo pipefail

COSMOS=/workspace/code/cosmos-framework
DK1=/workspace/code/cosmos-dk1
VENV=$COSMOS/.venv

export LD_LIBRARY_PATH=$VENV/lib/python3.13/site-packages/nvidia/cu13/lib:$VENV/lib/python3.13/site-packages/video_reader_rs.libs

# Option B peaks at ~70.5GB/80GB (86.5%) at the 65k token budget. expandable_segments
# curbs allocator fragmentation growth over a multi-day run, protecting that headroom.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export PYTHONPATH=$COSMOS:$DK1/configs:$DK1/data
export COSMOS_EXTRA_EXPERIMENTS=dk1_action_sft_optb
# Un-freeze (full-FT) the action I/O AND the gen-expert ATTENTION q/k/v/o. These must
# match the non-lora_ entries in the TOML optimizer.keys_to_select. The gen MLP is
# NOT here — it is adapted via LoRA (lora_target_modules), so it stays frozen+adapted.
export LORA_ALSO_TRAIN=action2llm,llm2action,action_modality_embed,q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen

export BASE_CHECKPOINT_PATH=$DK1/checkpoints/Cosmos3-Nano-dcp
export DK1_DATA_ROOT=${DK1_DATA_ROOT:-/workspace/data/dk1_black_and_white_swan_2026-05-13}
export DK1_ACTION_STATS=$DK1/data/dk1_action_normalization.json
export WAN_VAE_PATH=${WAN_VAE_PATH:-/workspace/code/fastwam/weights/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}

export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$DK1/outputs_optb}
mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"

WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-andreaskoepf}

NPROC=${NPROC:-2}
echo "Launching dk1_action_sft_optb | GPUs=$NPROC | wandb=$WANDB_MODE | out=$IMAGINAIRE_OUTPUT_ROOT"
cd "$COSMOS"
exec "$VENV/bin/torchrun" --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-12365}" \
    -m cosmos_framework.scripts.train \
    --sft-toml="$DK1/configs/dk1_action_sft_optb.toml" \
    job.wandb_mode="$WANDB_MODE"
