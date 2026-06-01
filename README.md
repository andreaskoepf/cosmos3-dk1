# cosmos3-dk1

Fine-tune **NVIDIA Cosmos 3 (Nano)** on the **DK-1 bimanual robot** datamix — world + action SFT.
Adapts the [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework) action post-training to a
new DK-1 embodiment (14-D bimanual joint-space). See [`PLAN.md`](PLAN.md) for the full design.

## Layout
- `configs/dk1_action_sft_nano.py` — experiment config (registers `dk1_action_sft_nano`): Nano model,
  LoRA + trainable action modules, the 21-source DK-1 blend, 480p.
- `configs/dk1_action_sft_nano.toml` — TOML overrides (LoRA, optimizer, batch, wandb).
- `data/dk1_lerobot_dataset.py` — `DK1LeRobotDataset` (3-cam concat-view, relative joint actions,
  5-frame conditioning + RTC prefix, video_reader-rs decode, reflection-pad to bucket) and
  `DK1BlendedDataset` (weighted blend of the 21 sources as one dataset).
- `data/dk1_action_normalization.json` — q01/q99 action stats over all 21 sources (relative joints,
  grippers mapped (0,1)→(-1,1)).
- `scripts/launch_dk1_sft.sh` — launcher (env + torchrun).
- `scripts/compute_dk1_action_stats.py` — regenerate the action stats.
- `docs/` — lerobot/Cosmos compatibility notes, dry-run checklist.
- `framework_patches.diff` — **required** patches to a `cosmos-framework` checkout (see below).

## Setup
1. Clone + install `NVIDIA/cosmos-framework` (cu130 venv); also `pip install video-reader-rs==0.4.3`.
2. Apply the framework patches: `git -C <cosmos-framework> apply framework_patches.diff`
   - registers the `dk1` embodiment (`domain_utils`, domain 25, 14-D)
   - a `COSMOS_EXTRA_EXPERIMENTS` hook in `make_config()` to register out-of-tree experiments
   - `LORA_ALSO_TRAIN` to keep the action projection trainable under LoRA
3. Download `nvidia/Cosmos3-Nano` and convert to DCP (`convert_model_to_dcp`).
4. Point env vars in `scripts/launch_dk1_sft.sh` at your paths, then `bash scripts/launch_dk1_sft.sh`.

## Key choices
- **Joint-space, relative** action targets (not cartesian) — cartesian/orientation noise causes IK
  failures on resting arms (real-robot finding); joint-space sidesteps IK entirely.
- **LoRA on the generator attention + full training of the new-embodiment action projection**
  (`action2llm`/`llm2action`/`action_modality_embed`) — a fresh embodiment must be learned, not just adapted.
- Built on, and shares the datamix with, the FastWAM DK-1 pretrain.
