# cosmos3-dk1

Fine-tune **NVIDIA Cosmos 3 (Nano)** on the **DK-1 bimanual robot** datamix — multi-mode world + action SFT.
Adapts the [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework) action post-training to a new
DK-1 embodiment (14-D bimanual joint-space). See [`PLAN.md`](PLAN.md) for the full design.

## Two recipes
- **`dk1_action_sft_optb`** — the current recipe (what we train). Gen-expert SFT: **attention full
  fine-tuned + gen-MLP LoRA + action I/O full-FT**, multi-mode training, RTC, per-mode loss logging.
- **`dk1_action_sft_nano`** — the simpler LoRA-only baseline (LoRA on the gen attention + full action I/O).
  Kept for reference / ablation.

## Layout
- `configs/dk1_action_sft_optb.py` / `.toml` — **current** experiment. Attention (`q/k/v/o_moe_gen`, ~1.51B)
  full-FT via `$LORA_ALSO_TRAIN`; gen MLP (`mlp_moe_gen.{gate,up,down}_proj`, ~5.44B) LoRA r16 with
  **path-qualified** targets (gen only, not the frozen reasoner's `mlp.*`); action I/O full-FT. 480p, 21-source
  blend, bs=8/grad_accum=2, token budget 65536 (~71GB/80GB, ~29s/iter on 2×H100), 25k iters.
- `configs/dk1_action_sft_nano.py` / `.toml` — LoRA-only baseline experiment.
- `configs/per_mode_loss.py` — `PerModeLossCallback`: buckets per-sample loss by training mode → W&B
  `train_per_mode/<mode>_<modality>` + `train_per_mode_frac/<mode>` (see **W&B metrics** below).
- `data/dk1_lerobot_dataset.py` — `DK1LeRobotDataset` (3-cam concat-view, relative joint actions,
  5-frame conditioning, multi-mode masking + RTC, video_reader-rs decode, reflection-pad to bucket) and
  `DK1BlendedDataset` (weighted blend of the 21 sources as one dataset).
- `data/dk1_action_normalization.json` — q01/q99 action stats over all 21 sources (relative joints,
  grippers mapped (0,1)→(-1,1)). Regenerate with `scripts/compute_dk1_action_stats.py`.
- `scripts/launch_dk1_sft_optb.sh` — launcher for the current recipe. `scripts/launch_dk1_sft.sh` — baseline.
- `docs/` — lerobot/Cosmos compatibility notes, dry-run checklist.
- `framework_patches.diff` — **required** patches to a `cosmos-framework` checkout (see Setup).

## Setup
1. Clone + install `NVIDIA/cosmos-framework` (cu130 venv); also `pip install video-reader-rs==0.4.3`.
2. Apply the framework patches: `git -C <cosmos-framework> apply framework_patches.diff` — 4 files:
   - `domain_utils.py` — register the `dk1` embodiment (domain 25, raw action dim 14).
   - `configs/base/config.py` — `COSMOS_EXTRA_EXPERIMENTS` hook in `make_config()` to register
     out-of-tree experiments.
   - `model/vfm/omni_mot_model.py` — (a) `add_lora` reads `$LORA_ALSO_TRAIN` to keep the listed non-LoRA
     modules (action I/O **and** the gen attention) trainable; (b) exposes
     `flow_matching_loss_action_per_instance` for per-mode loss logging.
   - `utils/vfm/lora.py` — path-qualified LoRA target matching (e.g. `mlp_moe_gen.gate_proj`), so MLP LoRA
     hits only the generation expert and not the reasoner's same-leaf `mlp.gate_proj`.
3. Download `nvidia/Cosmos3-Nano` and convert to DCP (`convert_model_to_dcp`).
4. Point env vars in the launcher at your paths, then `bash scripts/launch_dk1_sft_optb.sh`
   (`WANDB_MODE=offline` for a dry run; `PER_MODE_LOG_FREQ=<n>` to change per-mode log cadence).

## Multi-mode training
`mode="joint"` rolls one mode **per sample**, weighted by `mode_probs` (current: policy 0.30 /
causal_policy 0.30 / forward_dynamics 0.25 / inverse_dynamics 0.15). A packed batch mixes modes; each
entry carries its own condition mask and the loss follows that mask. Cosmos gen-attention is bidirectional,
so causality is enforced by which tokens are *present*, not by an attention mask.

| mode | given (clean) | predicted | direction |
|---|---|---|---|
| `policy` | 2 obs latents | actions **+ future video** | observe → imagine + act |
| `causal_policy` | past obs window only (clip **truncated**, no future latents) | actions (no video gen) | **causal realtime action decode** (FastWAM-style) |
| `forward_dynamics` | start frame + **full action trajectory** | future video | action → frames (controllable **world model**) |
| `inverse_dynamics` | full clip (all latents) | actions | frames → actions (non-causal) |

policy + forward_dynamics (0.55) keep training future-video prediction, which shapes the shared
gen-transformer video features that `causal_policy`'s action decode reads (the FastWAM finding).
**RTC** (policy & causal_policy): with prob `rtc_prob` a clean action prefix `K∈[1,rtc_action_prefix]` is
given, `P(K=k) ∝ exp(-rtc_decay·k)` (FastWAM parity).

## W&B metrics (per mode)
Logged every `PER_MODE_LOG_FREQ` steps (default 100) as a 100-step windowed mean:
- `train_per_mode/<mode>_<modality>` — `<mode> ∈ {policy, causal_policy, forward_dynamics, inverse_dynamics}`,
  `<modality> ∈ {vision, action}`. Signal-carrying keys: `policy_action`, `causal_policy_action`,
  `inverse_dynamics_action`, `policy_vision`, `forward_dynamics_vision`; the rest sit at ~0 (masked modality).
  Raw, **unweighted** per-instance losses (the `action_loss_weight=10` applies only to the total).
- `train_per_mode_frac/<mode>` — realized mode share (sanity-checks `mode_probs`).

## Key choices
- **Joint-space, relative** action targets (not cartesian) — cartesian/orientation noise causes IK failures
  on resting arms (real-robot finding); joint-space sidesteps IK entirely.
- **Attention full-FT + gen-MLP LoRA + full action I/O** (`action2llm`/`llm2action`/`action_modality_embed`)
  — a fresh embodiment must be learned, not just adapted; the 5.44B MLP stays frozen-but-LoRA-adapted to fit 2×H100.
- **Multi-mode** training (policy / causal_policy / forward_dynamics / inverse_dynamics) for action policy,
  cheap causal realtime inference, and a controllable world model in one model.
- Built on, and shares the datamix with, the FastWAM DK-1 pretrain.
