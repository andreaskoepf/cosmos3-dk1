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
  blend, **action chunk_length=32**, bs=8/grad_accum=2, token budget 65536 (~71GB/80GB, ~29s/iter on 2×H100),
  25k iters, warmup 1000.
- `configs/dk1_action_sft_nano.py` / `.toml` — LoRA-only baseline experiment.
- `configs/per_mode_loss.py` — `PerModeLossCallback`: per-sample loss bucketed by training mode → W&B
  `train_per_mode/<mode>_<modality>` + `train_per_mode_frac/<mode>` (see **W&B metrics**).
- `configs/action_viz_callback.py` — `EveryNActionViz`: in-training eval that generates on the held-out split
  and logs action-chunk joint plots + GT-vs-pred video to W&B (see **In-training eval**).
- `data/dk1_lerobot_dataset.py` — `DK1LeRobotDataset` (3-cam concat-view, relative joint actions + configurable
  `action_clip`, multi-mode masking + RTC, JSON caption metadata + CFG dropout, per-dataset eval split,
  configurable camera fit mode, video_reader-rs decode) and `DK1BlendedDataset` (weighted blend of 21 sources
  as one dataset).
- `data/dk1_action_normalization_relchunk32.json` — **chunk-aware** q01/q99 action stats (chunk-start-relative
  over 32-step windows) over all 21 sources; grippers mapped (0,1)→(-1,1). The launcher's `$DK1_ACTION_STATS`
  points here. Regenerate with `scripts/compute_dk1_action_stats.py --relative --chunk-length 32`.
  (`data/dk1_action_normalization.json` is the older per-frame/chunk-16 stats, kept for reference.)
- `data/cartesian_fk.py` + `scripts/compute_cartesian.py` + `urdf/dk1_dual_arm.urdf` — FK pipeline that builds
  per-arm EE-pose caches (`cache/cartesian/<dataset>/cartesian.parquet`) for the cartesian action space
  (vendored from [open-thought/fastwam](https://github.com/open-thought/fastwam), Apache-2.0). Run once:
  `python scripts/compute_cartesian.py <dataset-dir>...`.
- `data/dk1_action_normalization_cartesian.json` — single-step cartesian-delta stats (regenerate with
  `scripts/compute_dk1_cartesian_stats.py`).
- `requirements.txt` — extra deps beyond cosmos-framework (video-reader-rs, pytorch-kinematics).
- `scripts/launch_dk1_sft_optb.sh` — launcher for the current recipe. `scripts/launch_dk1_sft.sh` — baseline.
- `docs/` — lerobot/Cosmos compatibility notes, dry-run checklist.
- `framework_patches.diff` — **required** patches to a `cosmos-framework` checkout (see Setup).

## Setup
1. Clone + install `NVIDIA/cosmos-framework` (cu130 venv); then the extras:
   `uv pip install -r requirements.txt` (video-reader-rs; pytorch-kinematics — only needed to build cartesian FK caches).
2. Apply the framework patches: `git -C <cosmos-framework> apply framework_patches.diff` — 4 files:
   - `domain_utils.py` — register the `dk1` (domain 25, dim 14) and `dk1_cartesian` (domain 26, dim 20) embodiments.
   - `configs/base/config.py` — `COSMOS_EXTRA_EXPERIMENTS` hook in `make_config()` to register
     out-of-tree experiments.
   - `model/vfm/omni_mot_model.py` — (a) `add_lora` reads `$LORA_ALSO_TRAIN` to keep the listed non-LoRA
     modules (action I/O **and** the gen attention) trainable; (b) exposes
     `flow_matching_loss_action_per_instance` for per-mode loss logging.
   - `utils/vfm/lora.py` — path-qualified LoRA target matching (e.g. `mlp_moe_gen.gate_proj`), so MLP LoRA
     hits only the generation expert and not the reasoner's same-leaf `mlp.gate_proj`.
3. Download `nvidia/Cosmos3-Nano` and convert to DCP (`convert_model_to_dcp`).
4. Point env vars in the launcher at your paths, then `bash scripts/launch_dk1_sft_optb.sh`. Env knobs:
   - `WANDB_MODE=offline` — dry run, no W&B upload.
   - `VIDEO_FIT_MODE=crop|pad|stretch` — camera→bucket fit (default `crop`); drives training + viz (for ablation).
   - `PER_MODE_LOG_FREQ=<n>` — per-mode loss cadence (default 100). `ACTION_VIZ_EVERY_N=<n>` — viz cadence (default 250).
   - `DK1_ACTION_STATS=<path>` — action-stats JSON (default depends on `ACTION_SPACE`).
   - `ACTION_SPACE=joint|cartesian` — action representation (default `joint`; see **Action space**).
   - `ACTION_LOSS_WEIGHT=<f>` — action-vs-vision loss weight (launcher default 10 joint / 2 cartesian).
   - `DK1_CARTESIAN_CACHE=<dir>` — FK EE-pose cache dir (default `cache/cartesian`).

## Action space (`ACTION_SPACE`, default `joint`)
- **`joint`** — 14D, `[left_arm(6), left_gripper, right_arm(6), right_gripper]`, joints chunk-start-relative,
  grippers absolute. Uses the `dk1` embodiment + `dk1_action_normalization_relchunk32.json`, `action_loss_weight=10`.
- **`cartesian`** — 20D bimanual EE pose deltas mirroring Cosmos' DROID layout: per arm
  `[pos_delta(3), rot6d_delta(6), gripper(1)]`, left then right. Deltas are **single-step**
  (`backward_framewise`) of the realized end-effector trajectory (from the FK cache, via `pose_abs_to_rel`);
  grippers absolute. Uses the `dk1_cartesian` embodiment + `dk1_action_normalization_cartesian.json`, and the
  launcher defaults `action_loss_weight=2` (the 10× default starves the video objective). Single-step deltas are
  stationary and locally video-aligned, and Cosmos was designed for this convention. **Prereq:** build the FK
  caches first — `python scripts/compute_cartesian.py <dataset-dir>...` (needs `pytorch-kinematics`), then the
  stats via `scripts/compute_dk1_cartesian_stats.py`. DK-1 data is joints-only, so EE poses come from FK over
  `urdf/dk1_dual_arm.urdf` (link `tool0`).

## Data preprocessing (configurable)
- **Camera fit** (`video_fit_mode`): head cam fills the top `head_height_frac` (⅔) of the 544×736 bucket,
  the two wrist cams the bottom. `crop` = resize-cover + center-crop (real pixels, no borders, crops some FOV);
  `stretch` = resize-to-tile (full FOV, distorts aspect); `pad` = legacy mirror-padded. Ablate via `VIDEO_FIT_MODE`.
- **Action chunk** = 32 steps, joints **chunk-start-relative** (each step minus the chunk-start pose), grippers
  absolute→(−1,1). Quantile-normalized then clamped to **±`action_clip`** (default 1.5 — headroom past the q01/q99
  tail; Cosmos hard-clamps ±1, FastWAM used 5). Stats must be **chunk-aware** for the chunk length (see above) or
  the longer-horizon tail clips.
- **Caption** (`caption_metadata=True`): the task string is enriched into the framework's `ActionPromptJsonFormatter`
  JSON (viewpoint/duration/fps/resolution) — matches DROID training + inference. `cfg_dropout=0.1` empties the
  caption on 10% of samples (enables classifier-free guidance). No system prompt (Cosmos convention).
- **Eval split** (`eval_last_n_episodes=2`): the LAST N episodes of EACH source are held out (deterministic,
  per-dataset → stable under datamix changes, no train↔eval leakage on resume). Training uses `split=train`; the
  viz callback uses `split=eval`.

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
  Raw, **unweighted** per-instance losses (`action_loss_weight` applies only to the total: 10 joint / 2 cartesian).
- `train_per_mode_frac/<mode>` — realized mode share (sanity-checks `mode_probs`).

## In-training eval (`EveryNActionViz`)
Every `ACTION_VIZ_EVERY_N` steps (and at iter 1 via `run_at_start`), generates with
`model.generate_samples_from_batch` on a FIXED held-out (`split=eval`) set per mode and logs to W&B:
- `action_viz/<mode>_chunk` — matplotlib joint plots, predicted (dashed) vs GT (solid), for policy & causal_policy.
- `action_viz/<mode>_video_gt_vs_pred` — decoded GT|pred video for policy & forward_dynamics.
- `action_viz/<mode>_mse` — scalar.

FSDP-safe (all ranks generate, rank-0 logs) and failure-isolated (a generation error is logged but never kills
training). There is no separate train/val split otherwise — these eval samples come from the per-dataset held-out
episodes, so it's a qualitative monitor, not a generalization benchmark.

## Key choices
- **chunk_length=32** (divisible by 4 → 33 video frames = 4·8+1 for clean VAE temporal); longer horizon, fewer replans.
- **Action space: joint (default) or cartesian** — joint-space (14D chunk-relative) sidesteps the cartesian/
  orientation-noise IK failures seen on resting arms *at deployment*. Cartesian (20D single-step EE deltas,
  `ACTION_SPACE=cartesian`) mirrors Cosmos' DROID design (stationary, video-aligned deltas) and is under
  evaluation; its IK risk is deployment-only — as a world-model/training input it's unaffected.
- **Attention full-FT + gen-MLP LoRA + full action I/O** (`action2llm`/`llm2action`/`action_modality_embed`)
  — a fresh embodiment must be learned, not just adapted; the 5.44B MLP stays frozen-but-LoRA-adapted to fit 2×H100.
- **Multi-mode** training (policy / causal_policy / forward_dynamics / inverse_dynamics) for action policy,
  cheap causal realtime inference, and a controllable world model in one model.
- Built on, and shares the datamix with, the FastWAM DK-1 pretrain.
