# dk1 SFT dry-run checklist

Files:
- experiment config: `configs/dk1_action_sft_nano.py` (registers `dk1_action_sft_nano`)
- TOML overrides: `configs/dk1_action_sft_nano.toml` (LoRA, 2-GPU, max_iter=10)
- launch: `scripts/launch_dk1_sft.sh`
- dataset: `data/dk1_lerobot_dataset.py`, stats: `data/dk1_action_normalization.json`
- base DCP weights: `checkpoints/Cosmos3-Nano-dcp/`

Registration: `make_config()` was patched (config.py) to import modules in
`$COSMOS_EXTRA_EXPERIMENTS`; the launcher sets it to `dk1_action_sft_nano` and puts
`cosmos-dk1/{configs,data}` on PYTHONPATH. Also patched: `domain_utils` (dk1 = domain 25).

## Pre-GPU (CPU) checks — do these first
- [ ] `load_experiment_from_toml(dk1_action_sft_nano.toml)` resolves with the experiment
      registered (COSMOS_EXTRA_EXPERIMENTS hook fires). No Hydra/pydantic errors.
- [ ] LoRA knobs land on `model.config` (lora_enabled/rank/alpha/target_modules).
- [ ] dataloader resolves to `DK1LeRobotDataset` with our root/stats.

## GPU dry-run (after FastWAM frees the GPUs)
- [ ] `bash scripts/launch_dk1_sft.sh` — base DCP loads, LoRA adapters init, FSDP shards across 2 GPUs.
- [ ] **CRITICAL — trainable set**: `param_count` callback must show `action2llm`, `llm2action`,
      `action_modality_embed` as TRAINABLE (not just LoRA). dk1 is a new embodiment (domain 25), so its
      action projection rows are untrained and MUST be learned — LoRA-only would leave them at random init.
      If LoRA forces requires_grad=False on them, un-freeze explicitly (they're in keys_to_select but may
      need requires_grad=True set). See discussion: LoRA insufficient for a new action representation.
- [ ] Memory fits 2×80GB (16B base frozen + LoRA + activations + VAE tokenizer).
- [ ] A few training steps produce finite loss; checkpoint saves at iter 10.

## Things most likely to need fixing (first-pass assumptions)
1. **Video frames/resolution**: dataset emits T=chunk+1=17 frames @ 540×640. Model card says
   video ≤5 frames. May need to cap frames (reduce chunk_length, or subsample video to ≤5)
   and/or set a target resolution the VAE tokenizer expects.
2. **action-dataset → PackingDataLoader contract**: confirm the packer/model accept our sample
   dict (video uint8 [C,T,H,W] + action [chunk,14] + domain_id + mode + idle_frames). Possibly
   needs `raw_action_dim` or `initial_pose`. The collate already special-cases video/action/raw_action_dim.
3. **WAN2.2 VAE**: tokenizer uses the framework default VAE path; ensure it's present/downloadable.
4. **`mode="joint"`**: randomizes forward/inverse/policy. For pure behavior cloning, pin `mode="policy"`.
5. **Blend**: dry run uses ONE root ($DK1_DATA_ROOT). Expand `datasets=dict(...)` in the experiment
   config to the 21-source datamix (size-proportional ratios) for the real run.
6. **compile off** for the dry run; enable later (safe under FSDP/DDP, like the FastWAM finding).
