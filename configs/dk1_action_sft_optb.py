# SPDX-License-Identifier: OpenMDW-1.1
"""``dk1_action_sft_optb`` — Cosmos3-Nano gen-expert SFT, "Option B" recipe.

Differs from ``dk1_action_sft_nano`` (the LoRA-only baseline) in the gen-expert
training split and in RTC action conditioning:

  - GEN ATTENTION (q/k/v/o_moe_gen, ~1.51B) is FULL fine-tuned — not LoRA-rank-16.
    Done via ``$LORA_ALSO_TRAIN`` (un-freezes the matching non-LoRA params) plus the
    optimizer ``keys_to_select`` (TOML) selecting those modules.
  - GEN MLP (mlp_moe_gen gate/up/down, ~5.44B) is LoRA-adapted (kept frozen in the
    base recipe). The LoRA targets are PATH-QUALIFIED (``mlp_moe_gen.gate_proj`` …)
    so they hit ONLY the generation expert, not the frozen reasoner's ``mlp.*``
    (requires the path-aware LoRA matcher patch in cosmos_framework/utils/vfm/lora.py).
  - RTC training ENABLED: per-sample clean action prefix, FastWAM-parity
    (rtc_prob=0.25, rtc_action_prefix=8, exp-decay length).

Memory (2xH100, FSDP shard=2): attn full-FT adds ~12GB/GPU opt state over the LoRA
baseline; MLP-LoRA opt is ~0.5GB. Probe the token budget to confirm headroom.
"""
import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader

from dk1_lerobot_dataset import DK1LeRobotDataset, DK1BlendedDataset  # noqa: F401
from per_mode_loss import PerModeLossCallback

cs = ConfigStore.instance()

_NANO = copy.deepcopy(NANO_MODEL_CONFIG)
_NANO["resolution"] = "480"
_NANO["rectified_flow_training_config"]["shift"] = 5

# Full 21-source dk1 datamix (weight ∝ frames/1000) — identical to the baseline.
_DK1_DATAMIX = [
    ("swan_0513", "/workspace/data/dk1_black_and_white_swan_2026-05-13", 192.710),
    ("swan_0514", "/workspace/data/dk1_black_and_white_swan_2026-05-14", 194.446),
    ("cutlery_0421", "/workspace/data/dk1_cutlery_basket_2026-04-21", 312.364),
    ("cutlery_0422", "/workspace/data/dk1_cutlery_basket_2026-04-22", 180.708),
    ("cutlery_0423", "/workspace/data/dk1_cutlery_basket_2026-04-23", 140.543),
    ("cutlery_0424", "/workspace/data/dk1_cutlery_basket_2026-04-24", 147.945),
    ("cutlery_0425", "/workspace/data/dk1_cutlery_basket_2026-04-25", 90.887),
    ("cutlery_0426", "/workspace/data/dk1_cutlery_basket_2026-04-26_recover1", 40.701),
    ("duplo_disasm", "/workspace/data/dk1_duplo_disassembly_2026-04-04", 86.441),
    ("duplo_box_0408", "/workspace/data/dk1_duplo_in_box_2026-04-08", 97.310),
    ("duplo_box_0417", "/workspace/data/dk1_duplo_in_box_2026-04-17", 88.054),
    ("duplo_box_artlight", "/workspace/data/dk1_duplo_in_box_2026-04-18_artificial_light", 54.216),
    ("duplo_box_dense", "/workspace/data/dk1_duplo_in_box_2026-04-18_dense_center", 87.780),
    ("duplo_box_few", "/workspace/data/dk1_duplo_in_box_2026-04-18_few_blocks", 82.048),
    ("duplo_box_one", "/workspace/data/dk1_duplo_in_box_one_2026-04-09", 27.892),
    ("duplo_box_two", "/workspace/data/dk1_duplo_in_box_two_2026-04-09", 31.173),
    ("duplo_sort", "/workspace/data/dk1_duplo_sorting_by_size_2026-04-05", 60.393),
    ("duplo_stack", "/workspace/data/dk1_duplo_stack_simple", 65.812),
    ("dk1_merge", "/workspace/data/dk1-merge-2026-03", 2230.018),
    ("robotwin_three", "/workspace/data/robotwin_dk1_realcam_stack_blocks_three_1000", 323.557),
    ("robotwin_two", "/workspace/data/robotwin_dk1_realcam_stack_blocks_two_1000", 217.335),
]

# Multi-mode training: mode="joint" rolls a mode PER SAMPLE, weighted by mode_probs.
#   policy (0.30)          — 2 obs latents clean → predict actions + FUTURE VIDEO (joint imagine+act).
#   causal_policy (0.30)   — clip truncated to past obs window (no future frames) → predict
#                            actions from clean past frames only, NO video gen (cheap realtime path).
#   forward_dynamics(0.25) — start frame + GIVEN action trajectory → predict future video (world model).
#   inverse_dynamics(0.15) — FULLY observed video → predict actions (non-causal).
# A packed batch mixes all four, each entry with its own mask; the loss follows each mask.
# policy+FD (0.55) keep training future-video prediction → shapes the shared video features
# that causal_policy's action decode reads (the FastWAM finding). RTC applies to policy &
# causal_policy: with prob 0.25 a clean prefix K∈[1,8] is given, K~exp(-0.3K).
_DK1_BLEND = L(DK1BlendedDataset)(
    roots_weights=[[root, w] for _name, root, w in _DK1_DATAMIX],
    normalization_path="${oc.env:DK1_ACTION_STATS}",
    fps=30.0, chunk_length=16, mode="joint",
    mode_probs={"policy": 0.30, "causal_policy": 0.30, "forward_dynamics": 0.25, "inverse_dynamics": 0.15},
    num_clean_latent_frames=2,
    rtc_action_prefix=8, rtc_prob=0.25, rtc_decay=0.3,
    # Framework-parity caption: enrich with viewpoint/duration/fps/resolution JSON
    # (matches DROID training + inference _format_prompt) + CFG dropout for guidance.
    caption_metadata=True, cfg_dropout=0.1,
    relative_actions=True,
    tokenizer_config="${model.config.vlm_config.tokenizer}",
    video_hw=(544, 736),
)

dk1_action_sft_optb = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "local"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(project="cosmos3", group="sft", name="dk1_action_sft_optb", wandb_mode="disabled"),
        model=dict(config=_NANO),
        optimizer=dict(
            betas=[0.9, 0.95], eps=1.0e-06, fused=True,
            keys_to_select=["moe_gen", "time_embedder", "vae2llm", "llm2vae"],  # overridden by TOML
            lr=5.0e-04, lr_multipliers={}, optimizer_type="AdamW", weight_decay=0,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaCosine", cycle_lengths=[1000],
            f_max=[1.0], f_min=[0.0], f_start=[0.0], verbosity_interval=0, warm_up_steps=[50],
        ),
        trainer=dict(
            distributed_parallelism="fsdp", grad_accum_iter=2, logging_iter=1, max_iter=500,
            run_validation=False, run_validation_on_start=False, save_zero_checkpoint=False,
            seed=42, timeout_period=999999999,
            cudnn=dict(benchmark=True, deterministic=False),
            grad_scaler_args=dict(enabled=False),
            # Per-mode loss logging (merges into the basic/optimization/job_monitor
            # callback set). Buckets per-sample loss by data_batch["mode"] → W&B
            # train_per_mode/* and train_per_mode_frac/*.
            callbacks=dict(per_mode_loss=L(PerModeLossCallback)(
                log_freq=int(os.environ.get("PER_MODE_LOG_FREQ", "100")))),
        ),
        checkpoint=dict(
            keys_to_skip_loading=["net_ema."],
            load_path="???",
            load_training_state=False, strict_resume=False, save_iter=100, verbose=True,
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000, dataset_name="default",
            max_samples_per_batch=None, max_sequence_length=65536, patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16, tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=8, in_order=True, num_workers=8, persistent_workers=True,
                pin_memory=True, prefetch_factor=4, sampler=None,
                datasets=dict(action=dict(ratio=1, dataset=_DK1_BLEND)),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

for _item in [dk1_action_sft_optb]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
