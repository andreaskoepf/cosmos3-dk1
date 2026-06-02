# SPDX-License-Identifier: OpenMDW-1.1
"""``dk1_action_sft_nano`` — Cosmos3-Nano world+action SFT on the DK-1 datamix.

First-pass experiment config, modeled on cosmos_framework's ``vision_sft_nano``.
Swaps the video JSONL dataset for our ``DK1LeRobotDataset`` (which returns BOTH
``video`` and ``action`` per sample → drives world + action generation, both of
which NANO_MODEL_CONFIG enables: vision_gen=True, action_gen=True).

Register + run (LoRA / paths come from the paired TOML):
    PYTHONPATH=.:/workspace/code/cosmos-dk1/configs:/workspace/code/cosmos-dk1/data \\
      torchrun --nproc_per_node=2 -m cosmos_framework.scripts.train \\
      --sft-toml=/workspace/code/cosmos-dk1/configs/dk1_action_sft_nano.toml

⚠ DRY-RUN VALIDATION TARGETS (see cosmos-dk1/docs/dry_run_checklist.md):
  - action-dataset → PackingDataLoader contract (video tokenization of the
    540x640, T=chunk+1 frames; the model card mentions ≤5 video frames — may
    need to cap num_video_frames / chunk_length).
  - whether the action path needs `initial_pose` / `raw_action_dim` keys.
  - blend: currently a single root via $DK1_DATA_ROOT; expand to the 21-source
    datamix by adding entries to `datasets=dict(...)` with size-proportional ratio.
"""
import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict
from cosmos_framework.configs.base.experiment.sft.models.nano_model_config import NANO_MODEL_CONFIG
from cosmos_framework.data.vfm.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader

# DK1LeRobotDataset lives in cosmos-dk1/data (must be on PYTHONPATH).
from dk1_lerobot_dataset import DK1LeRobotDataset, DK1BlendedDataset

cs = ConfigStore.instance()

# Nano model config tweaked for the dk1 run at 480p:
#  - resolution "480" — matches the dataset's video_hw bucket (544x736, 4:3).
#  - scalar shift=5 instead of the resolution-dict {256:3,480:5,720:10}: an int
#    shift skips the per-sample resolution lookup the model otherwise requires,
#    and 5 is the correct 480p value. (If multi-res training is wanted later,
#    restore the dict and emit per-sample resolution info from the dataset.)
_NANO = copy.deepcopy(NANO_MODEL_CONFIG)
_NANO["resolution"] = "480"
_NANO["rectified_flow_training_config"]["shift"] = 5

# Full 21-source dk1 datamix (same blend + weights as FastWAM; weight ∝ frames/1000).
# dk1-merge dominates (~2230); RoboTwin synthetic + dk1 teleop fill the rest.
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


# Single blended dataset over all 21 roots (RankPartitionedDataLoader needs
# num_datasets <= world_size, so the blend must be one dataset).
_DK1_BLEND = L(DK1BlendedDataset)(
    roots_weights=[[root, w] for _name, root, w in _DK1_DATAMIX],
    normalization_path="${oc.env:DK1_ACTION_STATS}",
    fps=30.0, chunk_length=16, mode="joint",
    num_clean_latent_frames=2, rtc_action_prefix=0, relative_actions=True,
    tokenizer_config="${model.config.vlm_config.tokenizer}",
    video_hw=(544, 736),
)

dk1_action_sft_nano = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdacosine"},
            {"override /checkpoint": "local"},   # load from a local DCP dir (ckpt_type=dcp below)
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /cluster": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(project="cosmos3", group="sft", name="dk1_action_sft_nano", wandb_mode="disabled"),
        model=dict(config=_NANO),
        optimizer=dict(
            betas=[0.9, 0.95], eps=1.0e-06, fused=True,
            # overridden to LoRA-only by the TOML (keys_to_select=["lora_"]).
            keys_to_select=["moe_gen", "time_embedder", "vae2llm", "llm2vae"],
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
            # No partial callback overrides here — they'd lack a `_target_` base
            # unless the matching callback group is selected above. Defaults from
            # the basic/optimization/job_monitor groups apply.
        ),
        checkpoint=dict(
            keys_to_skip_loading=["net_ema."],
            load_path="???",            # set by TOML (BASE_CHECKPOINT_PATH = DCP dir)
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
                # Single blended dataset over all 21 sources (weighted internally).
                datasets=dict(action=dict(ratio=1, dataset=_DK1_BLEND)),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

for _item in [dk1_action_sft_nano]:
    _name = [k for k, v in globals().items() if v is _item][0]
    cs.store(group="experiment", package="_global_", name=_name, node=_item)
