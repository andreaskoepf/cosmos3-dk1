# SPDX-License-Identifier: OpenMDW-1.1
"""DK-1 bimanual LeRobot dataset for Cosmos 3 action SFT.

Adapted from cosmos_framework's DROIDLeRobotDataset. Differences:
  - 3 dk1 cameras: head (exterior, top) + left_wrist/right_wrist (bottom row).
  - 14D **joint-space** action read directly from the `action` column
    (vs DROID's 10D cartesian pose deltas) — no FK/pose math.
  - dk1 embodiment / domain id (registered in domain_utils as "dk1" = 25).
  - per-episode videos + from/to timestamps (already supported by the Cosmos
    decode path — see cosmos-dk1/docs/lerobot_compat.md).

Action layout (matches FastWAM dk1): [left_arm(6), left_gripper(1),
right_arm(6), right_gripper(1)] = 14D, indices 0..13.

⚠ First-pass / TODO (validate at dry-run):
  - `initial_pose` is omitted (joint-space has no single EE pose). Confirm the
    action training_step doesn't require it for forward/inverse/policy modes;
    if it does, supply an FK-derived pose or identity.
  - Gripper kept raw (DROID inverts via 1-g); quantile-norm handles range.
  - Action is **absolute** normalized joint setpoints. If the recipe expects
    relative/delta joint targets, switch to per-step deltas + recompute stats.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from lerobot.datasets.video_utils import decode_video_frames
from torch.utils.data import Dataset

from cosmos_framework.data.vfm.action.action_normalization import load_action_stats, normalize_action
from cosmos_framework.data.vfm.action.action_spec import Gripper, Joint, build_action_spec
from cosmos_framework.data.vfm.action.domain_utils import get_domain_id
from cosmos_framework.data.vfm.action.json_formatter import ActionPromptJsonFormatter
from cosmos_framework.data.vfm.action.pose_utils import compute_idle_frames
from cosmos_framework.data.vfm.sequence_packing import SequencePlan, add_special_tokens
from cosmos_framework.model.vfm.vlm.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate

_MAX_TEXT_TOKENS = 1024

# Physical dk1 cameras → view slots. head is the third-person/exterior view.
_TOP_VIEW = "observation.images.head"
_BOTTOM_LEFT = "observation.images.left_wrist"
_BOTTOM_RIGHT = "observation.images.right_wrist"
_ACTION_FEATURE = "action"            # 14D joint+gripper setpoints
_STATE_FEATURE = "observation.state"  # 40D: pos[:14] = joint+gripper, then vel[12], torque[14]

# 14D action layout: [left_arm(0-5), left_gripper(6), right_arm(7-12), right_gripper(13)].
_JOINT_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
_GRIPPER_DIMS = [6, 13]

_MODE_CHOICES = ("forward_dynamics", "inverse_dynamics", "policy", "causal_policy")


def dk1_action_spec():
    """14D bimanual joint-space ActionSpec."""
    return build_action_spec(
        Joint(n=6, label="left_arm"), Gripper(prefix="left"),
        Joint(n=6, label="right_arm"), Gripper(prefix="right"),
    )


class DK1LeRobotDataset(Dataset):
    """DK-1 bimanual joint-space action dataset (single LeRobot root).

    Blend across the 21-source dk1 datamix is handled at the config level
    (one instance per root + ConcatDataset / weighted sampler), mirroring how
    FastWAM weighted its blend.
    """

    def __init__(
        self,
        root: str,
        normalization_path: str,
        fps: float = 30.0,
        chunk_length: int = 16,
        # Per-sample training mode. "joint" → roll one of forward_dynamics /
        # inverse_dynamics / policy PER batch entry; a fixed mode name pins every
        # sample. Each mode sets its OWN condition mask (see _build_result), so a
        # packed batch mixes modes, each entry with its own clean/noisy pattern.
        mode: str = "joint",
        # Weights for the "joint" per-sample mode roll: dict over the 3 modes (missing
        # modes → weight 0). None → uniform. e.g. {"policy":0.6,"forward_dynamics":0.25,
        # "inverse_dynamics":0.15}. Ignored when `mode` is a fixed mode name.
        mode_probs: dict | None = None,
        tolerance_s: float = 1e-2,  # looser than DROID's 2e-4; dk1 ts precision
        # Observation conditioning: number of CLEAN latent vision frames at the
        # clip start. At 4x temporal compression, 2 latents ≈ 5 obs pixel frames
        # (matches FastWAM's 5-obs window: 1 + (5-1)//4 = 2). 0 → pure generation.
        num_clean_latent_frames: int = 2,
        temporal_compression_factor: int = 4,
        # RTC action prefix: MAX number of clean/conditioning action steps (the
        # rtc_max_prefix knob). When > 0 and the per-sample RTC gate fires (see
        # rtc_prob), the first K action steps are given clean and the rest predicted.
        # 0 → predict the whole chunk (no action conditioning, RTC disabled).
        rtc_action_prefix: int = 0,
        # RTC probability: per-sample chance of applying a (nonzero) clean action
        # prefix when rtc_action_prefix>0. Mirrors FastWAM's rtc_prob. 1.0 → every
        # sample gets a prefix; 0.25 → 25% conditioned, 75% cold-start (K=0). No
        # effect when rtc_action_prefix==0.
        rtc_prob: float = 1.0,
        # RTC prefix-length decay: when the gate fires, K∈[1,k_max] is sampled with
        # P(K=k) ∝ exp(-rtc_decay·k) (FastWAM parity — short prefixes favored).
        # 0.0 → uniform over [1, k_max].
        rtc_decay: float = 0.3,
        # Action target representation:
        #   True  → joint dims RELATIVE to the chunk-start obs state
        #           (observation.state[:14]); grippers absolute. Stats must be
        #           computed on the same relative deltas (compute_dk1_action_stats
        #           --relative, with gripper q01=0/q99=1 → linear (0,1)→(-1,1)).
        #   False → absolute joint setpoints.
        relative_actions: bool = True,
        # Qwen VLM tokenizer config (Hydra ref to model.config.vlm_config.tokenizer).
        # Required for real training — the joint dataloader needs text_token_ids.
        tokenizer_config: Any = None,
        # Output video size (H, W). Must be a valid Cosmos resolution bucket
        # (divisible by 32). Default = the 256p 4:3 bucket (W=320, H=256). The
        # tiled 3-cam frame is resized to this; keep the model's resolution config
        # consistent (e.g. "256").
        video_hw: tuple = (256, 320),
        # Pad the 14-D dk1 action up to the model's action width (max_action_dim).
        # raw_action_dim=14 is emitted so the loss masks the padded dims.
        max_action_dim: int = 64,
        # Video decode backend: "video_reader_rs" (fast Rust, frame-index, AV1;
        # needs video_reader_rs.libs on LD_LIBRARY_PATH) or "lerobot" (torchcodec
        # CPU, by timestamp). video_reader_rs hides the GPU-starving CPU decode.
        video_backend: str = "video_reader_rs",
        # Caption handling (framework parity with ActionTransformPipeline). When
        # caption_metadata=True, the plain task caption is enriched via
        # ActionPromptJsonFormatter into a structured JSON string carrying viewpoint,
        # duration, fps and resolution before tokenization — matching how DROID/Cosmos
        # action training and the inference _format_prompt build the prompt. When
        # caption_idle_frames=True, Pi0.7-style idle/total action-frame metadata is
        # also added. cfg_dropout replaces the caption with "" with that probability
        # (classifier-free-guidance dropout — required to enable CFG at inference).
        caption_metadata: bool = False,
        caption_idle_frames: bool = False,
        cfg_dropout: float = 0.0,
        # Deterministic per-dataset eval split: the LAST `eval_last_n_episodes`
        # episodes (by episode_index) are reserved for eval. split="train" excludes
        # them, split="eval" uses only them, split="all" (or n==0) disables the split.
        # Per-dataset + by episode order → stable when datasets are added/removed and
        # safe to resume with an extended datamix (no random train↔eval leakage).
        eval_last_n_episodes: int = 0,
        split: str = "train",
        # Fraction of the bucket HEIGHT given to the head cam (top); the wrist cams
        # share the remaining height (bottom, split L|R). Cameras are resize-cropped
        # (cover + center-crop, FastWAM-style) to exactly fill the bucket — NO mirror
        # padding (which would inject redundant pixels).
        head_height_frac: float = 2.0 / 3.0,
        # How cameras are fit to the bucket (configurable for ablation):
        #   "crop"    — resize-to-cover + center-crop each cam to fill its tile exactly
        #               (FastWAM-style; real pixels, no borders, crops some FOV).
        #   "stretch" — resize each cam straight to its tile, IGNORING aspect ratio
        #               (full FOV, no borders, but distorts geometry).
        #   "pad"     — tile head + half-size wrists, downscale-to-fit, center reflection-pad
        #               (keeps full FOV + aspect, but injects redundant mirrored borders).
        # head_height_frac sets the head/wrist height split for "crop" and "stretch".
        video_fit_mode: str = "crop",
    ) -> None:
        super().__init__()
        self._video_hw = (int(video_hw[0]), int(video_hw[1]))
        self._max_action_dim = int(max_action_dim)
        self._video_backend = str(video_backend)
        self._root = Path(root)
        self._fps = float(fps)
        self._chunk_length = int(chunk_length)
        self._mode = mode
        self._mode_names = list(_MODE_CHOICES)
        if mode_probs is not None:
            unknown = [m for m in mode_probs if m not in _MODE_CHOICES]
            if unknown:
                raise ValueError(f"mode_probs has unknown modes {unknown}; valid: {_MODE_CHOICES}")
            self._mode_weights = [float(mode_probs.get(m, 0.0)) for m in self._mode_names]
            if sum(self._mode_weights) <= 0:
                raise ValueError(f"mode_probs must sum to >0, got {mode_probs}")
        else:
            self._mode_weights = None
        self._relative_actions = bool(relative_actions)
        self._vlm_tokenizer = None
        if tokenizer_config is not None:
            self._vlm_tokenizer = lazy_instantiate(tokenizer_config).tokenizer
            self._vlm_tokenizer, _ = add_special_tokens(self._vlm_tokenizer)
        self._tolerance_s = float(tolerance_s)
        self._num_clean_latent_frames = int(num_clean_latent_frames)
        self._temporal_compression_factor = int(temporal_compression_factor)
        self._rtc_action_prefix = int(rtc_action_prefix)
        self._rtc_prob = float(rtc_prob)
        self._rtc_decay = float(rtc_decay)
        self._head_height_frac = float(head_height_frac)
        if video_fit_mode not in ("crop", "pad", "stretch"):
            raise ValueError(f"video_fit_mode must be 'crop'|'pad'|'stretch', got {video_fit_mode!r}")
        self._video_fit_mode = str(video_fit_mode)
        self._caption_idle_frames = bool(caption_idle_frames)
        self._cfg_dropout = float(cfg_dropout)
        self._caption_formatter = (
            ActionPromptJsonFormatter(
                viewpoint_templates={
                    "concat_view": "Top row: head (third-person) camera. "
                    "Bottom row: left-wrist and right-wrist cameras."
                }
            )
            if caption_metadata
            else None
        )
        self._normalization_path = str(normalization_path)
        self._domain_id = get_domain_id("dk1")
        self._norm_stats: dict[str, torch.Tensor] | None = None

        self._info = json.loads((self._root / "meta" / "info.json").read_text())
        self._episodes = {
            int(row["episode_index"]): row
            for path in sorted((self._root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            for row in pq.read_table(path).to_pylist()
        }
        tasks_path = self._root / "meta" / "tasks.parquet"
        self._tasks = (
            {int(r["task_index"]): str(r["task"]) for r in pq.read_table(tasks_path).to_pylist()}
            if tasks_path.exists() else {}
        )
        self._rows = sorted(
            (
                row
                for path in sorted((self._root / "data").glob("chunk-*/file-*.parquet"))
                for row in pq.read_table(path).to_pylist()
            ),
            key=lambda row: int(row["index"]),
        )
        # Valid window starts: deterministic per-dataset last-N-episode split, AND
        # every chunk window stays within a single episode (no cross-episode mixing).
        ep_order = sorted(self._episodes.keys())
        n_eval = max(0, int(eval_last_n_episodes))
        if n_eval > 0 and str(split) != "all" and n_eval < len(ep_order):
            held = set(ep_order[-n_eval:])
            allowed = held if str(split) == "eval" else (set(ep_order) - held)
        else:
            allowed = set(ep_order)
        cl = self._chunk_length
        ep_of = [int(r["episode_index"]) for r in self._rows]
        self._valid_starts = [
            i
            for i in range(len(self._rows) - cl)
            if ep_of[i] in allowed and ep_of[i + cl] == ep_of[i]
        ]
        self._split = str(split)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def chunk_length(self) -> int:
        return self._chunk_length

    @property
    def domain_id(self) -> int:
        return self._domain_id

    @property
    def action_dim(self) -> int:
        return 14

    @property
    def action_names(self) -> list[str]:
        return dk1_action_spec().names

    def _choose_mode(self) -> str:
        if self._mode != "joint":
            return self._mode
        if self._mode_weights is not None:
            return random.choices(self._mode_names, weights=self._mode_weights, k=1)[0]
        return random.choice(_MODE_CHOICES)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode = self._choose_mode()
        idx = self._valid_starts[int(idx)]
        first_row = self._rows[idx]
        episode = self._episodes[int(first_row["episode_index"])]

        observation_rows = self._rows[idx : idx + self._chunk_length + 1]
        action_rows = observation_rows[: self._chunk_length]

        video = self._load_concat_video(episode, observation_rows)
        raw_action = self._build_raw_action(observation_rows, action_rows)
        task = self._tasks.get(int(observation_rows[0].get("task_index", 0)), "")
        ai_caption = random.choice(task.split(" | ")) if task else ""

        return self._build_result(mode=mode, video=video, action=raw_action, ai_caption=ai_caption)

    def _load_concat_video(self, episode: dict[str, Any], observation_rows: list[dict[str, Any]]) -> torch.Tensor:
        timestamps = [float(row["timestamp"]) for row in observation_rows]

        def decode(video_key: str) -> torch.Tensor:
            path = self._video_path(episode, video_key)
            from_ts = float(episode.get(f"videos/{video_key}/from_timestamp", 0.0))
            abs_ts = [from_ts + ts for ts in timestamps]
            if self._video_backend == "video_reader_rs":
                return self._decode_vrs(path, abs_ts)
            return decode_video_frames(path, abs_ts, self._tolerance_s)

        if self._video_fit_mode == "pad":
            # Legacy: head full-size on top, half-size wrists on bottom, then
            # downscale-to-fit + center reflection-pad to the bucket (redundant borders).
            top = decode(_TOP_VIEW)
            left = decode(_BOTTOM_LEFT)
            right = decode(_BOTTOM_RIGHT)
            _, _, h, w = top.shape
            half_h, half_w = h // 2, w // 2
            left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
            right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
            tiled = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)
            return self._fit_to_bucket(tiled)

        # "crop"/"stretch": fit each camera to a tile so the composite EXACTLY fills the
        # (H, W) bucket — no padding. Head gets the top `head_height_frac` of the height
        # (full width); the two wrists split the bottom. crop=cover+center-crop (keep aspect),
        # stretch=resize-to-tile (give up aspect).
        H, W = self._video_hw
        head_h = max(1, round(H * self._head_height_frac))
        wrist_h = H - head_h
        left_w = W // 2
        right_w = W - left_w

        def fit(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
            if self._video_fit_mode == "stretch":
                return F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
            return self._resize_crop(img, h, w)

        top = fit(decode(_TOP_VIEW), head_h, W)
        left = fit(decode(_BOTTOM_LEFT), wrist_h, left_w)
        right = fit(decode(_BOTTOM_RIGHT), wrist_h, right_w)
        bottom = torch.cat([left, right], dim=-1)  # [T,3,wrist_h,W]
        return torch.cat([top, bottom], dim=-2)    # [T,3,H,W] — exact, no padding

    @staticmethod
    def _resize_crop(img: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[T,3,oh,ow] → [T,3,h,w]: scale (preserving aspect) so both sides ≥ target,
        then center-crop the excess. FastWAM-style — fills the tile with real pixels,
        no stretching, no padding."""
        _, _, oh, ow = img.shape
        scale = max(h / oh, w / ow)
        nh, nw = max(h, round(oh * scale)), max(w, round(ow * scale))
        if (nh, nw) != (oh, ow):
            img = F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)
        top, left = (nh - h) // 2, (nw - w) // 2
        return img[..., top : top + h, left : left + w]

    def _decode_vrs(self, path: str, abs_ts: list[float]) -> torch.Tensor:
        """Decode frames by index via video_reader-rs (fast Rust decoder).
        Maps absolute timestamps → native video frame indices using the file fps."""
        from video_reader import PyVideoReader
        r = PyVideoReader(path)
        fps = float(r.get_fps())
        n = int(r.get_info().get("frame_count", 0)) or 10 ** 9
        idx = [min(max(int(round(t * fps)), 0), n - 1) for t in abs_ts]
        frames = np.asarray(r.get_batch(idx))  # [T, H, W, 3] uint8
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W] in [0,1]

    def _fit_to_bucket(self, video: torch.Tensor) -> torch.Tensor:
        """(pad mode) Snap a [T,3,H,W] clip to the bucket WITHOUT distorting aspect:
        downscale (preserving aspect) only if it exceeds the bucket, then center
        reflection-pad to the exact bucket size."""
        ht, wt = self._video_hw
        _, _, h, w = video.shape
        scale = min(ht / h, wt / w)
        if scale < 1.0:
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            video = F.interpolate(video, size=(nh, nw), mode="bilinear", align_corners=False)
            _, _, h, w = video.shape
        pad_h, pad_w = ht - h, wt - w
        if pad_h or pad_w:
            pt, pl = pad_h // 2, pad_w // 2
            mode = "reflect" if (pt < h and (pad_h - pt) < h and pl < w and (pad_w - pl) < w) else "replicate"
            video = F.pad(video, (pl, pad_w - pl, pt, pad_h - pt), mode=mode)
        return video

    def _video_path(self, episode: dict[str, Any], video_key: str) -> str:
        chunk_idx = int(episode[f"videos/{video_key}/chunk_index"])
        file_idx = int(episode[f"videos/{video_key}/file_index"])
        rel = self._info["video_path"].format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx,
            episode_chunk=chunk_idx, episode_file=file_idx,
        )
        return str(self._root / rel)

    def _build_raw_action(
        self, observation_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]
    ) -> torch.Tensor:
        action = np.asarray([row[_ACTION_FEATURE] for row in action_rows], dtype=np.float32)
        action = action[-self._chunk_length :].copy()  # [chunk_length, 14]
        if self._relative_actions:
            # Joints relative to the chunk-start observation state (pos = first 14
            # of observation.state, same layout as action). Grippers stay absolute.
            obs_ref = np.asarray(observation_rows[0][_STATE_FEATURE], dtype=np.float32)[:14]
            action[:, _JOINT_DIMS] -= obs_ref[_JOINT_DIMS]
        return torch.from_numpy(action).float()

    def _build_result(self, *, mode: str, video: torch.Tensor, action: torch.Tensor, ai_caption: str) -> dict[str, Any]:
        spec = dk1_action_spec()
        idle_frames = compute_idle_frames(
            action, spec,
            eps_t=5e-3 / self._fps, eps_r=np.deg2rad(1.5) / self._fps,
            eps_g=1e-2, joint_threshold=5e-3 / self._fps, min_streak=3,
        )
        normalized_action = normalize_action(action, "quantile", self._load_norm_stats())
        # Zero-pad the 14-D action up to the model's action width (max_action_dim).
        raw_dim = normalized_action.shape[-1]
        if self._max_action_dim > raw_dim:
            normalized_action = F.pad(normalized_action, (0, self._max_action_dim - raw_dim))
        # causal_policy: truncate the clip to the observation (past) window so NO
        # future video latents enter the sequence — actions then attend only to clean
        # past frames (causal; matches realtime inference where future frames don't
        # exist). obs_pixel pixel frames encode to num_clean_latent_frames latents
        # (t_latent = 1 + (obs_pixel-1)//tcf). This is the only mode that drops frames.
        if mode == "causal_policy":
            tcf = self._temporal_compression_factor
            obs_pixel = 1 + max(0, self._num_clean_latent_frames - 1) * tcf
            video = video[:obs_pixel]
        formatted_video = (video * 255.0).clamp(0.0, 255.0).to(torch.uint8).permute(1, 0, 2, 3)

        # --- Conditioning plan ---
        # Vision: first `num_clean_latent_frames` latent frames are clean (the
        # observation window); the rest of the clip is generated/supervised.
        t_pixel = video.shape[0]
        t_latent = 1 + (t_pixel - 1) // self._temporal_compression_factor
        num_clean = max(0, min(self._num_clean_latent_frames, t_latent - 1))
        # Per-mode conditioning (clean = given as context; the rest are noised and
        # supervised). The mode is chosen per sample (_choose_mode), so a packed
        # batch mixes modes — each entry carries its own condition mask.
        #   policy           : first `num_clean` video latents clean + RTC action
        #                      prefix clean → predict remaining video + actions.
        #   forward_dynamics : first `num_clean` video latents clean + ALL actions
        #                      clean (trajectory given) → predict future video (world model).
        #   inverse_dynamics : ALL video latents clean + NO actions clean → predict
        #                      all actions from the fully-observed (full-clip) video.
        #   causal_policy    : clip already truncated to the past obs window above, so
        #                      ALL its (past) latents are clean + RTC action prefix →
        #                      predict the action chunk from past frames only, NO video
        #                      generation (cheap, causal — realtime inference path).
        if mode == "forward_dynamics":
            vision_clean = list(range(num_clean))
            action_clean = list(range(self._chunk_length))
        elif mode == "inverse_dynamics":
            vision_clean = list(range(t_latent))
            action_clean = []
        else:  # policy / causal_policy — both predict the action chunk with an optional
            # RTC clean prefix; they differ only in which video latents are clean.
            # causal_policy already dropped future frames (truncated clip) → all present
            # latents clean; policy keeps the full clip → only the first num_clean clean
            # (future video generated). RTC (FastWAM parity): with prob rtc_prob a clean
            # prefix K∈[1,k_max] is given, P(K=k) ∝ exp(-rtc_decay·k); else K=0 (cold start).
            vision_clean = list(range(t_latent)) if mode == "causal_policy" else list(range(num_clean))
            k = 0
            if self._rtc_action_prefix > 0 and random.random() < self._rtc_prob:
                k_max = min(self._rtc_action_prefix, self._chunk_length - 1)
                if self._rtc_decay > 0 and k_max > 1:
                    ks = range(1, k_max + 1)
                    weights = [math.exp(-self._rtc_decay * kk) for kk in ks]
                    k = random.choices(list(ks), weights=weights, k=1)[0]
                else:
                    k = random.randint(1, k_max)
            action_clean = list(range(k))
        sequence_plan = SequencePlan(
            has_text=bool(ai_caption),
            has_vision=True,
            condition_frame_indexes_vision=vision_clean,
            has_action=True,
            condition_frame_indexes_action=action_clean,
        )

        result = {
            "ai_caption": ai_caption,
            "video": formatted_video,
            "action": normalized_action,
            "conditioning_fps": torch.tensor(self._fps, dtype=torch.long),
            "mode": mode,
            "domain_id": torch.tensor(self._domain_id, dtype=torch.long),
            "raw_action_dim": torch.tensor(raw_dim, dtype=torch.long),  # 14 — loss masks padding
            "viewpoint": "concat_view",
            "idle_frames": torch.tensor(idle_frames, dtype=torch.long),
            "sequence_plan": sequence_plan,
            "additional_view_description": (
                "The top row is the head (third-person) camera. The bottom row contains the "
                "left-wrist and right-wrist camera views, horizontally concatenated."
            ),
        }
        # Caption: classifier-free-guidance dropout (empty caption with prob cfg_dropout
        # → unconditional sample, enables CFG at inference) OR framework-parity metadata
        # enrichment (viewpoint/duration/fps/resolution as a JSON string). A fresh dict
        # is passed to the formatter (it mutates/pops its input) so `result` is untouched.
        caption = ai_caption
        if self._cfg_dropout > 0.0 and random.random() < self._cfg_dropout:
            caption = ""
        elif self._caption_formatter is not None and caption:
            fmt_input: dict[str, Any] = {
                "ai_caption": caption,
                "viewpoint": "concat_view",
                "video": formatted_video,  # [C, T, H, W] — formatter reads shape[1] for duration
                "conditioning_fps": result["conditioning_fps"],
                "image_size": torch.tensor(self._video_hw, dtype=torch.long),  # [H, W]
                "action": normalized_action,
                "mode": mode,
            }
            if self._caption_idle_frames:
                fmt_input["idle_frames"] = result["idle_frames"]
                fmt_input["idle_frames_total"] = torch.tensor(self._chunk_length, dtype=torch.long)
            cap_obj = self._caption_formatter(fmt_input)[self._caption_formatter.caption_key]
            caption = json.dumps(cap_obj) if isinstance(cap_obj, dict) else cap_obj
        result["ai_caption"] = caption
        if self._vlm_tokenizer is not None:
            ids = tokenize_caption(caption, self._vlm_tokenizer, is_video=True,
                                   use_system_prompt=False)[:_MAX_TEXT_TOKENS]
            result["text_token_ids"] = torch.tensor(ids, dtype=torch.long)
        return result

    def _load_norm_stats(self) -> dict[str, torch.Tensor]:
        if self._norm_stats is None:
            self._norm_stats = {
                k: torch.from_numpy(v).float()
                for k, v in load_action_stats(self._normalization_path).items()
            }
        return self._norm_stats

    def __len__(self) -> int:
        return len(self._valid_starts)


class DK1BlendedDataset(Dataset):
    """Weighted blend of per-root DK1LeRobotDatasets exposed as ONE dataset.

    RankPartitionedDataLoader pins one dataset per rank (requires
    world_size >= num_datasets), so a 21-source blend on 2 GPUs must be a single
    dataset. __getitem__ samples a sub-dataset by weight, then a random window
    from it (stochastic weighted mixing — matches the FastWAM blend semantics).
    """

    def __init__(self, roots_weights, tokenizer_config: Any = None, **ds_kwargs) -> None:
        super().__init__()
        # Build the Qwen tokenizer ONCE and share it across all sub-datasets
        # (avoids 21x tokenizer loads).
        shared_tok = None
        if tokenizer_config is not None:
            shared_tok = lazy_instantiate(tokenizer_config).tokenizer
            shared_tok, _ = add_special_tokens(shared_tok)
        self._datasets = []
        for root, _w in roots_weights:
            d = DK1LeRobotDataset(root=root, tokenizer_config=None, **ds_kwargs)
            d._vlm_tokenizer = shared_tok
            self._datasets.append(d)
        w = np.asarray([float(x) for _r, x in roots_weights], dtype=np.float64)
        self._probs = w / w.sum()
        self._total = int(sum(len(d) for d in self._datasets))

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        di = int(np.random.choice(len(self._datasets), p=self._probs))
        d = self._datasets[di]
        return d[random.randrange(len(d))]
