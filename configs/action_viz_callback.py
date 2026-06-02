# SPDX-License-Identifier: OpenMDW-1.1
"""EveryNActionViz — in-training visual eval (action-chunk plots + video prediction).

Every ``every_n`` steps (and at step 0 when ``run_at_start``), runs the model's
sampler on a FIXED set of eval samples and logs to W&B:
  - action-chunk joint plots (predicted vs ground truth) for the action modes,
  - predicted-vs-GT video for the video (generative) modes,
  - scalar action-MSE / video-PSNR.

There is no train/val split in this project, so eval samples are drawn (once, fixed)
from the training data via mode-pinned DK1LeRobotDataset instances — a qualitative
monitor, not held-out generalization.

The whole eval is wrapped in try/except: a failure is logged loudly but never kills
training. Generation is an FSDP collective (all ranks call it with the same batch);
only rank 0 builds plots and logs to W&B.
"""
from __future__ import annotations

import io
import traceback
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import wandb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log

from dk1_lerobot_dataset import DK1LeRobotDataset, _ACTION_FEATURE

# dk1 14-D layout for plotting: left arm(0-5), left grip(6), right arm(7-12), right grip(13)
_PLOT_GROUPS = [
    ("left_arm", [0, 1, 2, 3, 4, 5]),
    ("left_gripper", [6]),
    ("right_arm", [7, 8, 9, 10, 11, 12]),
    ("right_gripper", [13]),
]


class EveryNActionViz(EveryN):
    def __init__(
        self,
        every_n: int,
        eval_root: str,
        dataset_kwargs: dict,
        n_samples: int = 2,
        action_modes: list[str] | None = None,
        video_modes: list[str] | None = None,
        guidance: float = 1.5,
        num_steps: int = 20,
        fps: int = 30,
        run_at_start: bool = True,
        step_size: int = 1,
    ) -> None:
        super().__init__(every_n, step_size, run_at_start=run_at_start)
        self.name = self.__class__.__name__
        self.eval_root = eval_root
        self.dataset_kwargs = dict(dataset_kwargs)
        self.n_samples = int(n_samples)
        self.action_modes = action_modes or ["policy", "causal_policy"]
        self.video_modes = video_modes or ["policy", "forward_dynamics"]
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.fps = int(fps)
        self._cache: dict[str, list[dict]] = {}     # mode -> list of raw sample dicts
        self._raw_dim = 14

    # ---- fixed eval-sample construction (once) ----
    def _ensure_samples(self) -> None:
        if self._cache:
            return
        modes = sorted(set(self.action_modes) | set(self.video_modes))
        for mode in modes:
            kw = dict(self.dataset_kwargs)
            # deterministic eval on the HELD-OUT split: pin the mode, no RTC/caption
            # randomness, only the last-N eval episodes.
            kw.update(mode=mode, mode_probs=None, rtc_prob=0.0, cfg_dropout=0.0, split="eval")
            ds = DK1LeRobotDataset(root=self.eval_root, **kw)
            idxs = self._high_motion_indices(ds, min(self.n_samples, len(ds)))
            self._cache[mode] = [ds[i] for i in idxs]
        log.info(f"[action-viz] cached eval samples (high-motion): "
                 + ", ".join(f"{m}={len(v)}" for m, v in self._cache.items()))

    @staticmethod
    def _high_motion_indices(ds, n: int, n_candidates: int = 300) -> list[int]:
        """Pick the n windows with the MOST joint motion (static segments are
        uninformative for judging motion prediction). Scores candidates cheaply from
        the raw joint actions (no video decode): total per-joint range over the chunk."""
        N = len(ds)
        if N == 0 or n <= 0:
            return []
        joint = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        cl = ds._chunk_length
        cand = list(range(0, N, max(1, N // n_candidates)))

        def motion(i: int) -> float:
            s = ds._valid_starts[i]
            a = np.asarray([ds._rows[s + t][_ACTION_FEATURE] for t in range(cl)], dtype=np.float32)[:, joint]
            return float((a.max(0) - a.min(0)).sum())  # total joint range over the chunk

        cand.sort(key=motion, reverse=True)
        picked: list[int] = []
        for i in cand:  # greedily take highest-motion windows, spaced ≥2 chunks apart (distinct clips)
            if all(abs(i - p) >= 2 * cl for p in picked):
                picked.append(i)
                if len(picked) >= n:
                    break
        return sorted(picked) if picked else sorted(cand[:n])

    def _build_batch(self, samples: list[dict]) -> dict:
        device = "cuda"
        H, W = samples[0]["video"].shape[-2], samples[0]["video"].shape[-1]
        return {
            "video": [[s["video"].to(device)] for s in samples],
            "action": [[s["action"].to(device)] for s in samples],
            "raw_action_dim": [s["raw_action_dim"] for s in samples],
            "mode": [s["mode"] for s in samples],
            "ai_caption": [s["ai_caption"] for s in samples],
            "conditioning_fps": [s["conditioning_fps"] for s in samples],
            # per-sample [target_h, target_w, orig_h, orig_w] (pixel space); orig==target
            # (our clips already fill the 544x736 bucket) → no latent crop.
            "image_size": [torch.tensor([H, W, H, W], dtype=torch.long, device=device) for _ in samples],
            "domain_id": [s["domain_id"] for s in samples],
            "sequence_plan": [s["sequence_plan"] for s in samples],
            "text_token_ids": [s["text_token_ids"] for s in samples] if "text_token_ids" in samples[0] else None,
        }

    def _to_np(self, a: torch.Tensor) -> np.ndarray:
        # NORMALIZED action [T, raw_dim] (model space, clamped ~[-action_clip,action_clip]).
        # No de-norm → MSE and plots are on the uniform scale the model is trained on, so
        # the metric reflects quality and isn't swamped by tiny absolute relative-joint deltas.
        return a[:, : self._raw_dim].detach().float().cpu().numpy()

    def _action_figure(self, mode: str, gt: np.ndarray, pred: np.ndarray) -> "plt.Figure":
        T = gt.shape[0]
        t = np.arange(T)
        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        for ax, (name, dims) in zip(axes.flat, _PLOT_GROUPS):
            for d in dims:
                ax.plot(t, gt[:, d], "-", lw=1.5, alpha=0.7, label=f"d{d} gt" if len(dims) == 1 else None)
                ax.plot(t, pred[:, d], "--", lw=1.5, label=f"d{d} pred" if len(dims) == 1 else None)
            ax.set_title(name); ax.set_xlabel("step"); ax.grid(alpha=0.3)
            ax.set_ylim(-1.6, 1.6)  # fixed normalized axis → close lines == low MSE (no autoscale)
        fig.suptitle(f"{mode}: NORMALIZED action chunk (solid=GT, dashed=pred)")
        fig.tight_layout()
        return fig

    @staticmethod
    def _to_video_np(vid: torch.Tensor) -> np.ndarray:
        # vid: [C,T,H,W] any range → [T,C,H,W] uint8
        v = vid.detach().float().cpu()
        if v.min() < -0.01:
            v = (v + 1.0) / 2.0
        if v.max() > 1.5:  # already 0-255
            v = v / 255.0
        v = v.clamp(0, 1)
        return (v.permute(1, 0, 2, 3).numpy() * 255.0).astype(np.uint8)

    @torch.no_grad()
    def every_n_impl(self, trainer, model: ImaginaireModel, data_batch, output_batch, loss, iteration: int) -> None:
        try:
            self._ensure_samples()
            info: dict[str, Any] = {}
            for mode, samples in self._cache.items():
                batch = self._build_batch(samples)
                seeds = list(range(len(samples)))
                out = model.generate_samples_from_batch(
                    batch, guidance=self.guidance, num_steps=self.num_steps, seed=seeds,
                )
                if not distributed.is_rank0() or wandb.run is None:
                    continue
                # ----- action plot (action modes) -----
                if mode in self.action_modes and out.get("action") is not None:
                    mses = []
                    for s in range(min(len(samples), len(out["action"]))):  # one plot per cached sample
                        gt = self._to_np(samples[s]["action"])            # normalized GT
                        pred = self._to_np(out["action"][s])              # normalized pred
                        Tm = min(gt.shape[0], pred.shape[0])
                        fig = self._action_figure(mode, gt[:Tm], pred[:Tm])
                        info[f"action_viz/{mode}_chunk_s{s}"] = wandb.Image(fig)
                        plt.close(fig)
                        mses.append(float(np.mean((gt[:Tm] - pred[:Tm]) ** 2)))
                    if mses:
                        info[f"action_viz/{mode}_mse"] = float(np.mean(mses))  # avg over samples
                # ----- video (video modes): one per cached sample -----
                if mode in self.video_modes and out.get("vision") is not None:
                    for s in range(min(len(samples), len(out["vision"]))):
                        vlat = out["vision"][s]  # already [B=1, C=48, T, H, W] from the sampler
                        if vlat.dim() == 4:
                            vlat = vlat.unsqueeze(0)
                        pred_vid = model.decode(vlat).squeeze(0)  # [C,T,H,W]
                        gt_np = self._to_video_np(samples[s]["video"])
                        pred_np = self._to_video_np(pred_vid)
                        Tm = min(gt_np.shape[0], pred_np.shape[0])
                        pair = np.concatenate([gt_np[:Tm], pred_np[:Tm]], axis=3)  # side-by-side on W
                        info[f"action_viz/{mode}_video_gt_vs_pred_s{s}"] = wandb.Video(pair, fps=self.fps, format="mp4")
            if distributed.is_rank0() and wandb.run is not None and info:
                info["trainer/global_step"] = iteration
                wandb.log(info, step=iteration)
                log.info(f"[action-viz] iter {iteration}: logged {sorted(info.keys())}")
        except Exception:
            log.critical(f"[action-viz] eval FAILED at iter {iteration} (training continues):\n{traceback.format_exc()}",
                         rank0_only=False)
        finally:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            torch.cuda.empty_cache()
