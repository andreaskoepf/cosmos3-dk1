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
        eval_roots: list[str],
        dataset_kwargs: dict,
        n_samples: int = 2,                 # # of FIXED high-motion anchors (each from a distinct dataset)
        add_random: bool = True,            # + 1 fully-random clip per eval (whole task distribution)
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
        self.eval_roots = list(eval_roots) if isinstance(eval_roots, (list, tuple)) else [eval_roots]
        self.dataset_kwargs = dict(dataset_kwargs)
        self.n_samples = int(n_samples)
        self.add_random = bool(add_random)
        self.action_modes = action_modes or ["policy", "causal_policy"]
        self.video_modes = video_modes or ["policy", "forward_dynamics"]
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.fps = int(fps)
        self._modes = sorted(set(self.action_modes) | set(self.video_modes))
        self._ds_cache: dict[tuple, "DK1LeRobotDataset"] = {}   # (root, mode) -> eval dataset (lazy)
        self._anchors: list[tuple[str, int]] | None = None      # fixed [(root, window_idx)], distinct datasets
        self._raw_dim = 14

    def _get_ds(self, root: str, mode: str):
        """Mode-pinned eval dataset for a root (lazy, cached). No tokenizer needed — the
        sampler tokenizes ai_caption itself — so these are cheap to build."""
        key = (root, mode)
        if key not in self._ds_cache:
            kw = dict(self.dataset_kwargs)
            kw.update(mode=mode, mode_probs=None, rtc_prob=0.0, cfg_dropout=0.0,
                      split="eval", tokenizer_config=None)
            self._ds_cache[key] = DK1LeRobotDataset(root=root, **kw)
        return self._ds_cache[key]

    def _ensure_anchors(self) -> None:
        """Pick n_samples FIXED high-motion anchors, each from a DISTINCT dataset (when
        >1 root). Motion is mode-independent, so score with any mode."""
        if self._anchors is not None:
            return
        m0 = self._modes[0]
        roots = self.eval_roots
        n = min(self.n_samples, len(roots)) if len(roots) > 1 else self.n_samples
        step = max(1, len(roots) // max(n, 1))
        chosen = [roots[(i * step) % len(roots)] for i in range(n)] if len(roots) > 1 else roots * n
        anchors: list[tuple[str, int]] = []
        for root in chosen[:self.n_samples]:
            ds = self._get_ds(root, m0)
            idx = self._high_motion_indices(ds, 1)
            if idx:
                anchors.append((root, idx[0]))
        self._anchors = anchors
        from pathlib import Path
        log.info("[action-viz] fixed high-motion anchors: "
                 + ", ".join(f"{Path(r).name}#{i}" for r, i in anchors))

    def _random_spec(self, iteration: int) -> tuple[str, int]:
        """A fully-random eval clip (random dataset + random window). Seeded by `iteration`
        so all FSDP ranks pick the SAME spec (avoids collective mismatch) but it varies
        each eval → samples the whole task distribution over time."""
        import random
        rng = random.Random(int(iteration) * 2654435761 & 0xFFFFFFFF)
        root = rng.choice(self.eval_roots)
        ds = self._get_ds(root, self._modes[0])
        return root, (rng.randrange(len(ds)) if len(ds) else 0)

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
            self._ensure_anchors()
            # specs (root, window_idx) + labels: fixed anchors (a0,a1,...) from distinct
            # datasets + one fully-random clip per eval (whole task distribution).
            specs = list(self._anchors)
            labels = [f"a{k}" for k in range(len(specs))]
            if self.add_random:
                specs.append(self._random_spec(iteration)); labels.append("rand")
            info: dict[str, Any] = {}
            for mode in self._modes:
                samples = [self._get_ds(r, mode)[i] for (r, i) in specs]
                batch = self._build_batch(samples)
                out = model.generate_samples_from_batch(
                    batch, guidance=self.guidance, num_steps=self.num_steps, seed=list(range(len(samples))),
                )
                if not distributed.is_rank0() or wandb.run is None:
                    continue
                # ----- action plots (action modes): one per spec -----
                if mode in self.action_modes and out.get("action") is not None:
                    anchor_mses = []
                    for k in range(min(len(samples), len(out["action"]))):
                        gt = self._to_np(samples[k]["action"])            # normalized GT
                        pred = self._to_np(out["action"][k])              # normalized pred
                        Tm = min(gt.shape[0], pred.shape[0])
                        fig = self._action_figure(mode, gt[:Tm], pred[:Tm])
                        info[f"action_viz/{mode}_chunk_{labels[k]}"] = wandb.Image(fig)
                        plt.close(fig)
                        mse = float(np.mean((gt[:Tm] - pred[:Tm]) ** 2))
                        if labels[k] == "rand":
                            info[f"action_viz/{mode}_mse_rand"] = mse   # noisy, whole-distribution sample
                        else:
                            anchor_mses.append(mse)
                    if anchor_mses:
                        info[f"action_viz/{mode}_mse"] = float(np.mean(anchor_mses))  # clean trend (fixed anchors)
                # ----- video (video modes): one per spec -----
                if mode in self.video_modes and out.get("vision") is not None:
                    for k in range(min(len(samples), len(out["vision"]))):
                        vlat = out["vision"][k]  # already [B=1, C=48, T, H, W] from the sampler
                        if vlat.dim() == 4:
                            vlat = vlat.unsqueeze(0)
                        pred_vid = model.decode(vlat).squeeze(0)  # [C,T,H,W]
                        gt_np = self._to_video_np(samples[k]["video"])
                        pred_np = self._to_video_np(pred_vid)
                        Tm = min(gt_np.shape[0], pred_np.shape[0])
                        pair = np.concatenate([gt_np[:Tm], pred_np[:Tm]], axis=3)  # side-by-side on W
                        info[f"action_viz/{mode}_video_gt_vs_pred_{labels[k]}"] = wandb.Video(pair, fps=self.fps, format="mp4")
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
