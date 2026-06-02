# SPDX-License-Identifier: OpenMDW-1.1
"""PerModeLossCallback — logs per-sample flow-matching loss bucketed by training mode.

The DK1 dataset rolls a mode (policy / causal_policy / forward_dynamics /
inverse_dynamics) PER sample, but the default W&B logging only aggregates loss by
modality (vision vs action), not by mode. This callback buckets the model's
per-instance losses by each sample's ``data_batch["mode"]`` and emits, every
``log_freq`` steps:

  train_per_mode/<mode>_<modality>   mean per-sample loss (e.g. causal_policy_action)
  train_per_mode_frac/<mode>         realized share of samples (sanity-checks mode_probs)

Only the meaningful (mode, modality) pairs carry signal — modes that don't predict
a modality contribute ~0 there (FD→action≈0, ID/causal_policy→vision≈0), because the
per-instance loss masks clean tokens.

Requires the framework patch exposing ``flow_matching_loss_action_per_instance`` in
output_batch (vision per-instance is already exposed).

Accumulation is rank-0-local (no collectives → no deadlock risk); over a log window
rank-0's samples are a representative draw. Alignment between the per-instance loss
tensor and the mode list is guarded by a length check — mismatched steps are skipped
rather than logged wrong, and train_per_mode_frac confirms the mapping is correct.
"""
from __future__ import annotations

from collections import defaultdict

import torch
import wandb

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.callback import Callback

# Which modality(ies) each mode actually supervises. The other modality is all-clean
# for that mode → its per-instance loss is a structural 0 (noisy_mask=0), so logging it
# is just noise. Unknown modes fall back to both (safe default). Restricting to these
# pairs leaves only the signal-carrying keys: policy_{vision,action}, causal_policy_action,
# forward_dynamics_vision, inverse_dynamics_action.
_MODE_MODALITIES: dict[str, tuple[str, ...]] = {
    "policy": ("vision", "action"),
    "causal_policy": ("action",),
    "forward_dynamics": ("vision",),
    "inverse_dynamics": ("action",),
}


class PerModeLossCallback(Callback):
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = max(1, int(log_freq))
        self._acc: dict[tuple[str, str], list] = defaultdict(lambda: [0.0, 0])  # (mode,modality)->[sum,count]
        self._mode_count: dict[str, int] = defaultdict(int)
        self._mode_total: int = 0

    def _reset(self) -> None:
        self._acc.clear()
        self._mode_count.clear()
        self._mode_total = 0

    @staticmethod
    def _as_mode_list(modes) -> list[str] | None:
        if modes is None:
            return None
        if isinstance(modes, str):
            return [modes]
        if isinstance(modes, (list, tuple)):
            return [str(m) for m in modes]
        return [str(modes)]

    @torch.no_grad()
    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict,
        output_batch: dict,
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Rank-0-local only — avoids any collective so asymmetric early-return is safe.
        if distributed.is_rank0():
            modes = self._as_mode_list(data_batch.get("mode"))
            if modes is not None:
                n = len(modes)
                for modality in ("vision", "action"):
                    pi = output_batch.get(f"flow_matching_loss_{modality}_per_instance")
                    if pi is None:
                        continue
                    pi = pi.detach().float().flatten()
                    if pi.numel() != n:  # alignment guard — skip rather than log wrong
                        continue
                    for m, v in zip(modes, pi.tolist()):
                        if modality not in _MODE_MODALITIES.get(m, ("vision", "action")):
                            continue  # mode doesn't supervise this modality → structural 0, skip
                        rec = self._acc[(m, modality)]
                        rec[0] += v
                        rec[1] += 1
                for m in modes:
                    self._mode_count[m] += 1
                    self._mode_total += 1

        if iteration % self.log_freq != 0:
            return
        if not distributed.is_rank0() or wandb.run is None:
            self._reset()
            return

        info: dict[str, float] = {}
        for (m, modality), (s, c) in self._acc.items():
            if c > 0:
                info[f"train_per_mode/{m}_{modality}"] = s / c
        if self._mode_total > 0:
            for m, c in self._mode_count.items():
                info[f"train_per_mode_frac/{m}"] = c / self._mode_total
        if info:
            wandb.log(info, step=iteration)
            log.info(
                f"[per-mode] iter {iteration} (rank0, n={self._mode_total}): "
                + ", ".join(f"{k.split('/', 1)[1]}={v:.4f}" for k, v in sorted(info.items()))
            )
        self._reset()
