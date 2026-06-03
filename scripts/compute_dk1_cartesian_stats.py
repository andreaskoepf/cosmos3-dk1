# SPDX-License-Identifier: OpenMDW-1.1
"""Cartesian (EE pose-delta) action normalization stats for the dk1 datamix.

Mirrors the 20D cartesian action the dataset emits in action_space="cartesian":
per arm [pos_delta(3), rot6d_delta(6), gripper(1)], left then right. Pose deltas
are SINGLE-STEP (backward_framewise) of the realized STATE end-effector
trajectory, so — unlike the chunk-relative joint stats — they are independent of
chunk_length: we just pool every within-episode consecutive framewise delta.

The two gripper dims (9, 19) are FORCED to q01=0 / q99=1 so the absolute (0,1)
gripper maps linearly to (-1,1) (0=open→-1, 1=closed→+1), matching the joint path.

Usage:
    python scripts/compute_dk1_cartesian_stats.py \
        --cache-root /workspace/code/fastwam/cache/cartesian \
        --out data/dk1_action_normalization_cartesian.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from dk1_lerobot_dataset import (  # noqa: E402
    _ACTION_FEATURE,
    _CART_STATE_LEFT,
    _CART_STATE_RIGHT,
    DK1LeRobotDataset,
)
from cosmos_framework.data.vfm.action.pose_utils import (  # noqa: E402
    build_abs_pose_from_components,
    pose_abs_to_rel,
)

_GRIPPER_DIMS = (9, 19)  # left_gripper, right_gripper in the 20D layout

# Full 21-source datamix (names only; weights irrelevant for pooled stats).
_DATAMIX = [
    "dk1_black_and_white_swan_2026-05-13", "dk1_black_and_white_swan_2026-05-14",
    "dk1_cutlery_basket_2026-04-21", "dk1_cutlery_basket_2026-04-22",
    "dk1_cutlery_basket_2026-04-23", "dk1_cutlery_basket_2026-04-24",
    "dk1_cutlery_basket_2026-04-25", "dk1_cutlery_basket_2026-04-26_recover1",
    "dk1_duplo_disassembly_2026-04-04", "dk1_duplo_in_box_2026-04-08",
    "dk1_duplo_in_box_2026-04-17", "dk1_duplo_in_box_2026-04-18_artificial_light",
    "dk1_duplo_in_box_2026-04-18_dense_center", "dk1_duplo_in_box_2026-04-18_few_blocks",
    "dk1_duplo_in_box_one_2026-04-09", "dk1_duplo_in_box_two_2026-04-09",
    "dk1_duplo_sorting_by_size_2026-04-05", "dk1_duplo_stack_simple",
    "dk1-merge-2026-03", "robotwin_dk1_realcam_stack_blocks_three_1000",
    "robotwin_dk1_realcam_stack_blocks_two_1000",
]


def deltas_for_dataset(root: str, cache_root: str) -> np.ndarray:
    """All within-episode single-step cartesian deltas for one dataset → [M, 20]."""
    ds = DK1LeRobotDataset(
        root=root, normalization_path="/dev/null", tokenizer_config=None,
        chunk_length=1, action_space="cartesian", cartesian_cache_root=cache_root, split="all",
    )
    # Group rows by episode (rows are index-sorted; episodes are contiguous).
    by_ep: dict[int, list] = {}
    for r in ds._rows:
        by_ep.setdefault(int(r["episode_index"]), []).append(r)

    out = []
    for rows in by_ep.values():
        if len(rows) < 2:
            continue
        cart = np.stack([r["_cart"] for r in rows]).astype(np.float32)  # [E, 28]
        arms = []
        for pos_sl, quat_sl in (_CART_STATE_LEFT, _CART_STATE_RIGHT):
            poses_abs = build_abs_pose_from_components(cart[:, pos_sl], cart[:, quat_sl], "quat_wxyz")
            # pose_abs_to_rel returns E-1 framewise deltas (delta[j] = motion frame j→j+1).
            rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
            arms.append(rel)  # [E-1, 9]
        # The dataset pairs delta[j] with the gripper at frame j → act[:-1].
        act = np.asarray([r[_ACTION_FEATURE] for r in rows], dtype=np.float32)[:-1]  # [E-1, 14]
        lg, rg = act[:, 6:7], act[:, 13:14]
        out.append(np.concatenate([arms[0], lg, arms[1], rg], axis=-1))  # [E-1, 20]
    return np.concatenate(out, axis=0) if out else np.zeros((0, 20), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="/workspace/code/cosmos-dk1/cache/cartesian")
    ap.add_argument("--data-root", default="/workspace/data")
    ap.add_argument("--out", default="data/dk1_action_normalization_cartesian.json")
    args = ap.parse_args()

    pools = []
    for name in _DATAMIX:
        d = deltas_for_dataset(f"{args.data_root}/{name}", args.cache_root)
        pools.append(d)
        print(f"  {name:48s} {d.shape[0]:>9,} deltas")
    alld = np.concatenate(pools, axis=0)
    print(f"pooled deltas: {alld.shape}")

    q01 = np.quantile(alld, 0.01, axis=0)
    q99 = np.quantile(alld, 0.99, axis=0)
    mean, std = alld.mean(0), alld.std(0)
    mn, mx = alld.min(0), alld.max(0)
    # Force grippers to the (0,1) linear map.
    for g in _GRIPPER_DIMS:
        q01[g], q99[g], mn[g], mx[g] = 0.0, 1.0, 0.0, 1.0

    stats = {k: np.asarray(v, np.float32).tolist()
             for k, v in dict(q01=q01, q99=q99, mean=mean, std=std, min=mn, max=mx).items()}
    out = Path(args.out)
    out.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out}")
    np.set_printoptions(precision=4, suppress=True)
    print("q01:", np.array(stats["q01"]))
    print("q99:", np.array(stats["q99"]))


if __name__ == "__main__":
    main()
