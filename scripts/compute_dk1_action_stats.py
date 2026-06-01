# SPDX-License-Identifier: OpenMDW-1.1
"""Compute Cosmos-format action normalization stats for the dk1 14D action.

Reads the `action` column from one or more dk1 LeRobot dataset roots, computes
per-dim quantiles/mean/std/min/max over the pooled actions, and writes a JSON
with the schema Cosmos's `load_action_stats` expects:
    {"q01": [..14..], "q99": [..14..], "mean": [...], "std": [...],
     "min": [...], "max": [...]}

Usage:
    python compute_dk1_action_stats.py --out dk1_action_normalization.json <root1> <root2> ...
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


JOINT_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
GRIPPER_DIMS = [6, 13]


def load_actions(root: str, relative: bool) -> np.ndarray:
    """Pooled per-frame action targets. When `relative`, joint dims are made
    relative to the SAME frame's observation.state[:14] (a per-frame proxy for
    the chunk-start reference — fine for computing q01/q99 ranges); grippers stay
    absolute."""
    files = sorted(glob.glob(str(Path(root) / "data" / "chunk-*" / "file-*.parquet")))
    chunks = []
    for f in files:
        cols = ["action", "observation.state"] if relative else ["action"]
        t = pq.read_table(f, columns=cols)
        a = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
        if relative:
            st = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)[:, :14]
            a = a.copy()
            a[:, JOINT_DIMS] -= st[:, JOINT_DIMS]
        chunks.append(a)
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 14), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="dk1 LeRobot dataset roots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relative", action="store_true",
                    help="joints relative to observation.state[:14]; must match the dataset setting")
    args = ap.parse_args()

    all_actions = []
    for r in args.roots:
        a = load_actions(r, args.relative)
        print(f"  {Path(r).name}: {a.shape[0]:>9d} frames, dim={a.shape[1] if a.size else '?'}")
        if a.size:
            all_actions.append(a)
    actions = np.concatenate(all_actions, axis=0)
    print(f"pooled: {actions.shape[0]} frames x {actions.shape[1]} dims | relative={args.relative}")

    q01 = np.quantile(actions, 0.01, axis=0)
    q99 = np.quantile(actions, 0.99, axis=0)
    # Grippers: force q01=0, q99=1 so the quantile normalizer becomes the exact
    # linear map (0,1) -> (-1,+1) the user wants (2*g - 1).
    for g in GRIPPER_DIMS:
        q01[g], q99[g] = 0.0, 1.0
    stats = {
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mean": actions.mean(axis=0).tolist(),
        "std": actions.std(axis=0).tolist(),
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
    }
    Path(args.out).write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.out}")
    for i, (lo, hi) in enumerate(zip(stats["q01"], stats["q99"])):
        print(f"  dim{i:2d}: q01={lo:+.4f} q99={hi:+.4f}")


if __name__ == "__main__":
    main()
