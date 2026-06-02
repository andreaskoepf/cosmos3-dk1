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


def load_actions(root: str, relative: bool, chunk_length: int, stride: int) -> np.ndarray:
    """Pooled action targets matching the dataset normalization.

    When `relative`, joints are made CHUNK-START-relative: for each length-`chunk_length`
    window (within one episode, every `stride` frames) the joint dims of all steps are
    offset by the window's first-frame observation.state[:14] — exactly what
    DK1LeRobotDataset does. This makes q01/q99 reflect the true horizon distribution
    (the cumulative drift grows with chunk_length), so longer chunks don't clip.
    Grippers stay absolute (forced to [0,1] in main)."""
    files = sorted(glob.glob(str(Path(root) / "data" / "chunk-*" / "file-*.parquet")))
    if not files:
        return np.empty((0, 14), np.float32)
    cols = ["action", "observation.state", "episode_index", "index"] if relative else ["action", "index"]
    A, S, EP, IDX = [], [], [], []
    for f in files:
        t = pq.read_table(f, columns=cols)
        A.append(np.asarray(t.column("action").to_pylist(), dtype=np.float32)[:, :14])
        IDX.append(np.asarray(t.column("index").to_pylist(), dtype=np.int64))
        if relative:
            S.append(np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)[:, :14])
            EP.append(np.asarray(t.column("episode_index").to_pylist(), dtype=np.int64))
    A = np.concatenate(A); idx = np.concatenate(IDX)
    order = np.argsort(idx, kind="stable"); A = A[order]
    if not relative:
        return A
    S = np.concatenate(S)[order]; EP = np.concatenate(EP)[order]
    n = len(A)
    out = []
    for s in range(0, n - chunk_length + 1, max(1, stride)):
        if EP[s + chunk_length - 1] != EP[s]:  # window must stay within one episode
            continue
        rel = A[s : s + chunk_length].copy()
        rel[:, JOINT_DIMS] -= S[s, JOINT_DIMS]
        out.append(rel)
    return np.concatenate(out, axis=0) if out else np.empty((0, 14), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="dk1 LeRobot dataset roots")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relative", action="store_true",
                    help="joints CHUNK-START-relative; must match the dataset setting")
    ap.add_argument("--chunk-length", type=int, default=32,
                    help="action chunk length the stats are computed for (must match the dataset)")
    ap.add_argument("--stride", type=int, default=0,
                    help="window stride (0 → non-overlapping = chunk_length)")
    args = ap.parse_args()
    stride = args.stride if args.stride > 0 else args.chunk_length
    print(f"relative={args.relative} chunk_length={args.chunk_length} stride={stride}")

    all_actions = []
    for r in args.roots:
        a = load_actions(r, args.relative, args.chunk_length, stride)
        print(f"  {Path(r).name}: {a.shape[0]:>9d} window-steps, dim={a.shape[1] if a.size else '?'}")
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
