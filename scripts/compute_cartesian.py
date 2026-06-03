# SPDX-License-Identifier: Apache-2.0
# Vendored from github.com/open-thought/fastwam (scripts/compute_cartesian.py), Apache-2.0.
# Changes: import from local cartesian_fk; pandas→pyarrow; URDF default → cosmos-dk1/urdf.
"""Per-dataset cartesian (end-effector) cache generator.

For one LeRobot dataset, reads every parquet shard, runs FK on the joint
positions of both states and actions, and writes a sidecar parquet aligned
1:1 with the dataset rows:

    cache/cartesian/<dataset_basename>/
        cartesian.parquet   # columns: state_*, action_* (pos[3] + quat[4] per arm)
        manifest.json       # URDF sha256, FK config, dataset row count

Convention:
    quaternions stored (w, x, y, z), 32-bit float
    EE link = {left,right}_tool0  (no gripper finger displacement)

Usage:
    python scripts/compute_cartesian.py /workspace/data/dk1_cutlery_basket_2026-04-21
    python scripts/compute_cartesian.py --urdf urdf/dk1_dual_arm.urdf <dataset> [<dataset>...]
    python scripts/compute_cartesian.py --all   # all datasets currently on disk
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from cartesian_fk import DK1ArmFK, verify_action_layout

CACHE_ROOT = Path(__file__).resolve().parents[1] / "cache" / "cartesian"
_URDF_DEFAULT = Path(__file__).resolve().parents[1] / "urdf" / "dk1_dual_arm.urdf"


def _load_dataset_action_names(dataset_path: Path) -> list[str]:
    info = json.loads((dataset_path / "meta" / "info.json").read_text())
    return list(info["features"]["action"]["names"])


def _list_parquet_shards(dataset_path: Path) -> list[Path]:
    data_dir = dataset_path / "data"
    shards = sorted(data_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards under {data_dir}")
    return shards


def compute_for_dataset(
    dataset_path: Path,
    urdf_path: Path,
    *,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 50000,
) -> Path:
    """Compute cartesian cache for one dataset. Returns path to written manifest."""
    dataset_path = dataset_path.resolve()
    print(f"\n=== {dataset_path.name} ===")

    action_names = _load_dataset_action_names(dataset_path)
    verify_action_layout(action_names)
    fk = DK1ArmFK(urdf_path=urdf_path, device=device, dtype=torch.float32)
    print(f"  URDF sha256: {fk.urdf_sha256[:16]}...  ({urdf_path})")

    out_dir = CACHE_ROOT / dataset_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "cartesian.parquet"
    manifest_path = out_dir / "manifest.json"

    shards = _list_parquet_shards(dataset_path)
    print(f"  {len(shards)} parquet shards")

    # We append rows from all shards in order, then concat at the end. Each row
    # corresponds 1:1 with the source dataset row (so downstream can mmap by
    # global index without per-shard logic).
    all_rows: list[np.ndarray] = []
    n_total = 0
    t0 = time.time()
    for shard in shards:
        tbl = pq.read_table(shard, columns=["action", "observation.state"])
        n = tbl.num_rows
        # Flatten the list<float> columns to dense [n, dim] (fast — avoids to_pylist).
        actions = (tbl.column("action").combine_chunks().flatten()
                   .to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1))      # [n, 14]
        states_full = (tbl.column("observation.state").combine_chunks().flatten()
                       .to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, -1))  # [n, 40]
        # Position block of state is the first 14 dims (verified via action.names).
        states = states_full[:, :14]
        n_total += n

        # Process in chunks to bound GPU memory.
        out_chunks = []
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            qs = torch.from_numpy(states[s:e]).float()
            qa = torch.from_numpy(actions[s:e]).float()
            r_state = fk.forward(qs)
            r_action = fk.forward(qa)
            # Pack: [L_pos(3) | L_quat(4) | R_pos(3) | R_quat(4)] for state and action → 28 dims total
            packed = torch.cat([
                r_state.left_pos, r_state.left_quat,
                r_state.right_pos, r_state.right_quat,
                r_action.left_pos, r_action.left_quat,
                r_action.right_pos, r_action.right_quat,
            ], dim=-1).cpu().numpy()
            out_chunks.append(packed)
        all_rows.append(np.concatenate(out_chunks, axis=0))

    arr = np.concatenate(all_rows, axis=0)
    dt = time.time() - t0
    print(f"  FK on {n_total:,} frames in {dt:.1f}s ({n_total/max(dt,1e-9):,.0f}/s) → arr {arr.shape} {arr.dtype}")

    # Write a single parquet with float32 list columns (one per arm/field).
    arr = arr.astype(np.float32, copy=False)
    col = lambda start, end: list(arr[:, start:end])
    out_tbl = pa.table({
        "state_ee_left_pos_xyz":    col(0, 3),
        "state_ee_left_quat_wxyz":  col(3, 7),
        "state_ee_right_pos_xyz":   col(7, 10),
        "state_ee_right_quat_wxyz": col(10, 14),
        "action_ee_left_pos_xyz":   col(14, 17),
        "action_ee_left_quat_wxyz": col(17, 21),
        "action_ee_right_pos_xyz":  col(21, 24),
        "action_ee_right_quat_wxyz": col(24, 28),
    })
    pq.write_table(out_tbl, out_parquet)
    print(f"  wrote {out_parquet}  ({out_parquet.stat().st_size / 1e6:.1f} MB)")

    manifest = {
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.name,
        "n_rows": int(n_total),
        "urdf_path": str(urdf_path.resolve()),
        "urdf_sha256": fk.urdf_sha256,
        "left_ee_link": fk.left_ee_link,
        "right_ee_link": fk.right_ee_link,
        "quaternion_convention": "wxyz",
        "columns": out_tbl.column_names,
        "format_version": 1,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="Dataset paths (under /workspace/data/dk1_*)")
    parser.add_argument("--urdf", default=str(_URDF_DEFAULT),
                        help="Path to the dk1_dual_arm URDF")
    parser.add_argument("--all", action="store_true",
                        help="Process every /workspace/data/dk1_* dataset (no robotwin/duplo_disassembly/etc.)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    urdf_path = Path(args.urdf).resolve()
    if not urdf_path.exists():
        sys.exit(f"URDF not found: {urdf_path}")

    targets: list[Path] = []
    if args.all:
        for p in sorted(Path("/workspace/data").glob("dk1_*")):
            if (p / "meta" / "info.json").exists() and (p / "data").is_dir():
                targets.append(p)
    targets.extend(Path(p) for p in args.datasets)
    targets = list(dict.fromkeys(targets))  # dedupe, preserve order
    if not targets:
        sys.exit("No datasets specified. Pass paths or --all.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, urdf={urdf_path}")
    print(f"targets ({len(targets)}):")
    for t in targets:
        print(f"  {t}")

    for ds in targets:
        compute_for_dataset(ds, urdf_path, device=device)

    print("\nDone.")


if __name__ == "__main__":
    main()
