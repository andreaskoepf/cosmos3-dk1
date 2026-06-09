# SPDX-License-Identifier: OpenMDW-1.1
"""Extract the trained-parameter DELTA from a Cosmos3-Nano DCP training checkpoint.

Reads a fixed list of trainable keys (the 365 gen-attention full-FT + action-I/O
full-FT + gen-MLP LoRA tensors), loads ONLY those from the DCP `model/` dir, and
writes them to a single safetensors file (bf16) for HuggingFace upload.

Usage:
  python extract_delta.py \
      --model   <checkpoint>/model \
      --keys    /tmp/delta_keys.json \
      --out     cosmos3-dk1-cartesian-delta.safetensors
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="DCP model/ dir of the checkpoint")
    ap.add_argument("--keys", required=True, help="JSON list of tensor keys to extract")
    ap.add_argument("--out", required=True, help="output safetensors path")
    args = ap.parse_args()

    want = set(json.load(open(args.keys)))

    reader = FileSystemReader(args.model)
    sdm = reader.read_metadata().state_dict_metadata

    missing = want - set(sdm.keys())
    if missing:
        raise KeyError(f"{len(missing)} requested keys not in checkpoint, e.g. {sorted(missing)[:3]}")

    # Allocate buffers ONLY for the wanted keys, then load just those from DCP.
    sd = {k: torch.empty(tuple(sdm[k].size), dtype=sdm[k].properties.dtype) for k in want}
    dcp.load(sd, storage_reader=reader)

    # Store bf16 (full-FT weights are already bf16; LoRA adapters too).
    out = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items()}
    save_file(out, args.out)
    n = sum(v.numel() for v in out.values())
    print(f"Wrote {len(out)} tensors ({n:,} params) -> {args.out}")


if __name__ == "__main__":
    main()
