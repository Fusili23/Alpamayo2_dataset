"""Build the generation work queue ordered by value.

Server access can end at any time, so ordering matters. The queue is consumed
from the front, so the most valuable clips go first:

  1. vla_golden (1,181): the curated high quality set used for paper evaluation
  2. remaining valid clips from clip_index, train split first

Shuffling uses a fixed seed so an interrupted run still leaves an unbiased remainder.
"""

import argparse
import os

import pandas as pd


def main() -> None:
    big = os.environ.get("BIG", "/NHNHOME/WORKSPACE/0526050025_A/alpamayo")
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default=f"{big}/alpamayo2/notebooks/clip_ids.parquet")
    p.add_argument("--out", default=f"{big}/data/clip_queue.parquet")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from huggingface_hub import hf_hub_download

    golden = pd.read_parquet(args.golden)["clip_id"].astype(str).tolist()
    print(f"vla_golden: {len(golden)}")

    idx_path = hf_hub_download(
        "nvidia/PhysicalAI-Autonomous-Vehicles", "clip_index.parquet", repo_type="dataset"
    )
    idx = pd.read_parquet(idx_path)
    print(f"clip_index: {idx.shape}, columns={list(idx.columns)}")

    idx = idx.reset_index()
    id_col = next((c for c in idx.columns if "clip" in c.lower() and "id" in c.lower()), None)
    if id_col is None:
        id_col = idx.columns[0]
    idx[id_col] = idx[id_col].astype(str)

    if "clip_is_valid" in idx.columns:
        before = len(idx)
        idx = idx[idx["clip_is_valid"].astype(bool)]
        print(f"clip_is_valid filter: {before} -> {len(idx)}")

    # train split first, the rest after. Preserve order when split is absent
    if "split" in idx.columns:
        print("split distribution:", idx["split"].value_counts().to_dict())
        order = {"train": 0, "val": 1, "validation": 1, "test": 2}
        idx["_o"] = idx["split"].map(lambda s: order.get(str(s).lower(), 3))
    else:
        idx["_o"] = 0

    rest = idx.sample(frac=1.0, random_state=args.seed).sort_values("_o", kind="stable")
    rest_ids = [c for c in rest[id_col].tolist() if c not in set(golden)]

    queue = golden + rest_ids
    out = pd.DataFrame({"clip_id": queue})
    out["priority"] = ["golden"] * len(golden) + ["extended"] * len(rest_ids)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nqueue saved: {args.out}")
    print(f"  golden   {len(golden):>7,}")
    print(f"  extended {len(rest_ids):>7,}")
    print(f"  total     {len(queue):>7,}  (about {len(queue)*26/1000:.0f} GB if fully processed)")


if __name__ == "__main__":
    main()
