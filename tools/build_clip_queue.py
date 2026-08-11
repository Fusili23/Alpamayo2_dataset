"""생성 작업 큐를 가치 순으로 만든다.

서버 접근이 언제 끊길지 모르므로 순서가 중요하다. 앞쪽부터 처리되니
가장 값어치 있는 클립을 앞에 둔다:

  1. vla_golden (1,181) — 큐레이트된 고품질 셋, 논문 평가에 쓰인 것
  2. clip_index의 나머지 유효 클립 — split 기준으로 train 먼저

셔플은 고정 seed로 한다. 중간에 끊겨도 남은 부분의 분포가 치우치지 않는다.
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
        print(f"clip_is_valid 필터: {before} -> {len(idx)}")

    # train split을 앞에, 나머지를 뒤에. split이 없으면 순서 유지
    if "split" in idx.columns:
        print("split 분포:", idx["split"].value_counts().to_dict())
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
    print(f"\n큐 저장: {args.out}")
    print(f"  golden   {len(golden):>7,}")
    print(f"  extended {len(rest_ids):>7,}")
    print(f"  합계     {len(queue):>7,}  (약 {len(queue)*26/1000:.0f} GB 전부 처리 시)")


if __name__ == "__main__":
    main()
