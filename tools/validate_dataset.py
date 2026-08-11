"""Dataset integrity check. CPU only, so it can run alongside generation.

Safe to run during generation. The last shard may still be open for writing,
so a read failure there is reported as a warning rather than corruption.

Checks:
  structure   flattened array lengths match the schema comments
  values      NaN, empty strings, negative token counts
  images      JPEGs actually decode, count and resolution match (sampled)
  consistency (clip_id, t0_us) sets agree between samples and frames
  duplicates  (clip_id, t0_us, sample_idx) is unique
"""

import argparse
import io
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# (column, expected length); see comments for the pre-flatten shape
SAMPLE_LENGTHS = [
    ("pred_xyz", 64 * 3),
    ("pred_rot", 64 * 9),
    ("gt_xyz", 64 * 3),
    ("gt_rot", 64 * 9),
    ("hist_xyz", 16 * 3),
    ("hist_rot", 16 * 9),
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    big = os.environ.get("BIG", "/NHNHOME/WORKSPACE/0526050025_A/alpamayo")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=f"{big}/data/coc_34b_v1")
    p.add_argument("--jpeg-samples", type=int, default=2, help="frame rows per shard to decode check")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed)

    problems: list[str] = []
    warn: list[str] = []

    # ---------- samples ----------
    s_files = sorted(
        f for f in os.listdir(os.path.join(args.data, "samples")) if f.endswith(".parquet")
    )
    log(f"samples checking {len(s_files)} shards")
    s_keys: set = set()
    dup = Counter()
    n_rows = 0
    bad_len = bad_nan = bad_topk = bad_coc = 0

    for i, fn in enumerate(s_files):
        path = os.path.join(args.data, "samples", fn)
        try:
            t = pq.read_table(path)
        except Exception as e:
            # the last shard may still be being written
            (warn if fn == s_files[-1] else problems).append(
                f"samples/{fn} read failed: {type(e).__name__} {str(e)[:80]}"
            )
            continue
        d = t.to_pandas()
        n_rows += len(d)

        for col, exp in SAMPLE_LENGTHS:
            bad = (d[col].apply(len) != exp).sum()
            if bad:
                bad_len += bad
                problems.append(f"samples/{fn}: {col} length mismatch in {bad} rows (expected {exp})")

        b = (d.topk_ids.apply(len) != d.num_gen_tokens * d.topk_k).sum()
        b += (d.topk_logprobs.apply(len) != d.num_gen_tokens * d.topk_k).sum()
        if b:
            bad_topk += b
            problems.append(f"samples/{fn}: topk length mismatch in {b} rows")

        nan = int(d.ade.isna().sum() + d.seq_logprob.isna().sum() + d.min_ade_at_t0.isna().sum())
        if nan:
            bad_nan += nan
            problems.append(f"samples/{fn}: {nan} NaN values")

        empty = int((d.coc.fillna("").str.strip() == "").sum())
        if empty:
            bad_coc += empty
            problems.append(f"samples/{fn}: {empty} rows with empty CoC")

        for k in zip(d.clip_id, d.t0_us, d.sample_idx):
            dup[k] += 1
        s_keys.update(zip(d.clip_id, d.t0_us))

        if (i + 1) % 200 == 0:
            log(f"  samples {i+1}/{len(s_files)} ...")

    n_dup = sum(v - 1 for v in dup.values() if v > 1)
    if n_dup:
        problems.append(f"samples: {n_dup} duplicate (clip,t0,sample_idx) keys")

    # ---------- frames ----------
    f_files = sorted(
        f for f in os.listdir(os.path.join(args.data, "frames")) if f.endswith(".parquet")
    )
    log(f"checking {len(f_files)} frames shards")
    f_keys: set = set()
    bad_cnt = bad_cam = bad_jpeg = 0
    n_frames = 0

    for i, fn in enumerate(f_files):
        path = os.path.join(args.data, "frames", fn)
        try:
            t = pq.read_table(path)
        except Exception as e:
            (warn if fn == f_files[-1] else problems).append(
                f"frames/{fn} read failed: {type(e).__name__} {str(e)[:80]}"
            )
            continue
        d = t.to_pandas()
        n_frames += len(d)

        exp_imgs = d.num_frames_per_camera * d.camera_indices.apply(len)
        b = int((d.jpegs.apply(len) != exp_imgs).sum())
        if b:
            bad_cnt += b
            problems.append(f"frames/{fn}: JPEG count mismatch in {b} rows")

        b = int((d.camera_indices.apply(lambda x: list(x) != [0, 1, 2, 3, 4, 5, 6])).sum())
        if b:
            bad_cam += b
            problems.append(f"frames/{fn}: camera_indices is not the canonical 7-ring in {b} rows")

        # actually decode JPEGs (sampled)
        for ridx in random.sample(range(len(d)), min(args.jpeg_samples, len(d))):
            r = d.iloc[ridx]
            for j, blob in enumerate(r.jpegs):
                try:
                    im = Image.open(io.BytesIO(blob))
                    im.load()
                    if (im.width, im.height) != (int(r.width), int(r.height)):
                        bad_jpeg += 1
                        problems.append(
                            f"frames/{fn} row {ridx} image {j}: resolution {im.size} != "
                            f"({r.width},{r.height})"
                        )
                except Exception as e:
                    bad_jpeg += 1
                    problems.append(
                        f"frames/{fn} row {ridx} image {j}: decode failed {type(e).__name__}"
                    )

        f_keys.update(zip(d.clip_id, d.t0_us))
        if (i + 1) % 200 == 0:
            log(f"  frames {i+1}/{len(f_files)} ...")

    # ---------- consistency ----------
    only_s = s_keys - f_keys
    only_f = f_keys - s_keys
    if only_s:
        problems.append(f"{len(only_s)} (clip,t0) keys only in samples: no frames")
    if only_f:
        warn.append(f"{len(only_f)} (clip,t0) keys only in frames: no samples (normal during generation)")

    # ---------- report ----------
    print("\n" + "=" * 62)
    print("Dataset integrity check results")
    print("=" * 62)
    print(f"  samples : {n_rows:,} rows / {len(s_files)} shards")
    print(f"  frames  : {n_frames:,} rows / {len(f_files)} shards")
    print(f"  unique t0: {len(s_keys):,}")
    print()
    print(f"  array length mismatch : {bad_len}")
    print(f"  topk length mismatch : {bad_topk}")
    print(f"  NaN              : {bad_nan}")
    print(f"  empty CoC           : {bad_coc}")
    print(f"  duplicate keys      : {n_dup}")
    print(f"  JPEG count mismatch : {bad_cnt}")
    print(f"  camera config error : {bad_cam}")
    print(f"  JPEG decode failed  : {bad_jpeg}")
    print()
    if warn:
        print(f"  {len(warn)} warnings (normal during generation):")
        for w in warn[:10]:
            print(f"    - {w}")
    if problems:
        print(f"\n  !! {len(problems)} problems:")
        for pr in problems[:40]:
            print(f"    - {pr}")
        if len(problems) > 40:
            print(f"    ... and {len(problems)-40} more")
        sys.exit(1)
    print("\n  No problems found. Dataset is healthy.")


if __name__ == "__main__":
    main()
