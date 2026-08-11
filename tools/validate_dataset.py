"""데이터셋 무결성 검사. 생성과 병행할 수 있도록 CPU만 쓴다.

생성 중에도 돌릴 수 있게 설계했다. 다만 마지막 shard는 아직 쓰이는 중일 수 있으므로
읽기 실패를 곧바로 "깨짐"으로 보지 않고 별도로 표시한다.

검사 항목:
  구조   — 평탄화된 배열 길이가 스키마 주석과 맞는가
  값     — NaN / 빈 문자열 / 음수 토큰 수
  이미지 — JPEG가 실제로 디코드되는가, 장수·해상도가 맞는가 (표본 검사)
  정합성 — samples 와 frames 의 (clip_id, t0_us) 집합이 일치하는가
  중복   — (clip_id, t0_us, sample_idx) 가 유일한가
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

# (컬럼, 기대 길이) — 평탄화 전 형상은 주석 참고
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
    p.add_argument("--jpeg-samples", type=int, default=2, help="shard당 디코드 검사할 프레임 행 수")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    random.seed(args.seed)

    problems: list[str] = []
    warn: list[str] = []

    # ---------- samples ----------
    s_files = sorted(
        f for f in os.listdir(os.path.join(args.data, "samples")) if f.endswith(".parquet")
    )
    log(f"samples shard {len(s_files)}개 검사 시작")
    s_keys: set = set()
    dup = Counter()
    n_rows = 0
    bad_len = bad_nan = bad_topk = bad_coc = 0

    for i, fn in enumerate(s_files):
        path = os.path.join(args.data, "samples", fn)
        try:
            t = pq.read_table(path)
        except Exception as e:
            # 마지막 shard는 쓰이는 중일 수 있다
            (warn if fn == s_files[-1] else problems).append(
                f"samples/{fn} 읽기 실패: {type(e).__name__} {str(e)[:80]}"
            )
            continue
        d = t.to_pandas()
        n_rows += len(d)

        for col, exp in SAMPLE_LENGTHS:
            bad = (d[col].apply(len) != exp).sum()
            if bad:
                bad_len += bad
                problems.append(f"samples/{fn}: {col} 길이 불일치 {bad}행 (기대 {exp})")

        b = (d.topk_ids.apply(len) != d.num_gen_tokens * d.topk_k).sum()
        b += (d.topk_logprobs.apply(len) != d.num_gen_tokens * d.topk_k).sum()
        if b:
            bad_topk += b
            problems.append(f"samples/{fn}: topk 길이 불일치 {b}행")

        nan = int(d.ade.isna().sum() + d.seq_logprob.isna().sum() + d.min_ade_at_t0.isna().sum())
        if nan:
            bad_nan += nan
            problems.append(f"samples/{fn}: NaN {nan}개")

        empty = int((d.coc.fillna("").str.strip() == "").sum())
        if empty:
            bad_coc += empty
            problems.append(f"samples/{fn}: 빈 CoC {empty}행")

        for k in zip(d.clip_id, d.t0_us, d.sample_idx):
            dup[k] += 1
        s_keys.update(zip(d.clip_id, d.t0_us))

        if (i + 1) % 200 == 0:
            log(f"  samples {i+1}/{len(s_files)} ...")

    n_dup = sum(v - 1 for v in dup.values() if v > 1)
    if n_dup:
        problems.append(f"samples: 중복 (clip,t0,sample_idx) {n_dup}건")

    # ---------- frames ----------
    f_files = sorted(
        f for f in os.listdir(os.path.join(args.data, "frames")) if f.endswith(".parquet")
    )
    log(f"frames shard {len(f_files)}개 검사 시작")
    f_keys: set = set()
    bad_cnt = bad_cam = bad_jpeg = 0
    n_frames = 0

    for i, fn in enumerate(f_files):
        path = os.path.join(args.data, "frames", fn)
        try:
            t = pq.read_table(path)
        except Exception as e:
            (warn if fn == f_files[-1] else problems).append(
                f"frames/{fn} 읽기 실패: {type(e).__name__} {str(e)[:80]}"
            )
            continue
        d = t.to_pandas()
        n_frames += len(d)

        exp_imgs = d.num_frames_per_camera * d.camera_indices.apply(len)
        b = int((d.jpegs.apply(len) != exp_imgs).sum())
        if b:
            bad_cnt += b
            problems.append(f"frames/{fn}: JPEG 장수 불일치 {b}행")

        b = int((d.camera_indices.apply(lambda x: list(x) != [0, 1, 2, 3, 4, 5, 6])).sum())
        if b:
            bad_cam += b
            problems.append(f"frames/{fn}: camera_indices 가 canonical 7-ring 이 아님 {b}행")

        # JPEG 실제 디코드 (표본)
        for ridx in random.sample(range(len(d)), min(args.jpeg_samples, len(d))):
            r = d.iloc[ridx]
            for j, blob in enumerate(r.jpegs):
                try:
                    im = Image.open(io.BytesIO(blob))
                    im.load()
                    if (im.width, im.height) != (int(r.width), int(r.height)):
                        bad_jpeg += 1
                        problems.append(
                            f"frames/{fn} 행{ridx} 이미지{j}: 해상도 {im.size} != "
                            f"({r.width},{r.height})"
                        )
                except Exception as e:
                    bad_jpeg += 1
                    problems.append(
                        f"frames/{fn} 행{ridx} 이미지{j}: 디코드 실패 {type(e).__name__}"
                    )

        f_keys.update(zip(d.clip_id, d.t0_us))
        if (i + 1) % 200 == 0:
            log(f"  frames {i+1}/{len(f_files)} ...")

    # ---------- 정합성 ----------
    only_s = s_keys - f_keys
    only_f = f_keys - s_keys
    if only_s:
        problems.append(f"samples 에만 있는 (clip,t0): {len(only_s)}건 — 프레임 없음")
    if only_f:
        warn.append(f"frames 에만 있는 (clip,t0): {len(only_f)}건 — 샘플 없음(생성 중이면 정상)")

    # ---------- 보고 ----------
    print("\n" + "=" * 62)
    print("데이터셋 무결성 검사 결과")
    print("=" * 62)
    print(f"  samples : {n_rows:,}행 / {len(s_files)} shard")
    print(f"  frames  : {n_frames:,}행 / {len(f_files)} shard")
    print(f"  고유 t0 : {len(s_keys):,}")
    print()
    print(f"  배열 길이 불일치 : {bad_len}")
    print(f"  topk 길이 불일치 : {bad_topk}")
    print(f"  NaN              : {bad_nan}")
    print(f"  빈 CoC           : {bad_coc}")
    print(f"  중복 키          : {n_dup}")
    print(f"  JPEG 장수 불일치 : {bad_cnt}")
    print(f"  카메라 구성 오류 : {bad_cam}")
    print(f"  JPEG 디코드 실패 : {bad_jpeg}")
    print()
    if warn:
        print(f"  경고 {len(warn)}건 (생성 중이면 정상):")
        for w in warn[:10]:
            print(f"    - {w}")
    if problems:
        print(f"\n  !! 문제 {len(problems)}건:")
        for pr in problems[:40]:
            print(f"    - {pr}")
        if len(problems) > 40:
            print(f"    ... 외 {len(problems)-40}건")
        sys.exit(1)
    print("\n  문제 없음. 데이터셋 정상.")


if __name__ == "__main__":
    main()
