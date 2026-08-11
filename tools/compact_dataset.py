"""중복 행을 제거하며 shard를 다시 쓴다.

중복이 생긴 경위: t0 간격 A/B 실험을 별도 디렉토리에서 돌린 뒤 본 출력에 병합했는데,
실험이 1초 간격이라 본 실행(2초 간격)이 이미 만든 짝수 t0를 다시 생성했다.
같은 seed·같은 입력이라 내용은 동일하므로 손상이 아니라 순수 중복이다.

안전 순서: 새 파일을 먼저 쓰고 -> 검증하고 -> 그 다음에 원본을 지운다.
검증 전에는 아무것도 삭제하지 않으므로 중간에 실패해도 원본이 남는다.

생성 워커를 반드시 멈춘 뒤 실행할 것. 돌아가는 중에 shard를 재작성하면
재개 스캔과 충돌한다.
"""

import argparse
import os
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def compact(sub_dir: str, key_cols: list, rows_per_shard: int, out_prefix: str) -> tuple:
    """sub_dir 안의 parquet을 읽어 중복을 뺀 새 shard로 쓴다. 원본은 건드리지 않는다."""
    files = sorted(f for f in os.listdir(sub_dir) if f.endswith(".parquet"))
    stage = os.path.join(sub_dir, "_packed")
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)

    seen: set = set()
    buf: list = []
    schema = None
    n_in = n_out = n_dup = shard = 0

    def flush():
        nonlocal buf, shard
        if not buf:
            return
        t = pa.Table.from_batches(buf) if isinstance(buf[0], pa.RecordBatch) else pa.concat_tables(buf)
        pq.write_table(t, os.path.join(stage, f"{out_prefix}-{shard:05d}.parquet"), compression="zstd")
        buf = []
        shard += 1

    for i, fn in enumerate(files):
        try:
            t = pq.read_table(os.path.join(sub_dir, fn))
        except Exception as e:
            log(f"  !! {fn} 읽기 실패: {type(e).__name__} — 중단")
            raise
        if schema is None:
            schema = t.schema
        n_in += t.num_rows
        cols = [t.column(c).to_pylist() for c in key_cols]
        keep = []
        for r, k in enumerate(zip(*cols)):
            if k in seen:
                n_dup += 1
            else:
                seen.add(k)
                keep.append(r)
        if keep:
            sel = t.take(keep)
            buf.append(sel)
            n_out += sel.num_rows
            if sum(b.num_rows for b in buf) >= rows_per_shard:
                flush()
        if (i + 1) % 200 == 0:
            log(f"  {os.path.basename(sub_dir)} {i+1}/{len(files)}  중복 {n_dup}")
    flush()
    return n_in, n_out, n_dup, stage, len(files)


def main() -> None:
    big = os.environ.get("BIG", "/NHNHOME/WORKSPACE/0526050025_A/alpamayo")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=f"{big}/data/coc_34b_v1")
    p.add_argument("--apply", action="store_true", help="검증 통과 시 원본을 실제로 교체")
    args = p.parse_args()

    # 워커가 돌고 있으면 거부 — shard 재작성과 재개 스캔이 충돌한다.
    # pgrep을 서브셸로 부르면 그 셸의 명령줄에 패턴 문자열이 들어가 자기 자신을 센다.
    # /proc을 직접 읽어 그 함정을 피한다.
    me = os.getpid()
    running = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if "generate_coc_34b.py" in cmd and "/bin/python3" in cmd:
            running += 1
    if running:
        log(f"!! 생성 워커 {running}개가 실행 중입니다. run.sh stop 후 다시 실행하세요.")
        sys.exit(1)

    plan = [
        ("samples", ["clip_id", "t0_us", "sample_idx"], 512, "packed"),
        ("frames", ["clip_id", "t0_us"], 42, "packed"),
    ]
    results = []
    for sub, keys, rps, prefix in plan:
        d = os.path.join(args.data, sub)
        log(f"=== {sub} 압축 시작 (키: {', '.join(keys)}) ===")
        n_in, n_out, n_dup, stage, n_files = compact(d, keys, rps, prefix)
        log(f"  입력 {n_in:,}행 / {n_files} shard -> 출력 {n_out:,}행  (중복 제거 {n_dup:,})")
        results.append((sub, d, stage, n_in, n_out, n_dup))

    # ---- 검증: 새 파일이 읽히고 행 수가 맞는가 ----
    log("=== 검증 ===")
    ok = True
    for sub, d, stage, n_in, n_out, n_dup in results:
        files = sorted(f for f in os.listdir(stage) if f.endswith(".parquet"))
        total = 0
        for f in files:
            try:
                total += pq.read_metadata(os.path.join(stage, f)).num_rows
            except Exception as e:
                log(f"  !! {sub}/_packed/{f} 읽기 실패: {type(e).__name__}")
                ok = False
        if total != n_out:
            log(f"  !! {sub}: 기록 {total:,} != 기대 {n_out:,}")
            ok = False
        elif n_in - n_dup != n_out:
            log(f"  !! {sub}: 산술 불일치 {n_in} - {n_dup} != {n_out}")
            ok = False
        else:
            log(f"  {sub}: {total:,}행 / {len(files)} shard  검증 통과")

    if not ok:
        log("검증 실패 — 원본을 건드리지 않고 종료합니다. _packed/ 를 지우고 다시 시도하세요.")
        sys.exit(1)

    if not args.apply:
        log("\n--apply 없이 실행되어 원본을 교체하지 않았습니다.")
        log("결과를 확인한 뒤 --apply 로 다시 실행하세요.")
        return

    # ---- 교체: 검증을 통과한 뒤에만 ----
    for sub, d, stage, *_ in results:
        old = [f for f in os.listdir(d) if f.endswith(".parquet")]
        for f in old:
            os.remove(os.path.join(d, f))
        for f in os.listdir(stage):
            shutil.move(os.path.join(stage, f), os.path.join(d, f))
        os.rmdir(stage)
        log(f"  {sub}: 원본 {len(old)} shard 제거, 압축본으로 교체 완료")
    log("\n압축 완료.")


if __name__ == "__main__":
    main()
