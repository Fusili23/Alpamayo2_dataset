"""Rewrite shards with duplicate rows removed.

How the duplicates arose: a t0 spacing A/B experiment was run in a separate
directory and merged back. The experiment used 1 second spacing, so it
regenerated the even t0 values the main run (2 second spacing) already had.
Same seed and same inputs, so the contents are identical: pure duplication,
not corruption.

Safe ordering: write new files, verify them, and only then delete the originals.
Nothing is deleted before verification, so a mid-run failure leaves the originals.

Stop the generation workers before running this. Rewriting shards while workers
are running conflicts with the resume scan.
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
    """Read parquet under sub_dir and write deduplicated shards. Originals are untouched."""
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
            log(f"  !! {fn} read failed: {type(e).__name__}, aborting")
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
            log(f"  {os.path.basename(sub_dir)} {i+1}/{len(files)}  duplicates {n_dup}")
    flush()
    return n_in, n_out, n_dup, stage, len(files)


def main() -> None:
    big = os.environ.get("BIG", "/NHNHOME/WORKSPACE/0526050025_A/alpamayo")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=f"{big}/data/coc_34b_v1")
    p.add_argument("--apply", action="store_true", help="replace originals once verification passes")
    args = p.parse_args()

    # refuse while workers run: rewriting shards conflicts with the resume scan.
    # calling pgrep through a subshell makes it match its own command line,
    # so read /proc directly to avoid counting ourselves.
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
        log(f"!! {running} generation workers are running. Run run.sh stop first.")
        sys.exit(1)

    plan = [
        ("samples", ["clip_id", "t0_us", "sample_idx"], 512, "packed"),
        ("frames", ["clip_id", "t0_us"], 42, "packed"),
    ]
    results = []
    for sub, keys, rps, prefix in plan:
        d = os.path.join(args.data, sub)
        log(f"=== {sub} compacting (key: {', '.join(keys)}) ===")
        n_in, n_out, n_dup, stage, n_files = compact(d, keys, rps, prefix)
        log(f"  in {n_in:,} rows / {n_files} shards -> out {n_out:,} rows (removed {n_dup:,} duplicates)")
        results.append((sub, d, stage, n_in, n_out, n_dup))

    # ---- verify: new files are readable and row counts match ----
    log("=== verification ===")
    ok = True
    for sub, d, stage, n_in, n_out, n_dup in results:
        files = sorted(f for f in os.listdir(stage) if f.endswith(".parquet"))
        total = 0
        for f in files:
            try:
                total += pq.read_metadata(os.path.join(stage, f)).num_rows
            except Exception as e:
                log(f"  !! {sub}/_packed/{f} read failed: {type(e).__name__}")
                ok = False
        if total != n_out:
            log(f"  !! {sub}: wrote {total:,} != expected {n_out:,}")
            ok = False
        elif n_in - n_dup != n_out:
            log(f"  !! {sub}: arithmetic mismatch {n_in} - {n_dup} != {n_out}")
            ok = False
        else:
            log(f"  {sub}: {total:,} rows / {len(files)} shards verified")

    if not ok:
        log("Verification failed. Originals untouched. Delete _packed/ and retry.")
        sys.exit(1)

    if not args.apply:
        log("\nRan without --apply, originals were not replaced.")
        log("Review the result and rerun with --apply.")
        return

    # ---- replace: only after verification passes ----
    for sub, d, stage, *_ in results:
        old = [f for f in os.listdir(d) if f.endswith(".parquet")]
        for f in old:
            os.remove(os.path.join(d, f))
        for f in os.listdir(stage):
            shutil.move(os.path.join(stage, f), os.path.join(d, f))
        os.rmdir(stage)
        log(f"  {sub}: removed {len(old)} original shards, replaced with compacted set")
    log("\nCompaction complete.")


if __name__ == "__main__":
    main()
