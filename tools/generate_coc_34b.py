"""Alpamayo 2 Super (34B) teacher CoC dataset generator.

Lessons carried over from the 10B version:
  Open camera objects once per clip and process many t0 values (30x difference).
  Encode the JPEG first and run inference on the decoded frames. Running
    inference on the originals and storing a JPEG misaligns the stored image
    against the stored CoC. Measured: q92 compression alone flips 1 of 6 CoC
    outputs and moves ADE by 1.14m.
  Clamp the t0 grid to each clip's video length (video is ~20s, egomotion ~140s).
  Store per sample ADE raw. The threshold must remain changeable later.

Specific to the 34B:
  Store all 7 cameras. The trajectory task uses (0,1,2,3,5,6) and vqa uses
    (0,1,2,3,4,5), so the union is 7. Cameras can be dropped at training time
    but never added back. Inference receives only the subset select_task_input
    returns.
  The model is 69GB, so 2 workers per GPU is the ceiling.

Someone else may need the GPUs, so this can be stopped and resumed at any time.
On SIGTERM or SIGINT the shards are flushed before exit, and a restart continues
from the remaining (clip_id, t0_us) keys.
"""

import argparse
import io
import json
import os
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from PIL import Image

NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
TIME_STEP = 0.1
NUM_FRAMES = 4


def parse_args() -> argparse.Namespace:
    big = os.environ.get("BIG", "/NHNHOME/WORKSPACE/0526050025_A/alpamayo")
    p = argparse.ArgumentParser()
    p.add_argument("--clips-file", default=f"{big}/alpamayo2/notebooks/clip_ids.parquet")
    p.add_argument("--out", default=f"{big}/data/coc_34b_v1")
    p.add_argument("--model", default="nvidia/Alpamayo2-Super")
    p.add_argument("--task", default="trajectory")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--t0-start", type=float, default=2.0, help="seconds")
    p.add_argument(
        "--t0-end",
        type=float,
        default=18.0,
        help="seconds. Video usually falls short of 20s, so t0=20 is clamped on 81% of clips. "
        "That leaves an unfinished t0 on completed clips, reopening them on every restart (3 hours per worker).",
    )
    p.add_argument("--t0-step", type=float, default=2.0, help="seconds")
    p.add_argument("--num-traj-samples", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ade-threshold", type=float, default=1.0, help="meters")
    p.add_argument("--shard-rows", type=int, default=512)
    p.add_argument("--flush-seconds", type=float, default=300.0, help="periodic flush interval")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--topk", type=int, default=20, help="store top-k distribution per token (0 disables)")
    p.add_argument("--encode-threads", type=int, default=4, help="JPEG encoding parallelism")
    p.add_argument("--t0-timeout", type=int, default=300, help="per t0 ceiling in seconds, prevents hangs")
    p.add_argument("--clip-open-timeout", type=int, default=600, help="clip open ceiling in seconds")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def smart_size(w: int, h: int, min_px: int, max_px: int, factor: int = 32) -> tuple[int, int]:
    """Same size the Qwen3-VL processor picks: a multiple of factor, within the pixel range."""
    import math

    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_px:
        s = math.sqrt(h * w / max_px)
        hb = max(factor, math.floor(h / s / factor) * factor)
        wb = max(factor, math.floor(w / s / factor) * factor)
    elif hb * wb < min_px:
        s = math.sqrt(min_px / (h * w))
        hb = math.ceil(h * s / factor) * factor
        wb = math.ceil(w * s / factor) * factor
    return wb, hb


def ego_window(egomotion, t0_us: int) -> dict[str, np.ndarray]:
    """History and future trajectory in the local frame at t0."""
    hist_off = np.arange(
        -(NUM_HISTORY_STEPS - 1) * TIME_STEP * 1e6, TIME_STEP * 1e6 / 2, TIME_STEP * 1e6
    ).astype(np.int64)
    fut_off = np.arange(
        TIME_STEP * 1e6, (NUM_FUTURE_STEPS + 0.5) * TIME_STEP * 1e6, TIME_STEP * 1e6
    ).astype(np.int64)

    h = egomotion(t0_us + hist_off)
    f = egomotion(t0_us + fut_off)
    h_xyz, h_quat = h.pose.translation, h.pose.rotation.as_quat()
    f_xyz, f_quat = f.pose.translation, f.pose.rotation.as_quat()

    t0_xyz = h_xyz[-1].copy()
    t0_quat = h_quat[-1].copy()  # scipy convention: xyzw
    r_inv = spt.Rotation.from_quat(t0_quat).inv()
    return {
        "hist_xyz": r_inv.apply(h_xyz - t0_xyz),
        "hist_rot": (r_inv * spt.Rotation.from_quat(h_quat)).as_matrix(),
        "fut_xyz": r_inv.apply(f_xyz - t0_xyz),
        "fut_rot": (r_inv * spt.Rotation.from_quat(f_quat)).as_matrix(),
        "t0_xyz": t0_xyz,
        # the model expects the inverse rotation quaternion in wxyz order
        "t0_inv_quat_wxyz": np.array(
            [t0_quat[3], -t0_quat[0], -t0_quat[1], -t0_quat[2]], dtype=np.float32
        ),
    }


SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("t0_us", pa.int64()),
        ("sample_idx", pa.int32()),
        ("task", pa.string()),
        ("coc", pa.string()),
        ("ade", pa.float32()),
        ("min_ade_at_t0", pa.float32()),
        ("pass_filter", pa.bool_()),
        ("pred_xyz", pa.list_(pa.float32())),  # 64*3
        ("pred_rot", pa.list_(pa.float32())),  # 64*9
        ("gt_xyz", pa.list_(pa.float32())),  # 64*3
        ("gt_rot", pa.list_(pa.float32())),  # 64*9
        ("hist_xyz", pa.list_(pa.float32())),  # 16*3
        ("hist_rot", pa.list_(pa.float32())),  # 16*9
        # Sequence logprob (sum of log P over chosen tokens): a teacher confidence signal.
        # The `logprob` the model returns is an all zero placeholder, so compute it here.
        ("seq_logprob", pa.float32()),
        ("num_gen_tokens", pa.int32()),
        # Top-k distribution for token level KD. The 34B and Qwen3-VL students share
        # an identical text tokenizer (all 151,669 tokens match), so this can be used
        # directly as a KL target. Cost is 16 KB per t0, about 1.2% of the frames.
        ("gen_token_ids", pa.list_(pa.int32())),  # steps
        ("topk_ids", pa.list_(pa.int32())),  # steps*k, row major
        ("topk_logprobs", pa.list_(pa.float32())),  # steps*k
        ("topk_k", pa.int32()),
        ("num_prompt_tokens", pa.int32()),
        ("gen_seconds", pa.float32()),
        ("model", pa.string()),
        ("seed", pa.int32()),
        ("temperature", pa.float32()),
        ("top_p", pa.float32()),
        # cameras actually fed to inference (a subset of the stored 7)
        ("input_camera_ids", pa.list_(pa.int32())),
    ]
)

# The interval where t0 is actually valid for a clip. Only knowable by opening
# the clip, so it is cached. Without this, short clips look like they still have
# pending t0 on every restart and get reopened (40 to 55 seconds) only to be discarded.
RANGE_SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("lo_us", pa.int64()),
        ("hi_us", pa.int64()),
    ]
)

FRAME_SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("t0_us", pa.int64()),
        # always stored in the canonical 7 camera ring order (0..6)
        ("camera_indices", pa.list_(pa.int32())),
        ("num_frames_per_camera", pa.int32()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("jpeg_quality", pa.int32()),
        ("jpegs", pa.list_(pa.binary())),  # 7*4 = 28 images, camera major then frame
        # timing needed for in vehicle reproduction (the model assumes 0.1s spacing)
        ("relative_timestamps", pa.list_(pa.float32())),  # 7*4
        ("absolute_timestamps", pa.list_(pa.int64())),  # 7*4
    ]
)


class ShardWriter:
    """Buffer rows and flush as parquet shards, avoiding many small files on lustre."""

    def __init__(self, out_dir: str, prefix: str, schema, worker_id: int, shard_rows: int):
        self.dir = out_dir
        self.prefix = prefix
        self.schema = schema
        self.worker = worker_id
        self.limit = shard_rows
        self.rows: list[dict] = []
        self.shard = 0
        os.makedirs(out_dir, exist_ok=True)
        while os.path.exists(self._path(self.shard)):
            self.shard += 1

    def _path(self, n: int) -> str:
        return os.path.join(self.dir, f"{self.prefix}-w{self.worker:02d}-{n:05d}.parquet")

    def add(self, row: dict) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.limit:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        tmp = self._path(self.shard) + ".tmp"
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, self._path(self.shard))
        log(f"  shard written: {os.path.basename(self._path(self.shard))} ({len(self.rows)} rows)")
        self.rows = []
        self.shard += 1


def already_done(out_dir: str) -> set:
    """Read (clip_id, t0_us) keys from existing samples shards to support resume."""
    done = set()
    sub = os.path.join(out_dir, "samples")
    if not os.path.isdir(sub):
        return done
    for fn in sorted(os.listdir(sub)):
        if not fn.endswith(".parquet"):
            continue
        try:
            t = pq.read_table(os.path.join(sub, fn), columns=["clip_id", "t0_us"])
            done.update(zip(t.column("clip_id").to_pylist(), t.column("t0_us").to_pylist()))
        except Exception as e:
            log(f"  warning: {fn} read failed ({type(e).__name__}), ignoring")
    return done


def load_ranges(out_dir: str) -> dict:
    """Load the per clip valid t0 interval cache. {clip_id: (lo_us, hi_us)}"""
    ranges: dict = {}
    sub = os.path.join(out_dir, "ranges")
    if not os.path.isdir(sub):
        return ranges
    for fn in sorted(os.listdir(sub)):
        if not fn.endswith(".parquet"):
            continue
        try:
            t = pq.read_table(os.path.join(sub, fn))
            for c, lo, hi in zip(
                t.column("clip_id").to_pylist(),
                t.column("lo_us").to_pylist(),
                t.column("hi_us").to_pylist(),
            ):
                ranges[c] = (lo, hi)
        except Exception as e:
            log(f"  warning: {fn} read failed ({type(e).__name__}), ignoring")
    return ranges


def snapshot_config(out_dir: str, model, args: argparse.Namespace) -> None:
    """Embed the specification needed for in vehicle porting alongside the dataset.

    Trajectory output is tied to the normalization constants of the unicycle
    action space. With different vehicle dynamics the same tokens mean a
    different trajectory, so this must not be separated from the model config.
    """
    path = os.path.join(out_dir, "_dataset_config.json")
    if os.path.exists(path):
        return
    cfg = model.config
    meta = {
        "model": args.model,
        "task": args.task,
        "num_history_steps": NUM_HISTORY_STEPS,
        "num_future_steps": NUM_FUTURE_STEPS,
        "time_step_s": TIME_STEP,
        "num_frames_per_camera": NUM_FRAMES,
        "jpeg_quality": args.jpeg_quality,
        "generation": {
            "num_traj_samples": args.num_traj_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_generation_length": args.max_generation_length,
        },
        "ade_threshold_m": args.ade_threshold,
    }
    for key in (
        "min_pixels",
        "max_pixels",
        "tokens_per_history_traj",
        "tokens_per_future_traj",
        "traj_vocab_size",
        "include_camera_ids",
        "frame_label",
        "token_layout",
    ):
        meta[key] = getattr(cfg, key, None)
    # expert_config may be a dict or a PretrainedConfig object
    expert = getattr(cfg, "expert_config", None)

    def _sub(name: str):
        if expert is None:
            return None
        if isinstance(expert, dict):
            return expert.get(name)
        return getattr(expert, name, None)

    meta["action_space_cfg"] = _sub("action_space_cfg")
    meta["diffusion_cfg"] = _sub("diffusion_cfg")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    log(f"config snapshot saved: {path}")


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import pandas as pd
    import physical_ai_av

    from alpamayo2_super import helper
    from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    # helper.prepare_model_inputs() calls get_processor() on every t0, and inside it
    # AutoProcessor.from_pretrained() reads from disk and rebuilds the processor.
    # Profiling showed tokenize was 34 to 46% of runtime, the largest phase
    # (inference was 14 to 17%). The processor is constant for a fixed
    # (tokenizer, config), so build it once.
    _proc_cache: dict = {}
    _orig_get_processor = helper.get_processor

    def _cached_get_processor(tokenizer, model_config):
        if "p" not in _proc_cache:
            _proc_cache["p"] = _orig_get_processor(tokenizer, model_config)
        return _proc_cache["p"]

    helper.get_processor = _cached_get_processor

    canonical_names = list(CAMERA_NAMES_TO_INDICES)  # order 0..6
    canonical_ids = list(CAMERA_NAMES_TO_INDICES.values())

    clips = pd.read_parquet(args.clips_file)["clip_id"].tolist()
    if args.max_clips:
        clips = clips[: args.max_clips]
    mine = clips[args.worker_id :: args.num_workers]
    log(f"worker {args.worker_id}/{args.num_workers}: {len(mine)} clips assigned, GPU {args.gpu}")

    out_dir = args.out
    done = already_done(out_dir)
    log(f"already generated (clip,t0): {len(done)}, skipping")
    clip_ranges = load_ranges(out_dir)
    log(f"t0 interval cache: {len(clip_ranges)} clips, filtered before opening")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    cam_feats = [getattr(avdi.features.CAMERA, n.upper()) for n in canonical_names]

    model = Alpamayo2Super.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}"
    )
    model.eval()
    log("model loaded (34B)")

    # Capture the per token distribution. `_generate_with_shared_prefill` overwrites
    # generation_config.output_logits = False internally, so setting it outside does nothing.
    # Wrapping the actual generate call is the only clean intervention point.
    # Measured: no time or memory increase (logits are computed anyway and the
    captured: dict = {}
    if args.topk > 0:
        _orig_generate = model.vlm.generate

        def _generate_capturing(*a, **kw):
            gc = kw.get("generation_config")
            if gc is not None:
                gc.output_logits = True
            out = _orig_generate(*a, **kw)
            captured["logits"] = out.logits
            captured["sequences"] = out.sequences
            return out

        model.vlm.generate = _generate_capturing
        log(f"token distribution capture enabled (top-{args.topk})")
    snapshot_config(out_dir, model, args)

    writer = ShardWriter(
        os.path.join(out_dir, "samples"), "coc", SCHEMA, args.worker_id, args.shard_rows
    )
    frame_writer = ShardWriter(
        os.path.join(out_dir, "frames"),
        "frames",
        FRAME_SCHEMA,
        args.worker_id,
        max(1, args.shard_rows // 12),
    )

    # interval cache rows are small and needed first on restart, so flush often
    range_writer = ShardWriter(
        os.path.join(out_dir, "ranges"), "range", RANGE_SCHEMA, args.worker_id, 64
    )

    def _flush_and_exit(signum, _frame):
        log(f"signal {signum} received: flushing shards and exiting (restart resumes)")
        try:
            writer.flush()
            frame_writer.flush()
            range_writer.flush()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _flush_and_exit)
    signal.signal(signal.SIGINT, _flush_and_exit)

    def _on_timeout(_signum, _frame):
        raise TimeoutError(f"t0 processing exceeded {args.t0_timeout}s")

    signal.signal(signal.SIGALRM, _on_timeout)
    pool = ThreadPoolExecutor(max_workers=args.encode_threads)

    t0_grid_s = np.arange(args.t0_start, args.t0_end + 1e-9, args.t0_step)
    tgt_w = tgt_h = None
    n_ok = n_pass = n_fail = 0
    # Cumulative time per phase, to identify why the GPU idles
    phase = {"open": 0.0, "decode": 0.0, "encode": 0.0, "tokenize": 0.0, "infer": 0.0, "post": 0.0}
    t_start = last_flush = time.time()

    def _open_clip(cid: str):
        """Open egomotion and 7 cameras. About 56 seconds of network streaming per clip."""
        e = avdi.get_clip_feature(cid, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
        cs = [avdi.get_clip_feature(cid, c, maybe_stream=True) for c in cam_feats]
        return e, cs

    def _pending_for(cid: str) -> list:
        """Pending t0 values. With an interval cache these are filtered before opening.

        Without a cache (a clip seen for the first time) the interval is unknown
        until the clip is opened, so nothing is filtered.
        """
        p = [int(round(s * 1e6)) for s in t0_grid_s if (cid, int(round(s * 1e6))) not in done]
        rng = clip_ranges.get(cid)
        if rng is not None:
            lo, hi = rng
            p = [t for t in p if lo <= t <= hi]
        return p

    def _has_work(cid: str) -> bool:
        return bool(_pending_for(cid))

    # Clip opening (56s) is 26% of per clip wall clock, and both CPU and GPU idle
    # Prefetch the next clip while the current clip's t0 values run, hiding that window.
    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    prefetch: dict[str, object] = {}

    def _schedule_prefetch(from_idx: int) -> None:
        for nxt in mine[from_idx:]:
            if _has_work(nxt):
                if prefetch.get("clip_id") != nxt:
                    prefetch["clip_id"] = nxt
                    prefetch["future"] = prefetch_pool.submit(_open_clip, nxt)
                return

    for ci, clip_id in enumerate(mine):
        pending = _pending_for(clip_id)
        if not pending:
            continue  # the cache lets us skip without opening the clip
        try:
            signal.alarm(args.clip_open_timeout)
            _t_open = time.time()
            if prefetch.get("clip_id") == clip_id and "future" in prefetch:
                ego, cams = prefetch.pop("future").result()
                prefetch.pop("clip_id", None)
            else:
                ego, cams = _open_clip(clip_id)
            phase["open"] += time.time() - _t_open
            signal.alarm(0)
            _schedule_prefetch(ci + 1)  # prefetch the next clip
        except Exception as e:
            signal.alarm(0)
            prefetch.pop("future", None)
            prefetch.pop("clip_id", None)
            log(f"[{ci+1}/{len(mine)}] {clip_id[:8]} clip open failed: {type(e).__name__} {str(e)[:90]}")
            continue

        # Video length varies per clip and is far shorter than egomotion (~20s vs ~140s)
        try:
            lo = max(int(c.timestamps.min()) for c in cams) + (NUM_FRAMES - 1) * int(
                TIME_STEP * 1e6
            )
            lo = max(lo, int(NUM_HISTORY_STEPS * TIME_STEP * 1e6) + 1)
            hi = min(
                min(int(c.timestamps.max()) for c in cams),
                int(ego.timestamps.max()) - int((NUM_FUTURE_STEPS + 1) * TIME_STEP * 1e6),
            )
            # record the interval so later runs can filter without opening the clip
            if clip_id not in clip_ranges:
                clip_ranges[clip_id] = (lo, hi)
                range_writer.add({"clip_id": clip_id, "lo_us": lo, "hi_us": hi})
            kept = [t for t in pending if lo <= t <= hi]
            if len(kept) != len(pending):
                log(
                    f"  {clip_id[:8]} t0 clamp [{lo/1e6:.1f}, {hi/1e6:.1f}]s: "
                    f"{len(pending)} -> {len(kept)}"
                )
            pending = kept
            if not pending:
                continue
        except Exception as e:
            log(f"  {clip_id[:8]} interval computation failed ({type(e).__name__}), using the original grid")

        n_clip = 0
        for t0_us in pending:
            try:
                # Prevent a worker from hanging forever on stalled network streaming.
                # A healthy t0 takes about 10 seconds, so a generous ceiling only catches faults.
                signal.alarm(args.t0_timeout)
                _t = time.time()
                img_ts = np.array(
                    [t0_us - (NUM_FRAMES - 1 - i) * int(TIME_STEP * 1e6) for i in range(NUM_FRAMES)],
                    dtype=np.int64,
                )
                frames, abs_ts = [], []
                for obj in cams:
                    img, ft = obj.decode_images_from_timestamps(img_ts)
                    frames.append(rearrange(torch.from_numpy(img), "t h w c -> t c h w"))
                    abs_ts.append(torch.from_numpy(ft.astype(np.int64)))
                phase["decode"] += time.time() - _t
                image_frames = torch.stack(frames, 0)  # (7, 4, 3, H, W)
                absolute_timestamps = torch.stack(abs_ts, 0)  # (7, 4)
                relative_timestamps = (
                    absolute_timestamps - absolute_timestamps.min()
                ).float() * 1e-6

                # Build the JPEG first, then run inference on the decoded frames.
                # Reversing the order misaligns the stored image against the stored CoC.
                if tgt_w is None:
                    tgt_w, tgt_h = smart_size(
                        int(image_frames.shape[-1]),
                        int(image_frames.shape[-2]),
                        model.config.min_pixels,
                        model.config.max_pixels,
                    )
                    log(
                        f"  stored frame size: {image_frames.shape[-1]}x{image_frames.shape[-2]}"
                        f" -> {tgt_w}x{tgt_h} ({tgt_w*tgt_h} px, q{args.jpeg_quality}), "
                        f"{len(cams)} cameras"
                    )
                # Encoding 28 images takes longer than GPU inference (2.8s). PIL releases
                # the GIL while encoding, so threading it is a direct win.
                def _encode(img):
                    pil = Image.fromarray(img.permute(1, 2, 0).numpy())
                    if (pil.width, pil.height) != (tgt_w, tgt_h):
                        pil = pil.resize((tgt_w, tgt_h), Image.BICUBIC)
                    buf = io.BytesIO()
                    pil.save(buf, "JPEG", quality=args.jpeg_quality)
                    raw = buf.getvalue()
                    back = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
                    return raw, torch.from_numpy(back).permute(2, 0, 1)

                _t = time.time()
                results = list(pool.map(_encode, list(image_frames.flatten(0, 1))))
                jpegs = [r[0] for r in results]
                image_frames = torch.stack([r[1] for r in results], 0).reshape(
                    len(cams), NUM_FRAMES, 3, tgt_h, tgt_w
                )
                phase["encode"] += time.time() - _t

                traj = ego_window(ego, t0_us)
                camera_tmin = int(absolute_timestamps.min().item())
                # Reproduce the load_physical_aiavdataset() return contract exactly.
                # We avoid that function to open cameras only once per clip, so a single
                # missing key is caught by select_task_input.
                source_data = {
                    "image_frames": image_frames,
                    "camera_indices": torch.tensor(canonical_ids, dtype=torch.int64),
                    "camera_names": canonical_names,
                    "relative_timestamps": relative_timestamps,
                    "absolute_timestamps": absolute_timestamps,
                    "camera_tmin": camera_tmin,
                    "ego_available": torch.tensor(True),
                    "ego_t0": torch.tensor([t0_us], dtype=torch.int64),
                    "ego_t0_relative": torch.tensor(
                        [(t0_us - camera_tmin) * 1e-6], dtype=torch.float32
                    ),
                    "ego_t0_frame_idx": torch.tensor([NUM_FRAMES - 1], dtype=torch.int64),
                    "prediction_start_offset": torch.zeros(1, dtype=torch.float32),
                    "ego_history_tvals": torch.arange(
                        -NUM_HISTORY_STEPS + 1, 1, dtype=torch.float32
                    )
                    * TIME_STEP,
                    "ego_future_tvals": torch.arange(1, NUM_FUTURE_STEPS + 1, dtype=torch.float32)
                    * TIME_STEP,
                    "ego_history_xyz": torch.from_numpy(traj["hist_xyz"]).float()[None, None],
                    "ego_history_rot": torch.from_numpy(traj["hist_rot"]).float()[None, None],
                    "ego_future_xyz": torch.from_numpy(traj["fut_xyz"]).float()[None, None],
                    "ego_future_rot": torch.from_numpy(traj["fut_rot"]).float()[None, None],
                    "ego_t0_xyz": torch.from_numpy(traj["t0_xyz"]).float().view(1, 1, 3),
                    "ego_t0_inv_quat": torch.from_numpy(traj["t0_inv_quat_wxyz"])
                    .float()
                    .view(1, 1, 4),
                    "t0_us": t0_us,
                    "clip_id": clip_id,
                }
                # store all 7, feed inference only the subset the task requires
                data = select_task_input(source_data, args.task)
                used_ids = [int(x) for x in data["camera_indices"].tolist()]

                _t = time.time()
                model_inputs = helper.prepare_model_inputs(data, model.config, model.tokenizer)
                n_tok = int(model_inputs["tokenized_data"]["input_ids"].shape[-1])
                model_inputs = helper.to_device(model_inputs, f"cuda:{args.gpu}")
                phase["tokenize"] += time.time() - _t

                torch.cuda.manual_seed_all(args.seed)
                t_gen = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, logprob, extra = model.sample_trajectories_from_data(
                        data=model_inputs,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        num_traj_samples=args.num_traj_samples,
                        max_generation_length=args.max_generation_length,
                        return_extra=True,
                    )
                gen_s = time.time() - t_gen
                phase["infer"] += gen_s
                _t = time.time()

                gt_xyz = traj["fut_xyz"].astype(np.float32)
                gt_rot = traj["fut_rot"].astype(np.float32)
                pxyz = pred_xyz.float().cpu().numpy().reshape(-1, NUM_FUTURE_STEPS, 3)
                prot = pred_rot.float().cpu().numpy().reshape(-1, NUM_FUTURE_STEPS, 3, 3)
                ade = np.linalg.norm(pxyz[:, :, :2] - gt_xyz[None, :, :2], axis=-1).mean(-1)

                # Extract the per token top-k distribution and the real sequence logprob.
                # The logprob the model returns is zeros_like(...) and unusable.
                topk_ids = topk_lps = gen_ids = None
                seq_lp = None
                if captured.get("logits"):
                    # Stacking all steps allocates (B, steps, 155776) fp32 at once, which
                    # is 957MB near the 256 step ceiling. With the GPU already 98% full
                    # the allocator thrashes. Per step processing pins the peak at
                    # (B, 155776) = 3.7MB. topk is invariant to normalization, so it can
                    # Calling .cpu() per step creates steps x 3 synchronizing transfers,
                    # each waiting on the GPU queue (126 for 42 steps, measured +7s per t0).
                    # The topk results are tiny at (B, k), so accumulate them on the GPU
                    # and transfer once at the end. Peak memory stays as low as per step.
                    steps = len(captured["logits"])
                    chosen = captured["sequences"][:, -steps:]  # (B, steps)
                    tk_v, tk_i, ch_lp = [], [], []
                    for s_i, step_logits in enumerate(captured["logits"]):
                        lg = step_logits.float()  # (B, vocab)
                        lse = torch.logsumexp(lg, dim=-1, keepdim=True)  # (B, 1)
                        v, i = lg.topk(args.topk, dim=-1)
                        tk_v.append(v - lse)
                        tk_i.append(i.to(torch.int32))
                        ch_lp.append(lg.gather(-1, chosen[:, s_i : s_i + 1]) - lse)
                        del lg, lse, v, i
                    topk_lps = torch.stack(tk_v, 1).cpu().numpy()  # (B, steps, k)
                    topk_ids = torch.stack(tk_i, 1).cpu().numpy()
                    seq_lp = torch.cat(ch_lp, 1).sum(-1).cpu().numpy()
                    gen_ids = chosen.to(torch.int32).cpu().numpy()
                    del tk_v, tk_i, ch_lp
                    captured.clear()
                min_ade = float(ade.min())
                cots = np.asarray(extra["cot"]).reshape(-1).tolist()

                phase["post"] += time.time() - _t
                for si in range(len(cots)):
                    ok = bool(ade[si] <= args.ade_threshold)
                    n_pass += int(ok)
                    writer.add(
                        {
                            "clip_id": clip_id,
                            "t0_us": t0_us,
                            "sample_idx": si,
                            "task": args.task,
                            "coc": str(cots[si]),
                            "ade": float(ade[si]),
                            "min_ade_at_t0": min_ade,
                            "pass_filter": ok,
                            "pred_xyz": pxyz[si].reshape(-1).tolist(),
                            "pred_rot": prot[si].reshape(-1).tolist(),
                            "gt_xyz": gt_xyz.reshape(-1).tolist(),
                            "gt_rot": gt_rot.reshape(-1).tolist(),
                            "hist_xyz": traj["hist_xyz"].astype(np.float32).reshape(-1).tolist(),
                            "hist_rot": traj["hist_rot"].astype(np.float32).reshape(-1).tolist(),
                            "seq_logprob": (
                                float(seq_lp[si]) if seq_lp is not None else float("nan")
                            ),
                            "num_gen_tokens": (
                                int(gen_ids.shape[1]) if gen_ids is not None else 0
                            ),
                            "gen_token_ids": (
                                gen_ids[si].tolist() if gen_ids is not None else []
                            ),
                            "topk_ids": (
                                topk_ids[si].reshape(-1).tolist() if topk_ids is not None else []
                            ),
                            "topk_logprobs": (
                                topk_lps[si].reshape(-1).tolist() if topk_lps is not None else []
                            ),
                            "topk_k": args.topk,
                            "num_prompt_tokens": n_tok,
                            "gen_seconds": gen_s,
                            "model": args.model,
                            "seed": args.seed,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "input_camera_ids": used_ids,
                        }
                    )

                frame_writer.add(
                    {
                        "clip_id": clip_id,
                        "t0_us": t0_us,
                        "camera_indices": canonical_ids,
                        "num_frames_per_camera": NUM_FRAMES,
                        "width": tgt_w,
                        "height": tgt_h,
                        "jpeg_quality": args.jpeg_quality,
                        "jpegs": jpegs,
                        "relative_timestamps": relative_timestamps.reshape(-1).tolist(),
                        "absolute_timestamps": absolute_timestamps.reshape(-1).tolist(),
                    }
                )
                n_ok += 1
                n_clip += 1

                # periodic flush bounds the work lost to an interruption
                if time.time() - last_flush > args.flush_seconds:
                    writer.flush()
                    frame_writer.flush()
                    range_writer.flush()
                    last_flush = time.time()
            except Exception as e:
                n_fail += 1
                if n_fail <= 5 or n_fail % 50 == 0:
                    log(
                        f"  {clip_id[:8]} t0={t0_us/1e6:.1f}s failed: "
                        f"{type(e).__name__} {str(e)[:110]}"
                    )
            finally:
                signal.alarm(0)

        el = time.time() - t_start
        tot = sum(phase.values()) or 1.0
        log("  phase%: " + " ".join(f"{k}={100*v/tot:.0f}" for k, v in phase.items())
            + f"  (sum {tot:.0f}s / elapsed {el:.0f}s)")
        log(
            f"[{ci+1}/{len(mine)}] {clip_id[:8]} +{n_clip}t0  "
            f"cum t0={n_ok} passed={n_pass} failed={n_fail}  "
            f"{n_ok/el*3600 if el else 0:.0f} t0/h"
        )

    writer.flush()
    frame_writer.flush()
    range_writer.flush()
    prefetch_pool.shutdown(wait=False, cancel_futures=True)
    pool.shutdown(wait=False)
    with open(os.path.join(out_dir, f"_meta-w{args.worker_id:02d}.json"), "w") as f:
        json.dump(
            {
                "worker_id": args.worker_id,
                "num_workers": args.num_workers,
                "args": vars(args),
                "t0_done": n_ok,
                "samples_passed": n_pass,
                "failures": n_fail,
                "elapsed_s": time.time() - t_start,
            },
            f,
            indent=2,
        )
    log(f"done: t0={n_ok} passed={n_pass} failed={n_fail}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
