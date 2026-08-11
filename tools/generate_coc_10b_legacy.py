"""Alpamayo 1.5 (10B) teacher CoC dataset generator. Kept for reference.

Opens cameras once per clip and processes many t0 values (30 seconds to open a
camera versus 1.4 seconds to decode a t0, so reuse is a 30x win). For each sample
it records the CoC text and trajectory the teacher produced along with ADE
against ground truth.

Filtering policy: each of the num_traj_samples outputs carries its own CoC text,
so the unit of rejection is a single sample, not a t0. Each sample's ADE is
stored raw with a `pass_filter` boolean alongside, so the threshold can change
later without revisiting the source.

This is the 10B version, superseded by generate_coc_34b.py. It uses 4 cameras,
which is incompatible with the 34B six camera configuration.
"""

import argparse
import io
import json
import os
import signal
import time
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from PIL import Image

CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}

NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
TIME_STEP = 0.1
NUM_FRAMES = 4


def parse_args() -> argparse.Namespace:
    big = os.environ.get("BIG", "/NHNHOME/WORKSPACE/0526050025_A/alpamayo")
    p = argparse.ArgumentParser()
    p.add_argument("--clips-file", default=f"{big}/alpamayo1.5/notebooks/clip_ids.parquet")
    p.add_argument("--out", default=f"{big}/data/coc_v1")
    p.add_argument("--model", default="nvidia/Alpamayo-1.5-10B")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--t0-start", type=float, default=2.0, help="seconds")
    p.add_argument("--t0-end", type=float, default=19.5, help="seconds")
    p.add_argument("--t0-step", type=float, default=1.0, help="seconds")
    p.add_argument("--num-traj-samples", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ade-threshold", type=float, default=1.0, help="meters")
    p.add_argument("--shard-rows", type=int, default=512)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument(
        "--no-frames",
        action="store_true",
        help="skip storing images (the dataset would then depend on re-streaming from HF)",
    )
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ego_window(egomotion, t0_us: int):
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
    r_inv = spt.Rotation.from_quat(h_quat[-1].copy()).inv()

    return {
        "hist_xyz": r_inv.apply(h_xyz - t0_xyz),
        "hist_rot": (r_inv * spt.Rotation.from_quat(h_quat)).as_matrix(),
        "fut_xyz": r_inv.apply(f_xyz - t0_xyz),
        "fut_rot": (r_inv * spt.Rotation.from_quat(f_quat)).as_matrix(),
    }


def build_frames(cams, cam_names, t0_us: int):
    """Decode 4 frames per camera and return (N_cam, 4, 3, H, W) with camera_indices."""
    ts = np.array(
        [t0_us - (NUM_FRAMES - 1 - i) * int(TIME_STEP * 1e6) for i in range(NUM_FRAMES)],
        dtype=np.int64,
    )
    frames, idxs = [], []
    for obj, name in zip(cams, cam_names):
        img, _ = obj.decode_images_from_timestamps(ts)
        frames.append(rearrange(torch.from_numpy(img), "t h w c -> t c h w"))
        idxs.append(CAMERA_NAME_TO_INDEX.get(name, 0))
    order = np.argsort(idxs)
    return (
        torch.stack([frames[i] for i in order], dim=0),
        torch.tensor([idxs[i] for i in order], dtype=torch.int64),
    )


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


FRAME_SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("t0_us", pa.int64()),
        ("camera_indices", pa.list_(pa.int32())),
        ("num_frames_per_camera", pa.int32()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("jpeg_quality", pa.int32()),
        # JPEG bytes flattened camera major then frame (N_cam * num_frames images)
        ("jpegs", pa.list_(pa.binary())),
    ]
)


SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("t0_us", pa.int64()),
        ("sample_idx", pa.int32()),
        ("coc", pa.string()),
        ("ade", pa.float32()),
        ("min_ade_at_t0", pa.float32()),
        ("pass_filter", pa.bool_()),
        ("pred_xyz", pa.list_(pa.float32())),  # 64*3, flattened
        ("pred_rot", pa.list_(pa.float32())),  # 64*9, flattened
        ("gt_xyz", pa.list_(pa.float32())),  # 64*3
        ("hist_xyz", pa.list_(pa.float32())),  # 16*3
        ("hist_rot", pa.list_(pa.float32())),  # 16*9
        ("num_prompt_tokens", pa.int32()),
        ("gen_seconds", pa.float32()),
        ("model", pa.string()),
        ("seed", pa.int32()),
        ("temperature", pa.float32()),
        ("top_p", pa.float32()),
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


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)

    import pandas as pd
    import physical_ai_av

    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    clips = pd.read_parquet(args.clips_file)["clip_id"].tolist()
    if args.max_clips:
        clips = clips[: args.max_clips]
    mine = clips[args.worker_id :: args.num_workers]
    log(f"worker {args.worker_id}/{args.num_workers}: {len(mine)} clips assigned, GPU {args.gpu}")

    out_dir = args.out
    done = already_done(out_dir)
    log(f"already generated (clip,t0): {len(done)}, skipping")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    cam_feats = [
        avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
        avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
        avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
    ]
    cam_names = [c.split("/")[-1].lower() for c in cam_feats]

    model = Alpamayo1_5.from_pretrained(args.model, dtype=torch.bfloat16).to(f"cuda:{args.gpu}")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    log("model loaded")

    writer = ShardWriter(
        os.path.join(out_dir, "samples"), "coc", SCHEMA, args.worker_id, args.shard_rows
    )

    frame_writer = ShardWriter(
        os.path.join(out_dir, "frames"),
        "frames",
        FRAME_SCHEMA,
        args.worker_id,
        max(1, args.shard_rows // 8),
    )
    def _flush_and_exit(signum, _frame):
        # Avoid losing unflushed rows on restart. Resume keys on (clip,t0), so a
        # partial shard is picked up cleanly by the next run.
        log(f"signal {signum} received: flushing shards and exiting")
        try:
            writer.flush()
            frame_writer.flush()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _flush_and_exit)
    signal.signal(signal.SIGINT, _flush_and_exit)

    t0_grid_s = np.arange(args.t0_start, args.t0_end + 1e-9, args.t0_step)

    n_ok = n_pass = n_fail = 0
    tgt_w = tgt_h = None
    t_start = time.time()

    for ci, clip_id in enumerate(mine):
        pending = [int(round(s * 1e6)) for s in t0_grid_s if (clip_id, int(round(s * 1e6))) not in done]
        if not pending:
            continue
        try:
            ego = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
            cams = [avdi.get_clip_feature(clip_id, c, maybe_stream=True) for c in cam_feats]
        except Exception as e:
            log(f"[{ci+1}/{len(mine)}] {clip_id[:8]} clip open failed: {type(e).__name__} {str(e)[:90]}")
            continue

        # Video length varies per clip and is far shorter than egomotion (~20s vs ~140s).
        # Uncovered t0 values raise during decode, so trim them up front.
        try:
            cam_lo = max(int(c.timestamps.min()) for c in cams)
            cam_hi = min(int(c.timestamps.max()) for c in cams)
            ego_hi = int(ego.timestamps.max())
            lo = cam_lo + (NUM_FRAMES - 1) * int(TIME_STEP * 1e6)
            lo = max(lo, int(NUM_HISTORY_STEPS * TIME_STEP * 1e6) + 1)
            hi = min(cam_hi, ego_hi - int((NUM_FUTURE_STEPS + 1) * TIME_STEP * 1e6))
            kept = [t for t in pending if lo <= t <= hi]
            if len(kept) != len(pending):
                log(
                    f"  {clip_id[:8]} t0 range clamp "
                    f"[{lo/1e6:.1f}, {hi/1e6:.1f}]s: {len(pending)} -> {len(kept)}"
                )
            pending = kept
            if not pending:
                continue
        except Exception as e:
            log(f"  {clip_id[:8]} range computation failed ({type(e).__name__}), using the original grid")

        n_clip = 0
        for t0_us in pending:
            try:
                frames, cam_idx = build_frames(cams, cam_names, t0_us)
                traj = ego_window(ego, t0_us)

                # Build the JPEG to be stored first, then run inference on the decoded
                # frames. Measured: q92 compression alone flips 1 of 6 CoC outputs and
                # moves ADE up to 1.14m (resize alone is 0.06m). Inferring on originals
                # and storing a JPEG misaligns the stored image against the stored CoC.
                flat = frames.flatten(0, 1)  # (N_cam*num_frames, 3, H, W)
                jpegs: list[bytes] = []
                if not args.no_frames:
                    if tgt_w is None:
                        tgt_w, tgt_h = smart_size(
                            int(flat.shape[-1]),
                            int(flat.shape[-2]),
                            helper.MIN_PIXELS,
                            helper.MAX_PIXELS,
                        )
                        log(
                            f"  stored frame size: {flat.shape[-1]}x{flat.shape[-2]} -> "
                            f"{tgt_w}x{tgt_h} ({tgt_w*tgt_h} px, q{args.jpeg_quality})"
                        )
                    decoded = []
                    for img in flat:
                        pil = Image.fromarray(img.permute(1, 2, 0).numpy())
                        if (pil.width, pil.height) != (tgt_w, tgt_h):
                            pil = pil.resize((tgt_w, tgt_h), Image.BICUBIC)
                        buf = io.BytesIO()
                        pil.save(buf, "JPEG", quality=args.jpeg_quality)
                        raw = buf.getvalue()
                        jpegs.append(raw)
                        back = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
                        decoded.append(torch.from_numpy(back).permute(2, 0, 1))
                    flat = torch.stack(decoded, 0)

                msgs = helper.create_message(frames=flat, camera_indices=cam_idx)
                inputs = processor.apply_chat_template(
                    msgs,
                    tokenize=True,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                n_tok = int(inputs["input_ids"].shape[-1])
                model_inputs = helper.to_device(
                    {
                        "tokenized_data": inputs,
                        "ego_history_xyz": torch.from_numpy(traj["hist_xyz"])
                        .float()[None, None],
                        "ego_history_rot": torch.from_numpy(traj["hist_rot"])
                        .float()[None, None],
                    },
                    f"cuda:{args.gpu}",
                )

                torch.cuda.manual_seed_all(args.seed)
                t_gen = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        num_traj_samples=args.num_traj_samples,
                        max_generation_length=args.max_generation_length,
                        return_extra=True,
                    )
                gen_s = time.time() - t_gen

                gt_xyz = traj["fut_xyz"].astype(np.float32)  # (64,3)
                pxyz = pred_xyz.float().cpu().numpy()[0, 0]  # (S,64,3)
                prot = pred_rot.float().cpu().numpy()[0, 0]  # (S,64,3,3)
                ade = np.linalg.norm(pxyz[:, :, :2] - gt_xyz[None, :, :2], axis=-1).mean(-1)
                min_ade = float(ade.min())
                cots = np.asarray(extra["cot"]).reshape(-1).tolist()

                for si in range(len(cots)):
                    ok = bool(ade[si] <= args.ade_threshold)
                    n_pass += int(ok)
                    writer.add(
                        {
                            "clip_id": clip_id,
                            "t0_us": t0_us,
                            "sample_idx": si,
                            "coc": str(cots[si]),
                            "ade": float(ade[si]),
                            "min_ade_at_t0": min_ade,
                            "pass_filter": ok,
                            "pred_xyz": pxyz[si].reshape(-1).tolist(),
                            "pred_rot": prot[si].reshape(-1).tolist(),
                            "gt_xyz": gt_xyz.reshape(-1).tolist(),
                            "hist_xyz": traj["hist_xyz"].astype(np.float32).reshape(-1).tolist(),
                            "hist_rot": traj["hist_rot"].astype(np.float32).reshape(-1).tolist(),
                            "num_prompt_tokens": n_tok,
                            "gen_seconds": gen_s,
                            "model": args.model,
                            "seed": args.seed,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                        }
                    )
                # Images are per t0, so they are stored separately from the samples
                # table (S rows per t0). Putting them together would duplicate S times.
                if jpegs:
                    frame_writer.add(
                        {
                            "clip_id": clip_id,
                            "t0_us": t0_us,
                            "camera_indices": cam_idx.tolist(),
                            "num_frames_per_camera": NUM_FRAMES,
                            "width": tgt_w,
                            "height": tgt_h,
                            "jpeg_quality": args.jpeg_quality,
                            "jpegs": jpegs,
                        }
                    )

                n_ok += 1
                n_clip += 1
            except Exception as e:
                n_fail += 1
                if n_fail <= 5 or n_fail % 50 == 0:
                    log(f"  {clip_id[:8]} t0={t0_us/1e6:.1f}s failed: {type(e).__name__} {str(e)[:110]}")

        el = time.time() - t_start
        rate = n_ok / el * 3600 if el else 0
        log(
            f"[{ci+1}/{len(mine)}] {clip_id[:8]} +{n_clip}t0  "
            f"cum t0={n_ok} passed={n_pass} failed={n_fail}  {rate:.0f} t0/h"
        )

    writer.flush()
    frame_writer.flush()
    meta = {
        "worker_id": args.worker_id,
        "num_workers": args.num_workers,
        "args": vars(args),
        "t0_done": n_ok,
        "samples_passed": n_pass,
        "failures": n_fail,
        "elapsed_s": time.time() - t_start,
    }
    with open(os.path.join(out_dir, f"_meta-w{args.worker_id:02d}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"done: t0={n_ok} passed={n_pass} failed={n_fail}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
