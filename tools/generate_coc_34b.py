"""Alpamayo 2 Super (34B) teacher CoC 데이터셋 생성기.

10B 버전에서 얻은 교훈을 그대로 반영한다:
  - 카메라 객체를 클립당 한 번만 열고 t0를 여러 개 처리한다 (30배 차이)
  - JPEG를 **먼저** 만들고 그것을 디코드한 프레임으로 추론한다.
    원본으로 추론하고 JPEG를 저장하면 (저장 이미지 -> 저장 CoC) 쌍이 어긋난다.
    실측: q92 압축만으로 6개 중 1개의 CoC가 뒤집히고 ADE가 1.14m 움직였다.
  - t0 grid를 클립별 영상 길이로 클램프한다 (영상은 ~20s, egomotion은 ~140s)
  - 샘플 단위 ADE를 원본 그대로 저장한다. 임계값은 나중에 바꿀 수 있어야 한다.

34B 고유 사항:
  - **7카메라 전부 저장한다.** trajectory 태스크는 (0,1,2,3,5,6), vqa는 (0,1,2,3,4,5)를
    쓰므로 합집합이 7대다. 학습 때 빼는 건 되지만 없는 걸 추가할 수는 없다.
    추론에는 select_task_input()이 고른 부분집합만 들어간다.
  - 모델이 69GB라 GPU당 워커 2개가 상한.

GPU를 다른 사람이 써야 할 수 있으므로 언제든 중단·재개 가능하다:
  SIGTERM/SIGINT를 받으면 shard를 flush하고 종료하며, 재시작하면
  (clip_id, t0_us) 키로 남은 것부터 이어서 한다.
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
        help="seconds. 영상이 대개 20초를 못 채워 t0=20은 81%의 클립에서 클램프된다. "
        "그러면 완료된 클립에도 미완 t0가 남아 재시작마다 클립을 다시 열게 된다(워커당 3시간).",
    )
    p.add_argument("--t0-step", type=float, default=2.0, help="seconds")
    p.add_argument("--num-traj-samples", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ade-threshold", type=float, default=1.0, help="meters")
    p.add_argument("--shard-rows", type=int, default=512)
    p.add_argument("--flush-seconds", type=float, default=300.0, help="주기 flush 간격")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--topk", type=int, default=20, help="토큰별 상위 k개 분포 저장 (0=끔)")
    p.add_argument("--encode-threads", type=int, default=4, help="JPEG 인코딩 병렬도")
    p.add_argument("--t0-timeout", type=int, default=300, help="t0 하나의 상한(초) — 멈춤 방지")
    p.add_argument("--clip-open-timeout", type=int, default=600, help="클립 열기 상한(초)")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def smart_size(w: int, h: int, min_px: int, max_px: int, factor: int = 32) -> tuple[int, int]:
    """Qwen3-VL 프로세서가 고르는 것과 같은 크기(factor의 배수, 픽셀수 구간 내)."""
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
    """t0 기준 로컬 프레임의 history / future 궤적."""
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
    t0_quat = h_quat[-1].copy()  # scipy 규약: xyzw
    r_inv = spt.Rotation.from_quat(t0_quat).inv()
    return {
        "hist_xyz": r_inv.apply(h_xyz - t0_xyz),
        "hist_rot": (r_inv * spt.Rotation.from_quat(h_quat)).as_matrix(),
        "fut_xyz": r_inv.apply(f_xyz - t0_xyz),
        "fut_rot": (r_inv * spt.Rotation.from_quat(f_quat)).as_matrix(),
        "t0_xyz": t0_xyz,
        # 모델은 wxyz 순서의 역회전 쿼터니언을 받는다
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
        # 시퀀스 logprob (선택된 토큰들의 log P 합) — teacher 확신도 신호.
        # 모델이 반환하는 `logprob`은 전부 0인 placeholder라 직접 계산한다.
        ("seq_logprob", pa.float32()),
        ("num_gen_tokens", pa.int32()),
        # token-level KD용 top-k 분포. 34B와 Qwen3-VL student는 텍스트 토크나이저가
        # 완전히 같으므로(151,669 토큰 일치) 이 분포를 그대로 KL 타깃으로 쓸 수 있다.
        # 저장 비용 16 KB/t0 = 프레임 대비 +1.2%.
        ("gen_token_ids", pa.list_(pa.int32())),  # steps
        ("topk_ids", pa.list_(pa.int32())),  # steps*k, 행 우선
        ("topk_logprobs", pa.list_(pa.float32())),  # steps*k
        ("topk_k", pa.int32()),
        ("num_prompt_tokens", pa.int32()),
        ("gen_seconds", pa.float32()),
        ("model", pa.string()),
        ("seed", pa.int32()),
        ("temperature", pa.float32()),
        ("top_p", pa.float32()),
        # 추론에 실제로 들어간 카메라 (저장된 7대의 부분집합)
        ("input_camera_ids", pa.list_(pa.int32())),
    ]
)

# 클립별로 t0가 실제로 가능한 구간. 클립을 열어야만 알 수 있는 값이라 캐시한다.
# 이게 없으면 영상이 짧은 클립은 재시작할 때마다 "남은 t0"가 있는 것처럼 보여
# 클립을 다시 열고(40~55초) 나서야 버리게 된다.
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
        # 저장은 항상 canonical 7카메라 ring (0..6) 순서
        ("camera_indices", pa.list_(pa.int32())),
        ("num_frames_per_camera", pa.int32()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("jpeg_quality", pa.int32()),
        ("jpegs", pa.list_(pa.binary())),  # 7*4 = 28장, 카메라 순 -> 프레임 순
        # 실차 재현에 필요한 타이밍 (모델은 0.1s 간격을 전제한다)
        ("relative_timestamps", pa.list_(pa.float32())),  # 7*4
        ("absolute_timestamps", pa.list_(pa.int64())),  # 7*4
    ]
)


class ShardWriter:
    """행을 모아 shard 단위 parquet으로 flush (lustre 소파일 회피)."""

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
        log(f"  shard 저장: {os.path.basename(self._path(self.shard))} ({len(self.rows)} rows)")
        self.rows = []
        self.shard += 1


def already_done(out_dir: str) -> set:
    """기존 samples shard에서 (clip_id, t0_us) 키를 읽어 재개를 지원."""
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
            log(f"  경고: {fn} 읽기 실패 ({type(e).__name__}), 무시")
    return done


def load_ranges(out_dir: str) -> dict:
    """클립별 t0 가능 구간 캐시를 읽는다. {clip_id: (lo_us, hi_us)}"""
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
            log(f"  경고: {fn} 읽기 실패 ({type(e).__name__}), 무시")
    return ranges


def snapshot_config(out_dir: str, model, args: argparse.Namespace) -> None:
    """실차 이식에 필요한 규격을 데이터셋에 같이 박아둔다.

    궤적 출력은 unicycle 액션 공간의 정규화 상수에 묶여 있다. 차량 동역학이
    다르면 같은 토큰이 다른 궤적을 뜻하므로, 모델 config와 분리해 보관하면 안 된다.
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
    # expert_config는 dict일 수도 PretrainedConfig 객체일 수도 있다
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
    log(f"config 스냅샷 저장: {path}")


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

    # helper.prepare_model_inputs()가 t0마다 get_processor()를 부르고, 그 안에서
    # AutoProcessor.from_pretrained()가 매번 디스크를 읽어 프로세서를 새로 만든다.
    # 계측 결과 tokenize 구간이 전체의 34~46%로 최대 병목이었다(추론은 14~17%).
    # 프로세서는 (tokenizer, config)가 고정이면 항상 같으므로 한 번만 만든다.
    _proc_cache: dict = {}
    _orig_get_processor = helper.get_processor

    def _cached_get_processor(tokenizer, model_config):
        if "p" not in _proc_cache:
            _proc_cache["p"] = _orig_get_processor(tokenizer, model_config)
        return _proc_cache["p"]

    helper.get_processor = _cached_get_processor

    canonical_names = list(CAMERA_NAMES_TO_INDICES)  # 0..6 순서
    canonical_ids = list(CAMERA_NAMES_TO_INDICES.values())

    clips = pd.read_parquet(args.clips_file)["clip_id"].tolist()
    if args.max_clips:
        clips = clips[: args.max_clips]
    mine = clips[args.worker_id :: args.num_workers]
    log(f"worker {args.worker_id}/{args.num_workers}: 담당 클립 {len(mine)}개, GPU {args.gpu}")

    out_dir = args.out
    done = already_done(out_dir)
    log(f"이미 생성된 (clip,t0): {len(done)}개 — 건너뜀")
    clip_ranges = load_ranges(out_dir)
    log(f"t0 구간 캐시: {len(clip_ranges)}개 클립 — 열기 전에 걸러냄")

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    cam_feats = [getattr(avdi.features.CAMERA, n.upper()) for n in canonical_names]

    model = Alpamayo2Super.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}"
    )
    model.eval()
    log("모델 로드 완료 (34B)")

    # 토큰별 분포를 붙잡는다. `_generate_with_shared_prefill`이 내부에서
    # generation_config.output_logits = False 로 덮어쓰므로 바깥에서 설정해도 소용없다.
    # 실제 generate 호출 지점을 감싸는 게 유일하게 깨끗한 개입점이다.
    # 실측: 시간·메모리 증가 없음 (로짓은 어차피 계산되고, 스텝 수가 ~23으로 짧다).
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
        log(f"토큰 분포 저장 활성화 (top-{args.topk})")
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

    # 구간 캐시는 행이 작고 재시작 때 가장 먼저 필요하므로 자주 flush 한다
    range_writer = ShardWriter(
        os.path.join(out_dir, "ranges"), "range", RANGE_SCHEMA, args.worker_id, 64
    )

    def _flush_and_exit(signum, _frame):
        log(f"신호 {signum} 수신 — shard flush 후 종료 (재시작하면 이어서 진행)")
        try:
            writer.flush()
            frame_writer.flush()
            range_writer.flush()
        finally:
            os._exit(0)

    signal.signal(signal.SIGTERM, _flush_and_exit)
    signal.signal(signal.SIGINT, _flush_and_exit)

    def _on_timeout(_signum, _frame):
        raise TimeoutError(f"t0 처리가 {args.t0_timeout}s를 넘김")

    signal.signal(signal.SIGALRM, _on_timeout)
    pool = ThreadPoolExecutor(max_workers=args.encode_threads)

    t0_grid_s = np.arange(args.t0_start, args.t0_end + 1e-9, args.t0_step)
    tgt_w = tgt_h = None
    n_ok = n_pass = n_fail = 0
    # 구간별 누적 시간 — GPU가 노는 이유를 실측으로 가리기 위한 계측
    phase = {"open": 0.0, "decode": 0.0, "encode": 0.0, "tokenize": 0.0, "infer": 0.0, "post": 0.0}
    t_start = last_flush = time.time()

    def _open_clip(cid: str):
        """egomotion + 7카메라를 연다. 클립당 ~56초의 네트워크 스트리밍."""
        e = avdi.get_clip_feature(cid, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
        cs = [avdi.get_clip_feature(cid, c, maybe_stream=True) for c in cam_feats]
        return e, cs

    def _pending_for(cid: str) -> list:
        """아직 안 만든 t0. 구간 캐시가 있으면 **클립을 열기 전에** 걸러낸다.

        캐시가 없으면(처음 보는 클립) 열어봐야 알 수 있으므로 그대로 둔다.
        """
        p = [int(round(s * 1e6)) for s in t0_grid_s if (cid, int(round(s * 1e6))) not in done]
        rng = clip_ranges.get(cid)
        if rng is not None:
            lo, hi = rng
            p = [t for t in p if lo <= t <= hi]
        return p

    def _has_work(cid: str) -> bool:
        return bool(_pending_for(cid))

    # 클립 열기(56s)가 클립당 wall-clock의 26%인데 그동안 CPU도 GPU도 논다.
    # 현재 클립의 t0를 처리하는 동안 다음 클립을 미리 열어 그 구간을 숨긴다.
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
            continue  # 캐시 덕에 클립을 열지 않고 넘어간다
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
            _schedule_prefetch(ci + 1)  # 다음 클립을 미리 연다
        except Exception as e:
            signal.alarm(0)
            prefetch.pop("future", None)
            prefetch.pop("clip_id", None)
            log(f"[{ci+1}/{len(mine)}] {clip_id[:8]} 클립 열기 실패: {type(e).__name__} {str(e)[:90]}")
            continue

        # 영상 길이는 클립마다 다르고 egomotion보다 훨씬 짧다 (~20s vs ~140s)
        try:
            lo = max(int(c.timestamps.min()) for c in cams) + (NUM_FRAMES - 1) * int(
                TIME_STEP * 1e6
            )
            lo = max(lo, int(NUM_HISTORY_STEPS * TIME_STEP * 1e6) + 1)
            hi = min(
                min(int(c.timestamps.max()) for c in cams),
                int(ego.timestamps.max()) - int((NUM_FUTURE_STEPS + 1) * TIME_STEP * 1e6),
            )
            # 다음 실행부터는 이 클립을 열지 않고 걸러낼 수 있게 구간을 남긴다
            if clip_id not in clip_ranges:
                clip_ranges[clip_id] = (lo, hi)
                range_writer.add({"clip_id": clip_id, "lo_us": lo, "hi_us": hi})
            kept = [t for t in pending if lo <= t <= hi]
            if len(kept) != len(pending):
                log(
                    f"  {clip_id[:8]} t0 클램프 [{lo/1e6:.1f}, {hi/1e6:.1f}]s: "
                    f"{len(pending)} -> {len(kept)}"
                )
            pending = kept
            if not pending:
                continue
        except Exception as e:
            log(f"  {clip_id[:8]} 범위 계산 실패({type(e).__name__}), 원래 grid 사용")

        n_clip = 0
        for t0_us in pending:
            try:
                # 네트워크 스트리밍이 걸려 워커가 영영 멈추는 걸 막는다.
                # 정상 t0는 10초대이므로 넉넉히 잡아도 사고만 걸러낸다.
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

                # JPEG를 먼저 만들고 그것을 디코드한 프레임으로 추론한다.
                # 순서를 바꾸면 저장 이미지와 저장 CoC의 쌍이 어긋난다.
                if tgt_w is None:
                    tgt_w, tgt_h = smart_size(
                        int(image_frames.shape[-1]),
                        int(image_frames.shape[-2]),
                        model.config.min_pixels,
                        model.config.max_pixels,
                    )
                    log(
                        f"  프레임 저장 크기: {image_frames.shape[-1]}x{image_frames.shape[-2]}"
                        f" -> {tgt_w}x{tgt_h} ({tgt_w*tgt_h} px, q{args.jpeg_quality}), "
                        f"카메라 {len(cams)}대"
                    )
                # 28장 인코딩이 GPU 추론(2.8s)보다 오래 걸린다. PIL은 인코딩 중
                # GIL을 놓으므로 스레드로 병렬화하면 그대로 이득이다.
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
                # load_physical_aiavdataset()의 반환 계약을 그대로 재현한다.
                # 카메라를 클립당 한 번만 열려고 그 함수를 안 쓰는 것이므로,
                # 키가 하나라도 빠지면 select_task_input에서 걸린다.
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
                # 저장은 7대 전부, 추론에는 태스크가 요구하는 부분집합만
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

                # 토큰별 top-k 분포와 실제 시퀀스 logprob을 뽑는다.
                # 모델이 반환하는 logprob은 zeros_like(...)라 쓸 수 없다.
                topk_ids = topk_lps = gen_ids = None
                seq_lp = None
                if captured.get("logits"):
                    # 스텝을 통째로 stack하면 (B, steps, 155776) fp32를 한 번에 잡는다.
                    # steps가 상한(256)에 가까우면 957MB고, GPU가 이미 98% 차 있어
                    # 할당자가 스래싱한다. 스텝별로 처리하면 피크가 (B, 155776)=3.7MB로
                    # 고정된다. topk는 정규화에 불변이므로 원본 로짓에서 골라도 된다.
                    # 스텝마다 .cpu()를 부르면 스텝 수 x 3 회의 동기화 전송이 생기고
                    # 매번 GPU 큐를 기다린다(42스텝이면 126회, 실측 +7초/t0).
                    # topk 결과는 (B, k)로 아주 작으므로 GPU에 모아두었다가
                    # 마지막에 한 번만 옮긴다. 피크 메모리는 스텝별 처리 그대로 유지된다.
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

                # 주기 flush — 언제 중단돼도 잃는 작업을 이 간격으로 제한한다
                if time.time() - last_flush > args.flush_seconds:
                    writer.flush()
                    frame_writer.flush()
                    range_writer.flush()
                    last_flush = time.time()
            except Exception as e:
                n_fail += 1
                if n_fail <= 5 or n_fail % 50 == 0:
                    log(
                        f"  {clip_id[:8]} t0={t0_us/1e6:.1f}s 실패: "
                        f"{type(e).__name__} {str(e)[:110]}"
                    )
            finally:
                signal.alarm(0)

        el = time.time() - t_start
        tot = sum(phase.values()) or 1.0
        log("  구간%: " + " ".join(f"{k}={100*v/tot:.0f}" for k, v in phase.items())
            + f"  (합 {tot:.0f}s / 경과 {el:.0f}s)")
        log(
            f"[{ci+1}/{len(mine)}] {clip_id[:8]} +{n_clip}t0  "
            f"누적 t0={n_ok} 통과샘플={n_pass} 실패={n_fail}  "
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
    log(f"완료: t0={n_ok} 통과샘플={n_pass} 실패={n_fail}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
