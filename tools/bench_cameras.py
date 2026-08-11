"""Alpamayo 2 Super — 카메라 수·프레임 수에 따른 지연 시간 측정.

Thor 배포에서 prefill이 감당 가능한지 판단하려면 "카메라를 줄이면 얼마나 빨라지는가"를
알아야 한다. 입력 토큰은 카메라 수에 선형이지만 attention은 제곱이라 지연은 초선형일 수 있다.

측정 항목:
  - 프롬프트 토큰 수 (구성만으로 정해지는 값)
  - prefill 지연 (max_new_tokens=1 로 근사)
  - 전체 지연 (CoC 생성 + diffusion expert 궤적)
  - 품질 (minADE) — 카메라를 줄였을 때의 대가

배포 시나리오를 가정해 num_traj_samples=1 로 잰다(데이터 생성 때의 6과 다름).
"""

import argparse
import json
import os
import time

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    big = os.environ.get("BIG", "/NHNHOME/WORKSPACE/0526050025_A/alpamayo")
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="nvidia/Alpamayo2-Super")
    p.add_argument("--clip-id", default="0347d9f9-1493-4954-865d-1d8464e28501")
    p.add_argument("--t0-us", type=int, default=5_000_000)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--num-traj-samples", type=int, default=1)
    p.add_argument("--out", default=f"{big}/data/bench_cameras.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from alpamayo2_super import helper
    from alpamayo2_super.input_profiles import (
        DRIVING_SIX_CAMERA_FOUR_FRAME,
        InputProfile,
        select_input_profile,
    )
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    # 프로세서는 한 번만 (t0마다 새로 만들면 그 비용이 측정에 섞인다)
    _cache: dict = {}
    _orig = helper.get_processor

    def _cached(tokenizer, model_config):
        if "p" not in _cache:
            _cache["p"] = _orig(tokenizer, model_config)
        return _cache["p"]

    helper.get_processor = _cached

    print(f"클립 로드: {args.clip_id} @ t0={args.t0_us/1e6:.1f}s", flush=True)
    source = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    model = Alpamayo2Super.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}"
    ).eval()
    print("모델 로드 완료\n", flush=True)

    gt = source["ego_future_xyz"].cpu().numpy()[0, 0]
    base_cams = list(DRIVING_SIX_CAMERA_FOUR_FRAME.camera_ids)  # (0,1,2,3,5,6)

    def run_once(profile: InputProfile, max_new: int | None):
        data = select_input_profile(source, profile)
        mi = helper.prepare_model_inputs(data, model.config, model.tokenizer)
        n_tok = int(mi["tokenized_data"]["input_ids"].shape[-1])
        mi = helper.to_device(mi, f"cuda:{args.gpu}")
        torch.cuda.synchronize()
        torch.cuda.manual_seed_all(42)
        t = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            px, _, _, _ = model.sample_trajectories_from_data(
                data=mi,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=args.num_traj_samples,
                max_generation_length=max_new,
                return_extra=True,
            )
        torch.cuda.synchronize()
        dt = time.time() - t
        p = px.float().cpu().numpy().reshape(-1, 64, 3)
        ade = np.linalg.norm(p[:, :, :2] - gt[None, :, :2], axis=-1).mean(-1).min()
        return n_tok, dt, float(ade)

    def bench(label: str, profile: InputProfile) -> dict:
        # 워밍업 1회 (커널 컴파일·캐시 채우기)
        run_once(profile, 256)
        prefills, totals, ades, n_tok = [], [], [], 0
        for _ in range(args.repeats):
            n_tok, dt1, _ = run_once(profile, 1)  # prefill 근사
            prefills.append(dt1)
            _, dt2, ade = run_once(profile, 256)  # 전체
            totals.append(dt2)
            ades.append(ade)
        r = {
            "label": label,
            "cameras": len(profile.camera_ids),
            "frames": len(profile.frame_indices),
            "images": len(profile.camera_ids) * len(profile.frame_indices),
            "prompt_tokens": n_tok,
            "prefill_s": float(np.median(prefills)),
            "total_s": float(np.median(totals)),
            "min_ade": float(np.median(ades)),
        }
        print(
            f"  {label:16s} 이미지 {r['images']:2d}  토큰 {r['prompt_tokens']:5d}  "
            f"prefill {r['prefill_s']:.3f}s  전체 {r['total_s']:.3f}s  minADE {r['min_ade']:.3f}",
            flush=True,
        )
        return r

    results = []
    print("=== 카메라 수 변화 (프레임 4 고정) ===", flush=True)
    for n in range(1, 7):
        results.append(
            bench(f"{n}카메라x4프레임", InputProfile(tuple(base_cams[:n]), (0, 1, 2, 3)))
        )

    print("\n=== 프레임 수 변화 (6카메라 고정) ===", flush=True)
    for fr in ((3,), (2, 3), (1, 2, 3)):
        results.append(
            bench(f"6카메라x{len(fr)}프레임", InputProfile(tuple(base_cams), fr))
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n저장: {args.out}")

    base = next(r for r in results if r["cameras"] == 6 and r["frames"] == 4)
    print("\n=== 6카메라x4프레임 대비 ===")
    print("구성                 토큰      전체지연     속도향상   minADE")
    for r in results:
        print(
            f"  {r['label']:16s} {r['prompt_tokens']:5d}  {r['total_s']:7.3f}s  "
            f"{base['total_s']/r['total_s']:6.2f}x   {r['min_ade']:.3f}"
        )


if __name__ == "__main__":
    main()
