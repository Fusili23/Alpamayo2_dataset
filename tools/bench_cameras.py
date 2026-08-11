"""Alpamayo 2 Super: inference latency by camera count and frame count.

To judge whether prefill is affordable on Thor we need to know how much faster
inference gets when cameras are removed. Input tokens scale linearly with camera
count, but attention is quadratic, so latency may scale superlinearly.

Measured:
  prompt token count (determined purely by configuration)
  prefill latency (approximated with max_new_tokens=1)
  total latency (CoC generation plus diffusion expert trajectory)
  quality (minADE), the cost of dropping cameras

Uses num_traj_samples=1 to model a deployment scenario, unlike the 6 used for
dataset generation.
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

    # build the processor once; rebuilding per t0 would pollute the measurement
    _cache: dict = {}
    _orig = helper.get_processor

    def _cached(tokenizer, model_config):
        if "p" not in _cache:
            _cache["p"] = _orig(tokenizer, model_config)
        return _cache["p"]

    helper.get_processor = _cached

    print(f"loading clip: {args.clip_id} @ t0={args.t0_us/1e6:.1f}s", flush=True)
    source = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    model = Alpamayo2Super.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=f"cuda:{args.gpu}"
    ).eval()
    print("model loaded\n", flush=True)

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
        # one warmup pass (kernel compilation, cache fill)
        run_once(profile, 256)
        prefills, totals, ades, n_tok = [], [], [], 0
        for _ in range(args.repeats):
            n_tok, dt1, _ = run_once(profile, 1)  # prefill approximation
            prefills.append(dt1)
            _, dt2, ade = run_once(profile, 256)  # full
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
            f"  {label:16s} images {r['images']:2d}  tokens {r['prompt_tokens']:5d}  "
            f"prefill {r['prefill_s']:.3f}s  total {r['total_s']:.3f}s  minADE {r['min_ade']:.3f}",
            flush=True,
        )
        return r

    results = []
    print("=== varying camera count (4 frames fixed) ===", flush=True)
    for n in range(1, 7):
        results.append(
            bench(f"{n}cam x 4frame", InputProfile(tuple(base_cams[:n]), (0, 1, 2, 3)))
        )

    print("\n=== varying frame count (6 cameras fixed) ===", flush=True)
    for fr in ((3,), (2, 3), (1, 2, 3)):
        results.append(
            bench(f"6cam x {len(fr)}frame", InputProfile(tuple(base_cams), fr))
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved: {args.out}")

    base = next(r for r in results if r["cameras"] == 6 and r["frames"] == 4)
    print("\n=== relative to 6 cameras x 4 frames ===")
    print("config              tokens    total        speedup    minADE")
    for r in results:
        print(
            f"  {r['label']:16s} {r['prompt_tokens']:5d}  {r['total_s']:7.3f}s  "
            f"{base['total_s']/r['total_s']:6.2f}x   {r['min_ade']:.3f}"
        )


if __name__ == "__main__":
    main()
