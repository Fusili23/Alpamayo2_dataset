"""Train a small student on the Alpamayo 2 format dataset and confirm loss decreases.

The goal is format validation, not performance. Once the 34B teacher is gone the
dataset cannot be regenerated, so we must confirm now that it fits a real
training path.

The SFT recipe in NVlabs/alpamayo-recipes targets Alpamayo 1.5 (chat template
r1/r1_5, QwenProcessor, published OOD labels) and cannot be used as is. Instead
this uses a minimal loop over the alpamayo2_super library directly, which is the
same path already validated by the generation pipeline.

The training forward pass does not use the diffusion expert. Trajectories are
fused as discrete tokens and learned with next token loss alongside the text
(Stage 1 in the recipes README).
"""

import argparse
import glob
import io
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
TIME_STEP = 0.1
NUM_FRAMES = 4


def parse_args() -> argparse.Namespace:
    big = os.environ.get("BIG", "/NHNHOME/WORKSPACE/0526050025_A/alpamayo")
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=f"{big}/data/coc_34b_v1")
    p.add_argument("--teacher", default="nvidia/Alpamayo2-Super", help="source of config and tokenizer")
    p.add_argument("--backbone", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--out", default=f"{big}/logs/train_smoke_result.json")
    return p.parse_args()


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_pairs(data_dir: str, limit: int) -> list:
    """Join samples and frames on (clip_id, t0_us) to build training examples."""
    fr = {}
    need = limit
    for fn in sorted(glob.glob(f"{data_dir}/frames/*.parquet")):
        t = pq.read_table(fn).to_pandas()
        for _, r in t.iterrows():
            fr[(r.clip_id, r.t0_us)] = r
        if len(fr) >= need:
            break
    log(f"frames {len(fr)} loaded")

    out = []
    seen = set()
    for fn in sorted(glob.glob(f"{data_dir}/samples/*.parquet")):
        t = pq.read_table(fn).to_pandas()
        # one best sample per t0 (lowest ADE); enough for validation
        t = t.sort_values("ade").drop_duplicates(["clip_id", "t0_us"], keep="first")
        for _, s in t.iterrows():
            k = (s.clip_id, s.t0_us)
            if k in seen or k not in fr:
                continue
            seen.add(k)
            out.append((s, fr[k]))
            if len(out) >= limit:
                return out
    return out


def build_source_data(s, f, canonical_names, canonical_ids) -> dict:
    """Rebuild the load_physical_aiavdataset() return contract from stored JPEGs and trajectories."""
    imgs = [
        torch.from_numpy(np.array(Image.open(io.BytesIO(b)).convert("RGB"))).permute(2, 0, 1)
        for b in f.jpegs
    ]
    n_cam = len(f.camera_indices)
    image_frames = torch.stack(imgs, 0).reshape(n_cam, NUM_FRAMES, 3, int(f.height), int(f.width))
    abs_ts = torch.tensor(list(f.absolute_timestamps), dtype=torch.int64).reshape(n_cam, NUM_FRAMES)
    ctmin = int(abs_ts.min().item())
    t0 = int(s.t0_us)

    hx = torch.tensor(np.array(s.hist_xyz, dtype=np.float32).reshape(NUM_HISTORY_STEPS, 3))
    hr = torch.tensor(np.array(s.hist_rot, dtype=np.float32).reshape(NUM_HISTORY_STEPS, 3, 3))
    fx = torch.tensor(np.array(s.gt_xyz, dtype=np.float32).reshape(NUM_FUTURE_STEPS, 3))
    frot = torch.tensor(np.array(s.gt_rot, dtype=np.float32).reshape(NUM_FUTURE_STEPS, 3, 3))

    return {
        "image_frames": image_frames,
        "camera_indices": torch.tensor(canonical_ids, dtype=torch.int64),
        "camera_names": canonical_names,
        "relative_timestamps": (abs_ts - abs_ts.min()).float() * 1e-6,
        "absolute_timestamps": abs_ts,
        "camera_tmin": ctmin,
        "ego_available": torch.tensor(True),
        "ego_t0": torch.tensor([t0], dtype=torch.int64),
        "ego_t0_relative": torch.tensor([(t0 - ctmin) * 1e-6], dtype=torch.float32),
        "ego_t0_frame_idx": torch.tensor([NUM_FRAMES - 1], dtype=torch.int64),
        "prediction_start_offset": torch.zeros(1, dtype=torch.float32),
        "ego_history_tvals": torch.arange(-NUM_HISTORY_STEPS + 1, 1, dtype=torch.float32) * TIME_STEP,
        "ego_future_tvals": torch.arange(1, NUM_FUTURE_STEPS + 1, dtype=torch.float32) * TIME_STEP,
        "ego_history_xyz": hx[None, None],
        "ego_history_rot": hr[None, None],
        "ego_future_xyz": fx[None, None],
        "ego_future_rot": frot[None, None],
        # local frame at t0, so origin and identity rotation
        "ego_t0_xyz": torch.zeros(1, 1, 3),
        "ego_t0_inv_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4),
        "t0_us": t0,
        "clip_id": s.clip_id,
    }


def prepare_training_inputs(data, coc_text, model, helper, build_conversation):
    """Tokenize for training. helper.prepare_model_inputs hardcodes generation_mode=True.

    Generation mode removes the target from the sequence, leaving zero future
    trajectory placeholders, so fuse_traj_tokens fails trying to insert 128.
    Training needs the CoC text and trajectory placeholders inside the sequence
    so a loss can be computed on them.
    """
    cfg = model.config
    data = dict(data)
    data["cot"] = coc_text  # target text that becomes the assistant response

    messages = build_conversation(
        data=data,
        num_tokens_per_history_traj=cfg.tokens_per_history_traj,
        num_tokens_per_future_traj=cfg.tokens_per_future_traj,
        components_order=["image", "traj_history", "prompt", "cot", "traj_future"],
        components_prompt=["cot", "traj_future"],
        generation_mode=False,  # training: the target is part of the sequence
        include_camera_ids=cfg.include_camera_ids,
        camera_ids=data["camera_indices"],
        include_frame_nums=cfg.frame_label == "frame_num",
    )
    processor = helper.get_processor(model.tokenizer, cfg)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_vision_id=False
    )
    images = data["image_frames"].flatten(0, 1)
    images = (images.float() / 255.0) if images.dtype == torch.uint8 else images.float()
    return dict(
        processor(text=text, images=images, videos=None, padding=False,
                  return_tensors="pt", do_rescale=False)
    )


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    dev = f"cuda:{args.gpu}"

    from alpamayo2_super import helper
    from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES
    from alpamayo2_super.config import Alpamayo2SuperConfig
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.chat_template.conversation import build_conversation
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    names = list(CAMERA_NAMES_TO_INDICES)
    ids = list(CAMERA_NAMES_TO_INDICES.values())

    # Do not load and edit the teacher config: vlm_config is already fixed to the 34B
    # and traj_ids are computed against the 34B tokenizer.
    # Passing only vlm_name_or_path and constructing from scratch makes the config
    # recompute tokenizer expansion, traj_ids and vocab size for the new backbone.
    t_cfg = Alpamayo2SuperConfig.from_pretrained(args.teacher)
    cfg = Alpamayo2SuperConfig(
        vlm_name_or_path=args.backbone,
        hist_traj_tokenizer_cfg=t_cfg.hist_traj_tokenizer_cfg,
        future_traj_tokenizer_cfg=t_cfg.future_traj_tokenizer_cfg,
        history_vocab_size=t_cfg.history_vocab_size,
        future_vocab_size=t_cfg.future_vocab_size,
        tokens_per_history_traj=t_cfg.tokens_per_history_traj,
        tokens_per_future_traj=t_cfg.tokens_per_future_traj,
        min_pixels=t_cfg.min_pixels,
        max_pixels=t_cfg.max_pixels,
        include_camera_ids=t_cfg.include_camera_ids,
        token_layout=t_cfg.token_layout,
        frame_label=t_cfg.frame_label,
        loss_weights=t_cfg.loss_weights,
        enable_expert=False,  # the training forward does not use the expert
    )
    log(f"student config: backbone {args.backbone}, expert disabled")
    log(
        f"  traj_vocab={cfg.traj_vocab_size} hist_tok={cfg.tokens_per_history_traj} "
        f"fut_tok={cfg.tokens_per_future_traj}"
    )

    model = Alpamayo2Super(cfg)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"model built: {n_par/1e9:.2f}B parameters (randomly initialized)")

    # Load the pretrained backbone. Loss falling from random init proves little,
    # since memorizing the token distribution alone would do it.
    # The vocab is extended by the trajectory tokens, so embeddings and lm_head
    import transformers

    pre = getattr(transformers, cfg.vlm_class).from_pretrained(
        args.backbone, dtype=torch.bfloat16
    )
    sd_pre, sd_new = pre.state_dict(), model.vlm.state_dict()
    copied = partial = skipped = 0
    for k, v in sd_pre.items():
        if k not in sd_new:
            skipped += 1
            continue
        if sd_new[k].shape == v.shape:
            sd_new[k].copy_(v)
            copied += 1
        elif sd_new[k].ndim == v.ndim and sd_new[k].shape[1:] == v.shape[1:]:
            n = min(sd_new[k].shape[0], v.shape[0])  # extended vocab: copy the leading rows only
            sd_new[k][:n].copy_(v[:n])
            partial += 1
        else:
            skipped += 1
    model.vlm.load_state_dict(sd_new)
    del pre, sd_pre, sd_new
    log(f"pretrained load: full copy {copied}, partial copy {partial} (vocab expansion), skipped {skipped}")

    model = model.to(dev, dtype=torch.bfloat16)

    # build the processor once (lesson from the generator: rebuilding per t0 wastes 40%)
    _cache: dict = {}
    _orig = helper.get_processor
    helper.get_processor = lambda t, c: _cache.setdefault("p", _orig(t, c))

    pairs = load_pairs(args.data, args.max_samples)
    log(f"{len(pairs)} training examples")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()

    losses = []
    t_start = time.time()
    step = 0
    idx = 0
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.grad_accum):
            s, f = pairs[idx % len(pairs)]
            idx += 1
            src = build_source_data(s, f, names, ids)
            data = select_task_input(src, "trajectory")
            td = prepare_training_inputs(
                data, str(s.coc), model, helper, build_conversation
            )
            td = helper.to_device(td, dev)
            traj = {
                "ego_history_xyz": data["ego_history_xyz"].to(dev),
                "ego_history_rot": data["ego_history_rot"].to(dev),
                "ego_future_xyz": data["ego_future_xyz"].to(dev),
                "ego_future_rot": data["ego_future_rot"].to(dev),
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(tokenized_data=td, traj_data=traj)
            (out.loss / args.grad_accum).backward()
            acc += float(out.loss.detach()) / args.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(acc)
        step += 1
        if step % args.log_every == 0 or step == 1:
            recent = np.mean(losses[-args.log_every :])
            log(
                f"step {step:4d}/{args.steps}  loss {acc:.4f}  "
                f"recent avg {recent:.4f}  {(time.time()-t_start)/step:.1f}s/step  "
                f"mem {torch.cuda.max_memory_allocated()/1e9:.0f}GB"
            )

    first = float(np.mean(losses[: max(1, len(losses) // 5)]))
    last = float(np.mean(losses[-max(1, len(losses) // 5) :]))
    log("")
    log(f"=== result ===")
    log(f"  first 20% mean loss : {first:.4f}")
    log(f"  last 20% mean       : {last:.4f}")
    log(f"  decrease            : {first-last:+.4f}  ({100*(first-last)/first:+.1f}%)")
    verdict = "loss decreased; dataset format is valid" if last < first * 0.95 else "insufficient decrease; needs investigation"
    log(f"  verdict: {verdict}")

    with open(args.out, "w") as fp:
        json.dump(
            {
                "losses": losses,
                "first_20pct": first,
                "last_20pct": last,
                "verdict": verdict,
                "params_B": n_par / 1e9,
                "args": vars(args),
            },
            fp,
            indent=2,
        )
    log(f"saved: {args.out}")


if __name__ == "__main__":
    main()
