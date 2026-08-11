# Alpamayo 2 Super CoC Dataset Generation Pipeline

Code for building a Chain-of-Causation reasoning dataset from autonomous driving clips using the NVIDIA Alpamayo 2 Super (34B) teacher model. The goal is to distill a small student model that can run on a Jetson Thor in a vehicle.

The teacher is 69GB and takes 1.4 seconds per inference even on a B200, so it cannot go in a car. The approach is to extract as much teacher output as possible while GPU server time is available, then train the student later on smaller hardware.

## What gets generated

For each clip, timestamps (t0) are sampled every 2 seconds. At each t0 the teacher generates 6 samples. The following is stored:

| Table | Row unit | Contents |
| :-- | :-- | :-- |
| `frames/` | (clip, t0) | 7 cameras x 4 frames = 28 JPEGs, timestamps |
| `samples/` | (clip, t0, sample) | CoC text, predicted trajectory, GT trajectory, ego history, token distributions, ADE |
| `ranges/` | clip | Valid t0 interval for the clip (resume optimization) |

Storage is about 1.15MB per t0, of which 98 percent is imagery.

## Why token distributions are stored

Storing only the sampled text produces sequence level SFT data. Storing the top 20 probabilities at each token position enables token level KL distillation.

Looking at actual stored values, the teacher weighs "Adapt" (42 percent) against "Nudge" (37 percent) on the very first token. If only the sampled text is kept, that 37 percent is lost forever.

The cost is 24KB per t0, roughly 2 percent of the frame storage. The text tokenizers of the 34B teacher and Qwen3-VL students are identical (all 151,669 tokens match), so these distributions can be used directly as KL targets.

This cannot be added later. Unlike frames, which can be recomputed, it would require rerunning the teacher.

## Files

| File | Purpose |
| :-- | :-- |
| `tools/generate_coc_34b.py` | Main generator |
| `tools/run.sh` | Start, stop, status |
| `tools/build_clip_queue.py` | Build work queue (vla_golden first) |
| `tools/validate_dataset.py` | Integrity check (safe to run during generation) |
| `tools/compact_dataset.py` | Deduplicate and compact shards |
| `tools/bench_cameras.py` | Measure inference latency by camera count |
| `tools/train_student_smoke.py` | Student training validation |
| `tools/generate_coc_10b_legacy.py` | Alpamayo 1.5 (10B) version, kept for reference |
| `pyfix/sitecustomize.py` | Workaround for huggingface.co connectivity failures |

## Usage

Set up environment variables first. Copy `env.sh.example`, fill in your HF token, and save it as `env.sh`.

```bash
source /path/to/alpamayo/env.sh

./tools/run.sh queue     # build clip queue (once)
./tools/run.sh start     # start generation
./tools/run.sh status    # progress
./tools/run.sh stop      # stop and release GPUs
```

`stop` is safe at any time. Each worker flushes its in memory rows before exiting, and `start` resumes from where it left off. Even after a hard kill, only work since the last periodic flush (every 5 minutes) is lost.

To use a subset of GPUs:

```bash
NUM_GPUS=3 ./tools/run.sh start          # only 3 GPUs
WORKERS_PER_GPU=1 ./tools/run.sh start   # one worker per GPU
```

## Design decisions and evidence

Every value below came from measurement. Nothing was chosen by guesswork.

### Store all 7 cameras

The trajectory task uses cameras (0,1,2,3,5,6) and vqa uses (0,1,2,3,4,5). The union is all 7. Inference receives only the subset a task requires, but everything is stored.

Dropping cameras at training time is always possible; adding cameras that were never stored is not. This project actually discarded an entire dataset over this. The first version was built with the 10B model, which uses 4 cameras, and it was incompatible with the 34B, which uses 6.

The storage difference between 7 and 6 cameras is about 3GB.

### Encode JPEG first, then decode it for inference

Order matters. Running inference on the original frames and then storing a JPEG means the stored image and the stored CoC came from different inputs.

Controlled experiment:

| Comparison | CoC match | Max ADE difference |
| :-- | :-- | :-- |
| original vs original (same seed, rerun) | 6/6 | 0.0000 |
| original vs resize only | 6/6 | 0.06 |
| resize vs JPEG q92 | 5/6 | 1.14 |

JPEG compression alone flips 1 of 6 CoC outputs. Encoding first makes the dataset bit exact reproducible from stored contents. Reverification showed CoC 6/6 match and ADE difference 0.0000.

### t0 spacing of 2 seconds

A/B test holding the logprob setting constant:

| Spacing | t0/h | clips/h |
| :-- | :-- | :-- |
| 1 second | 1,278 | 78 |
| 2 seconds | 1,371 | 152 |

2 seconds wins on both metrics. An earlier comparison appeared to favor 1 second, but that comparison had logprob capture as a confounded second variable.

### num_traj_samples of 6

k=12 uses 95GB of GPU memory, which exceeds the budget for 2 workers per GPU (183GB). Text diversity is covered by the token distributions instead, so 6 is sufficient.

### top-k of 20

Whether to increase k was investigated and rejected. Measured over 38,430 tokens:

| k | Cumulative probability |
| :-- | :-- |
| 20 | 0.8195 (measured) |
| 50 | 0.8214 (extrapolated) |
| 100 | 0.8215 (extrapolated) |

Probability concentrates in ranks 1 and 2 and then scatters into a flat tail. The 18 percent held by the 150,000 tokens below rank 20 amounts to roughly 1e-6 each, so a larger k cannot capture it. Regenerating would cost 9 GPU hours for a 0.2 percentage point gain.

Per token, 79 percent of positions have 99 percent or more captured by top 20. The 20 percent that drags the average down are genuine decision branch points where the teacher is uncertain.

## Performance

Measured with 8 workers across 4 B200 GPUs.

| Metric | Value |
| :-- | :-- |
| Throughput | about 1,500 t0/h, 168 clips/h |
| Size | 1.147 MB per t0 |
| Prompt | 4,580 tokens (6 cameras x 4 frames) |
| Inference | 2.8 seconds per t0 (6 samples) |

The GPU is not the bottleneck. Phase timing:

| Phase | Share |
| :-- | :-- |
| tokenize (includes image preprocessing) | 19 percent |
| decode (video) | 22 percent |
| open (network streaming) | 20 percent |
| infer (GPU) | 15 percent |
| encode (JPEG) | 6 percent |

JPEG encoding was initially assumed to be the bottleneck and was parallelized with a thread pool, but measurement showed it was only 6 percent. The real bottleneck was tokenize at 39 percent, caused by `helper.prepare_model_inputs` calling `AutoProcessor.from_pretrained` on every t0. Caching the processor once dropped it from 39 to 19 percent and raised GPU utilization from 18 to 62 percent.

## Inference latency by camera count

Measured for Thor deployment planning. B200, num_traj_samples=1.

| Cameras | Images | Tokens | Latency | Speedup |
| :-- | :-- | :-- | :-- | :-- |
| 1 | 4 | 838 | 0.591s | 2.38x |
| 2 | 8 | 1,585 | 0.913s | 1.54x |
| 3 | 12 | 2,333 | 1.033s | 1.36x |
| 4 | 16 | 3,082 | 1.211s | 1.16x |
| 6 | 24 | 4,580 | 1.408s | 1.00x |

Token count is exactly linear in camera count, at 187 tokens per image.

Latency is not linear. A 5.5x increase in tokens produces only a 2.4x increase in latency. Fixed costs (diffusion expert, CoC decoding) dominate, so even a 4 image configuration takes 0.59 seconds.

The implication is that reducing cameras alone cannot reach real time (10Hz). Going from 6 cameras to 3 gives 1.36x when 14x is needed. Model size reduction is mandatory; camera reduction is secondary.

## Student training validation

Once the teacher is gone the dataset cannot be regenerated, so the format was verified against a real training path while server time remained.

An Alpamayo2Super model was constructed with a Qwen3-VL-2B backbone and trained on actual data:

```
pretrained load: full copy 624, partial copy 2 (vocab expansion), skipped 0
step  1: loss 38.33
step  5: loss 26.59
step 10: loss 21.13
step 15: loss 19.47
```

No pretrained weights were dropped (skipped 0) and loss decreases monotonically.

Two obstacles are worth recording.

First, the teacher config cannot be loaded and edited to swap the backbone. Its `vlm_config` is already fixed to the 34B and `traj_ids` are computed against the 34B tokenizer. Passing only `vlm_name_or_path` and constructing from scratch makes the config recompute tokenizer expansion and traj_ids for the new backbone.

Second, `helper.prepare_model_inputs` hardcodes `generation_mode=True` and cannot be used for training. Generation mode removes the target from the sequence, so there are zero future trajectory placeholders and `fuse_traj_tokens` fails trying to insert 128. Training requires calling `build_conversation` directly with `generation_mode=False`.

The training forward pass does not use the diffusion expert. Trajectories are fused as discrete tokens and learned with next token loss alongside the text. The expert refines those tokens into continuous trajectories at inference time.

## Operational notes

### Network workaround is required

On this node `huggingface.co` frequently fails. It resolves through CloudFront anycast to several IPs, many of which silently drop SYN packets. Python's default connect timeout is unlimited, so hitting a dead IP hangs forever with no error.

`pyfix/sitecustomize.py` is applied automatically through `PYTHONPATH`. It works because CloudFront edges route by SNI, so connecting to a live HF edge (such as `cdn-lfs.hf.co`) while presenting `huggingface.co` as SNI returns a valid response. No IP is hardcoded, so it keeps working when edges change.

To confirm the symptom, run `ss -tnp | grep <PID>` and look for `SYN-SENT`.

### Validation and compaction

```bash
# validation (safe during generation, niced down)
nice -n 15 python tools/validate_dataset.py

# compaction (must stop first)
./tools/run.sh stop
python tools/compact_dataset.py           # dry run
python tools/compact_dataset.py --apply   # apply
./tools/run.sh start
```

Compaction writes new files, verifies them, and only then deletes the originals. If it fails midway the originals remain.

Running experiments in a separate output directory and merging them back can introduce duplicates. Using the same output directory lets the resume logic prevent this.

### Split train and val by clip_id, not by t0

t0 spacing is 2 seconds while the prediction horizon is 6.4 seconds, so adjacent t0 windows share about 70 percent of their future trajectory. Splitting by t0 puts nearly identical scenes in both sets and inflates measured performance.

## Reading the data

```python
import pyarrow.parquet as pq, glob, numpy as np, io
from PIL import Image

D = "path/to/coc_34b_v1"
s = pq.read_table(sorted(glob.glob(f"{D}/samples/*.parquet"))).to_pandas()
f = pq.read_table(sorted(glob.glob(f"{D}/frames/*.parquet"))).to_pandas()

r = s.iloc[0]
traj = np.array(r.pred_xyz).reshape(64, 3)
dist = np.array(r.topk_logprobs).reshape(r.num_gen_tokens, r.topk_k)

fr = f[(f.clip_id == r.clip_id) & (f.t0_us == r.t0_us)].iloc[0]
imgs = [np.array(Image.open(io.BytesIO(b))) for b in fr.jpegs]
```

Camera order is always 0 through 6: cross_left, front_wide, cross_right, rear_left, rear_tele, rear_right, front_tele. Each camera contributes 4 consecutive frames.

`pass_filter` is an advisory flag marking whether `ade <= 1.0m`. The raw ADE is stored, so the threshold can be changed later without regenerating anything.

## Related repositories

This code depends on the following NVIDIA repositories.

1. NVlabs/alpamayo2 : 34B inference code
2. NVlabs/alpamayo-recipes : SFT and RL post training recipes (for Alpamayo 1.5)
3. nvidia/Alpamayo2-Super : model weights (gated)
4. nvidia/PhysicalAI-Autonomous-Vehicles : source driving data (gated)
