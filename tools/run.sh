#!/usr/bin/env bash
# Alpamayo 34B CoC dataset generation: start / stop / status
#
# Someone else may need the GPUs, so this can be stopped safely at any time.
#   ./run.sh stop    sends SIGTERM. Each worker flushes its shards and exits.
#   ./run.sh start   resumes from the remaining (clip_id, t0_us) keys.
# Even after kill -9, only work since the last flush is lost and it resumes.

set -euo pipefail

BIG=${BIG:-/NHNHOME/WORKSPACE/0526050025_A/alpamayo}
source "$BIG/env.sh"

OUT=${OUT:-$BIG/data/coc_34b_v1}
QUEUE=${QUEUE:-$BIG/data/clip_queue.parquet}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}   # the 34B is 69GB, so 2 per GPU (183GB)
NUM_GPUS=${NUM_GPUS:-4}
NUM_WORKERS=$((WORKERS_PER_GPU * NUM_GPUS))
VENV=/NHNHOME/venvs/alpamayo2
PATTERN="generate_coc_34b.py"

# count only real python processes, not the uv run wrappers (which would double it)
worker_pids() { pgrep -f "^$VENV/bin/python3 -u $BIG/tools/$PATTERN" 2>/dev/null || true; }

case "${1:-status}" in

start)
  if [ -n "$(worker_pids)" ]; then
    echo "Already running. Run './run.sh stop' first."; exit 1
  fi
  [ -f "$QUEUE" ] || { echo "No clip queue at: $QUEUE"; echo "Run ./run.sh queue first"; exit 1; }
  mkdir -p "$OUT" "$BIG/logs"
  cd "$BIG/alpamayo2"
  echo "starting $NUM_WORKERS workers ($WORKERS_PER_GPU per GPU)"
  for w in $(seq 0 $((NUM_WORKERS - 1))); do
    UV_PROJECT_ENVIRONMENT=$VENV nohup uv run python -u "$BIG/tools/generate_coc_34b.py" \
      --out "$OUT" --clips-file "$QUEUE" \
      --num-workers "$NUM_WORKERS" --worker-id "$w" --gpu $((w % NUM_GPUS)) \
      > "$BIG/logs/gen34_w$w.log" 2>&1 &
    sleep 0.3
  done
  echo "started. status: ./run.sh status"
  ;;

stop)
  pids=$(worker_pids)
  if [ -z "$pids" ]; then echo "No workers running."; exit 0; fi
  echo "sending SIGTERM, each worker flushes shards then exits..."
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 60); do
    [ -z "$(worker_pids)" ] && break
    sleep 1
  done
  if [ -n "$(worker_pids)" ]; then
    echo "still alive after 60s, sending SIGKILL (only work since last flush is lost)"
    # shellcheck disable=SC2086
    kill -9 $(worker_pids) 2>/dev/null || true
  fi
  echo "stopped. GPUs released:"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  ;;

status)
  n=$(worker_pids | wc -w)
  echo "running workers: $n"
  if [ -d "$OUT/samples" ]; then
    t0=$(ls "$OUT"/frames/*.parquet 2>/dev/null | wc -l)
    echo "frames shard: $t0   total size: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
  fi
  echo "--- last progress per worker ---"
  for f in "$BIG"/logs/gen34_w*.log; do
    [ -f "$f" ] || continue
    printf "%s: %s\n" "$(basename "$f" .log)" "$(grep 'cum t0' "$f" 2>/dev/null | tail -1 | cut -c1-110)"
  done
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  ;;

queue)
  cd "$BIG/alpamayo2"
  UV_PROJECT_ENVIRONMENT=$VENV uv run python "$BIG/tools/build_clip_queue.py" --out "$QUEUE"
  ;;

*)
  echo "usage: $0 {start|stop|status|queue}"; exit 1
  ;;
esac
