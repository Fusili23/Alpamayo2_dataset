#!/usr/bin/env bash
# Alpamayo 34B CoC 데이터셋 생성 — 시작 / 중단 / 상태
#
# GPU를 다른 사람이 써야 할 수 있으므로 언제든 안전하게 멈출 수 있다.
#   ./run.sh stop    -> SIGTERM. 각 워커가 shard를 flush하고 종료한다.
#   ./run.sh start   -> 남은 (clip_id, t0_us)부터 이어서 진행한다.
# 강제 종료(kill -9)로 죽어도 마지막 flush 이후 분량만 잃고 재개된다.

set -euo pipefail

BIG=${BIG:-/NHNHOME/WORKSPACE/0526050025_A/alpamayo}
source "$BIG/env.sh"

OUT=${OUT:-$BIG/data/coc_34b_v1}
QUEUE=${QUEUE:-$BIG/data/clip_queue.parquet}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-2}   # 34B는 69GB -> GPU(183GB)당 2개
NUM_GPUS=${NUM_GPUS:-4}
NUM_WORKERS=$((WORKERS_PER_GPU * NUM_GPUS))
VENV=/NHNHOME/venvs/alpamayo2
PATTERN="generate_coc_34b.py"

# uv run 래퍼가 아니라 실제 python 프로세스만 센다 (래퍼까지 세면 2배로 보인다)
worker_pids() { pgrep -f "^$VENV/bin/python3 -u $BIG/tools/$PATTERN" 2>/dev/null || true; }

case "${1:-status}" in

start)
  if [ -n "$(worker_pids)" ]; then
    echo "이미 실행 중입니다. 먼저 './run.sh stop'"; exit 1
  fi
  [ -f "$QUEUE" ] || { echo "클립 큐가 없습니다: $QUEUE"; echo "먼저: ./run.sh queue"; exit 1; }
  mkdir -p "$OUT" "$BIG/logs"
  cd "$BIG/alpamayo2"
  echo "워커 $NUM_WORKERS개 시작 (GPU당 $WORKERS_PER_GPU개)"
  for w in $(seq 0 $((NUM_WORKERS - 1))); do
    UV_PROJECT_ENVIRONMENT=$VENV nohup uv run python -u "$BIG/tools/generate_coc_34b.py" \
      --out "$OUT" --clips-file "$QUEUE" \
      --num-workers "$NUM_WORKERS" --worker-id "$w" --gpu $((w % NUM_GPUS)) \
      > "$BIG/logs/gen34_w$w.log" 2>&1 &
    sleep 0.3
  done
  echo "시작됨. 상태: ./run.sh status"
  ;;

stop)
  pids=$(worker_pids)
  if [ -z "$pids" ]; then echo "실행 중인 워커가 없습니다."; exit 0; fi
  echo "SIGTERM 전송 — 각 워커가 shard flush 후 종료합니다..."
  # shellcheck disable=SC2086
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 60); do
    [ -z "$(worker_pids)" ] && break
    sleep 1
  done
  if [ -n "$(worker_pids)" ]; then
    echo "60초 내 미종료 — SIGKILL (마지막 flush 이후 분량만 손실, 재개 가능)"
    # shellcheck disable=SC2086
    kill -9 $(worker_pids) 2>/dev/null || true
  fi
  echo "정지 완료. GPU 반납됨:"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  ;;

status)
  n=$(worker_pids | wc -w)
  echo "실행 중인 워커: $n"
  if [ -d "$OUT/samples" ]; then
    t0=$(ls "$OUT"/frames/*.parquet 2>/dev/null | wc -l)
    echo "frames shard: $t0   총 용량: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
  fi
  echo "--- 워커별 마지막 진행 ---"
  for f in "$BIG"/logs/gen34_w*.log; do
    [ -f "$f" ] || continue
    printf "%s: %s\n" "$(basename "$f" .log)" "$(grep '누적 t0' "$f" 2>/dev/null | tail -1 | cut -c1-110)"
  done
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  ;;

queue)
  cd "$BIG/alpamayo2"
  UV_PROJECT_ENVIRONMENT=$VENV uv run python "$BIG/tools/build_clip_queue.py" --out "$QUEUE"
  ;;

*)
  echo "사용법: $0 {start|stop|status|queue}"; exit 1
  ;;
esac
