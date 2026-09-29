set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-8}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-7b-ov-hf}
RATIOS=${RATIOS:-"0.01 0.05 0.10 0.15 0.20 0.25"}
MASTER_PORT=${MASTER_PORT:-29501}
export PYTHONPATH="$ROOT:$ROOT/LLaVA-NeXT${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%6N)
LOG_PATH="$ROOT/logs/feature_analysis/llava_onevision/$RUN_ID.log"
RUN_ROOT="$ROOT/logs/runs/feature_analysis_$RUN_ID"
mkdir -p "$(dirname "$LOG_PATH")"

for RATIO in $RATIOS; do
  LABEL=$(python -c 'import sys; print(round(float(sys.argv[1]) * 100))' "$RATIO")
  RATIO_DIR="$RUN_ROOT/feature_analysis/r$LABEL"
  mkdir -p "$RATIO_DIR"
  python -m accelerate.commands.launch --multi_gpu --num_processes "$NUM_PROCESSES" \
    --main_process_port "$MASTER_PORT" --mixed_precision no \
    -m simplecluster.feature_analysis \
    --model-path "$MODEL_PATH" --ratio-percent "$LABEL" \
    --run-root "$RUN_ROOT" --log-path "$LOG_PATH" \
    --output-dir "$RATIO_DIR/records" \
    2>&1 | tee "$RATIO_DIR/console.txt"
done
