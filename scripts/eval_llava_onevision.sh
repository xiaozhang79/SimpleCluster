set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-8}
MODEL_PATH=${MODEL_PATH:-llava-hf/llava-onevision-qwen2-7b-ov-hf}
RATIOS=${RATIOS:-"0.01 0.05 0.10 0.15 0.20 0.25"}
MASTER_PORT=${MASTER_PORT:-29500}
export PYTHONPATH="$ROOT:$ROOT/LLaVA-NeXT${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%6N)
LOG_PATH="$ROOT/logs/performance/llava_onevision/$RUN_ID.log"
RUN_ROOT="$ROOT/logs/runs/llava_onevision_$RUN_ID"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG_PATH")"

for RATIO in $RATIOS; do
  LABEL=$(python -c 'import sys; print(round(float(sys.argv[1]) * 100))' "$RATIO")
  RATIO_DIR="$RUN_ROOT/evaluation/llava_onevision/r$LABEL"
  mkdir -p "$RATIO_DIR"
  python -m accelerate.commands.launch --multi_gpu --num_processes "$NUM_PROCESSES" \
    --main_process_port "$MASTER_PORT" --mixed_precision no \
    -m lmms_eval \
    --model llava_onevision_simplecluster \
    --model_args "pretrained=$MODEL_PATH,model_name=llava_qwen,max_frames_num=32,attn_implementation=flash_attention_2,retention_ratio=$RATIO,token_audit_path=$RATIO_DIR/token_audit.jsonl" \
    --tasks mvbench,egoschema,longvideobench_val_v,videomme --output_path "$RATIO_DIR" \
    2>&1 | tee "$RATIO_DIR/console.txt"
  RESULT_JSON=$(find "$RATIO_DIR" -type f -name '*_results.json' -print -quit)
  python -m simplecluster.result_logs --run-root "$RUN_ROOT" --model llava_onevision \
    --ratio-percent "$LABEL" --result-json "$RESULT_JSON" --log-path "$LOG_PATH"
done
