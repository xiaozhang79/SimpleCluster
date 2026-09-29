set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-8}
MODEL_PATH=${MODEL_PATH:-LanguageBind/LLaVA-Video-7B-Qwen2}
RATIOS=${RATIOS:-"0.01 0.05 0.10 0.15 0.20 0.25"}
MASTER_PORT=${MASTER_PORT:-29500}
export PYTHONPATH="$ROOT:$ROOT/LLaVA-NeXT${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S_%6N)
LOG_PATH="$ROOT/logs/performance/llava_video/$RUN_ID.log"
RUN_ROOT="$ROOT/logs/runs/llava_video_$RUN_ID"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG_PATH")"

for RATIO in $RATIOS; do
  LABEL=$(python -c 'import sys; print(round(float(sys.argv[1]) * 100))' "$RATIO")
  RATIO_DIR="$RUN_ROOT/evaluation/llava_video/r$LABEL"
  mkdir -p "$RATIO_DIR"
  python -m accelerate.commands.launch --multi_gpu --num_processes "$NUM_PROCESSES" \
    --main_process_port "$MASTER_PORT" --mixed_precision no \
    -m lmms_eval \
    --model llava_video_simplecluster \
    --model_args "pretrained=$MODEL_PATH,model_name=llava_qwen,conv_template=qwen_1_5,video_decode_backend=decord,video_fps=1,max_frames_num=64,force_sample=True,add_time_instruction=False,mm_spatial_pool_mode=average,mm_resampler_location=before,mm_newline_position=grid,attn_implementation=flash_attention_2,retention_ratio=$RATIO,token_audit_path=$RATIO_DIR/token_audit.jsonl" \
    --tasks mvbench,egoschema,longvideobench_val_v,videomme --output_path "$RATIO_DIR" \
    2>&1 | tee "$RATIO_DIR/console.txt"
  RESULT_JSON=$(find "$RATIO_DIR" -type f -name '*_results.json' -print -quit)
  python -m simplecluster.result_logs --run-root "$RUN_ROOT" --model llava_video \
    --ratio-percent "$LABEL" --result-json "$RESULT_JSON" --log-path "$LOG_PATH"
done
