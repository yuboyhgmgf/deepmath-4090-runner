#!/usr/bin/env bash
# 單臂 eval：一台 4070 只跑一臂。
#   ARM=base  → gen+score+upload base（不 merge）
#   ARM=large → merge DM-large → gen+score+upload large
# 兩台各跑一臂、各自上傳 HF；之後任一台跑 run_gate.sh 收尾算 gate。
set -euo pipefail
ARM="${ARM:?請設 ARM=base 或 ARM=large}"
VENV="${VENV:-$HOME/dmeval}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/code" && pwd)"
DEV="${DEV:-$HOME/dm_prep/prep/math_l45_internal_dev.jsonl}"
ADAPTER="${ADAPTER:-Yuivdldk/math-model-dm-large-l45-g4b-lora-lr0.0001-ep1-s42-n50000-dd45cf97b90}"
BASE_REPO="${BASE_REPO:-mathvl-l45-dmdev-base-g4b-mn4096bi-n500}"
LARGE_REPO="${LARGE_REPO:-mathvl-l45-dmdev-large-g4b-mn4096bi-n500}"
GPU_UTIL="${GPU_UTIL:-0.88}"   # 4070 12GB 較緊；若 OOM 調到 0.80 再試
WORK="${WORK:-$HOME/dmeval_${ARM}}"
MERGED="${MERGED:-$HOME/dmeval_merged_large}"
TAG=math_l45_internal_dev
: "${HF_TOKEN:?export HF_TOKEN}"
[ -f "$DEV" ] || { echo "[FATAL] internal-dev 不在 $DEV"; exit 1; }
n=$(wc -l < "$DEV"); [ "$n" = "500" ] || { echo "[FATAL] internal-dev 應 500 行, got $n"; exit 1; }
mkdir -p "$WORK"; cd "$CODE"

if [ "$ARM" = "base" ]; then
  MODEL_ARGS=(MODEL=google/gemma-3-4b-it MODEL_REVISION=093f9f388b31de276ce2de164bdc2081324b9767)
  REPO="$BASE_REPO"; KIND=baseline; LABEL="gemma-3-4b-it base (4070 BI)"
elif [ "$ARM" = "large" ]; then
  echo "== merge DM-large（bf16, CPU）=="
  rm -rf "$MERGED"
  BASE=google/gemma-3-4b-it BASE_REVISION=093f9f388b31de276ce2de164bdc2081324b9767 \
    ADAPTER="$ADAPTER" OUT="$MERGED" REQUIRE_TRAINING_PROVENANCE=1 \
    "$VENV/bin/python" merge_lora.py
  MODEL_ARGS=(MODEL="$MERGED")
  REPO="$LARGE_REPO"; KIND=finetuned; LABEL="DM-large (4070 BI)"
else
  echo "[FATAL] ARM 只能是 base 或 large"; exit 1
fi

echo "== gen $ARM（BI, 500 題, GPU_UTIL=$GPU_UTIL）=="
env "${MODEL_ARGS[@]}" EVAL_SETS="$DEV" OUT_PREFIX="$WORK/gen_${ARM}" GPU_UTIL="$GPU_UTIL" \
  VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VENV/bin/python" vllm_gen_math.py

echo "== score $ARM（上傳 HF, private）=="
EVAL_SET="$DEV" GEN="$WORK/gen_${ARM}_${TAG}.jsonl" KIND="$KIND" \
  REPO="$REPO" MODEL_LABEL="$LABEL" UPLOAD=1 PUBLIC=0 \
  "$VENV/bin/python" score_math_vllm.py

echo ""
echo "ARM_DONE ✅ [$ARM] 已上傳 → Yuivdldk/$REPO"
echo "兩臂都跑完後，任一台跑： bash run_gate.sh"
