#!/usr/bin/env bash
# 4 題 base-only BI smoke。只為證明：4090 上 vLLM BI 尺能【載入→生成→計分】，
# 且 cuda_runtime_library provenance 有被記到（scorer 會擋沒有的）。
# 這步過了，才值得跑全量 run_dev_gate.sh。不上傳、不評 large。
set -euo pipefail
VENV="${VENV:-$HOME/dmeval}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/code" && pwd)"
DEV="${DEV:-$HOME/dm_prep/prep/math_l45_internal_dev.jsonl}"
WORK="${WORK:-$HOME/dmeval_smoke}"
: "${HF_TOKEN:?export HF_TOKEN（scorer selftest / 之後上傳會用；smoke 不上傳但仍需在環境）}"
[ -f "$DEV" ] || { echo "[FATAL] internal-dev 不在 $DEV —— 先跑過訓練那包的 setup_and_build.sh（它會下載 prep/math_l45_internal_dev.jsonl）"; exit 1; }

rm -rf "$WORK"; mkdir -p "$WORK"
head -4 "$DEV" > "$WORK/math_l45_smoke4.jsonl"
echo "== smoke 樣本：$(wc -l < "$WORK/math_l45_smoke4.jsonl") 題 =="

cd "$CODE"
echo "== 1/2 gen（base, 4 題, BI 尺）=="
MODEL=google/gemma-3-4b-it MODEL_REVISION=093f9f388b31de276ce2de164bdc2081324b9767 \
  EVAL_SETS="$WORK/math_l45_smoke4.jsonl" OUT_PREFIX="$WORK/gen_smoke" \
  VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VENV/bin/python" vllm_gen_math.py

echo "== 2/2 score（不上傳；驗 BI provenance + math-verify selftest）=="
EVAL_SET="$WORK/math_l45_smoke4.jsonl" GEN="$WORK/gen_smoke_math_l45_smoke4.jsonl" \
  KIND=baseline REPO=dmdev-smoke-local MODEL_LABEL="smoke base" UPLOAD=0 \
  "$VENV/bin/python" score_math_vllm.py

echo ""
echo "SMOKE_OK ✅ —— BI 尺在你 4090 上可載入/生成/計分。可以跑 run_dev_gate.sh 全量。"
