#!/usr/bin/env bash
# 全量官方 BI dev-gate（4090）：
#   merge DM-large → base+large 各評 500 internal-dev（BI 尺）→ score（上傳 HF）→ gate。
# gate 規則：large_fallback - base_fallback >= 1pp → 過（exit 0，值得再燒三 seed formal）；否則 exit 3。
# ⚠️ 先跑過 smoke_eval.sh 確認環境，再跑這支。重跑前把 $WORK 清掉（gen 拒絕覆寫）。
set -euo pipefail
VENV="${VENV:-$HOME/dmeval}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/code" && pwd)"
DEV="${DEV:-$HOME/dm_prep/prep/math_l45_internal_dev.jsonl}"
ADAPTER="${ADAPTER:-Yuivdldk/math-model-dm-large-l45-g4b-lora-lr0.0001-ep1-s42-n50000-dd45cf97b90}"
WORK="${WORK:-$HOME/dmeval_run}"
MERGED="${MERGED:-$HOME/dmeval_merged_large}"
BASE_REPO="${BASE_REPO:-mathvl-l45-dmdev-base-g4b-mn4096bi-n500}"
LARGE_REPO="${LARGE_REPO:-mathvl-l45-dmdev-large-g4b-mn4096bi-n500}"
TAG=math_l45_internal_dev
: "${HF_TOKEN:?export HF_TOKEN}"
[ -f "$DEV" ] || { echo "[FATAL] internal-dev 不在 $DEV"; exit 1; }
n=$(wc -l < "$DEV"); [ "$n" = "500" ] || { echo "[FATAL] internal-dev 應為 500 行，got $n"; exit 1; }
mkdir -p "$WORK"; cd "$CODE"

echo "== 1/6 merge DM-large（bf16，CPU）=="
rm -rf "$MERGED"
BASE=google/gemma-3-4b-it BASE_REVISION=093f9f388b31de276ce2de164bdc2081324b9767 \
  ADAPTER="$ADAPTER" OUT="$MERGED" REQUIRE_TRAINING_PROVENANCE=1 \
  "$VENV/bin/python" merge_lora.py

echo "== 2/6 gen base（BI, 500 題；最花時間，~數十分鐘）=="
MODEL=google/gemma-3-4b-it MODEL_REVISION=093f9f388b31de276ce2de164bdc2081324b9767 \
  EVAL_SETS="$DEV" OUT_PREFIX="$WORK/gen_base" \
  VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VENV/bin/python" vllm_gen_math.py

echo "== 3/6 gen large（BI, 500 題）=="
MODEL="$MERGED" EVAL_SETS="$DEV" OUT_PREFIX="$WORK/gen_large" \
  VLLM_BATCH_INVARIANT=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VENV/bin/python" vllm_gen_math.py

echo "== 4/6 score base（上傳 HF，private）=="
EVAL_SET="$DEV" GEN="$WORK/gen_base_${TAG}.jsonl" KIND=baseline \
  REPO="$BASE_REPO" MODEL_LABEL="gemma-3-4b-it base (4090 BI dev)" UPLOAD=1 PUBLIC=0 \
  "$VENV/bin/python" score_math_vllm.py

echo "== 5/6 score large（上傳 HF，private）=="
EVAL_SET="$DEV" GEN="$WORK/gen_large_${TAG}.jsonl" KIND=finetuned \
  REPO="$LARGE_REPO" MODEL_LABEL="DM-large (4090 BI dev)" UPLOAD=1 PUBLIC=0 \
  "$VENV/bin/python" score_math_vllm.py

echo "== 6/6 dev gate（large - base >= 1pp）=="
set +e
"$VENV/bin/python" gate_deepmath_dev.py \
  --internal-dev "$DEV" \
  --baseline "$HOME/vres_${BASE_REPO}/eval_records.json" \
  --large-seed42 "$HOME/vres_${LARGE_REPO}/eval_records.json" \
  --out "$WORK/dev_gate.json"
GATE_RC=$?
set -e
echo ""
echo "==================== 結果 ===================="
echo "dev_gate.json：$WORK/dev_gate.json"
echo "base/large 結果已上傳：Yuivdldk/${BASE_REPO}  與  Yuivdldk/${LARGE_REPO}"
if [ "$GATE_RC" = "0" ]; then
  echo "GATE：✅ 通過（large - base >= 1pp）→ DeepMath 有搞頭，值得再燒三 seed formal。"
else
  echo "GATE：❌ 未過（large - base < 1pp，gate exit=$GATE_RC）→ 就此止損，別燒 formal。"
fi
echo "把 dev_gate.json 裡的 baseline.fallback_acc / large.fallback_acc / delta 回報我。"
