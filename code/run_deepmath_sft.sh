#!/usr/bin/env bash
# Guarded DeepMath -> Gemma-3-4B LoRA launcher.  This trains only; merge and all
# reported evaluation use merge_lora.py + the final vLLM ruler.
set -euo pipefail

MODE="${MODE:-smoke}"                 # smoke | development | formal
ARM="${ARM:?set ARM=small or ARM=large}"
TRAIN_DATA="${TRAIN_DATA:?set TRAIN_DATA to dm_small_train.jsonl or dm_large_train.jsonl}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$ARM" in
  small|large) ;;
  *) echo "[FATAL] ARM must be small or large" >&2; exit 2 ;;
esac
case "$MODE" in
  smoke|development|formal) ;;
  *) echo "[FATAL] MODE must be smoke, development, or formal" >&2; exit 2 ;;
esac
case "$SEED" in
  42|43|44) ;;
  *) echo "[FATAL] SEED must be 42, 43, or 44" >&2; exit 2 ;;
esac
if [[ "$MODE" == "development" && "$SEED" != "42" ]]; then
  echo "[FATAL] registered development gate uses seed 42 only" >&2
  exit 2
fi
if [[ "$MODE" == "development" && "$ARM" == "small" ]]; then
  echo "[FATAL] small development is not registered; prove the large arm first" >&2
  exit 2
fi
if [[ "$MODE" == "formal" ]]; then
  DEV_GATE="${DEV_GATE:?formal training requires DEV_GATE=path/to/dev_gate.json}"
  if [[ ! -f "$DEV_GATE" ]]; then
    echo "[FATAL] missing development gate artifact: $DEV_GATE" >&2
    exit 2
  fi
  "$PYTHON_BIN" - "$DEV_GATE" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("stage") != "complete" or gate.get("kind") != "deepmath_internal_dev_gate":
    raise SystemExit("[FATAL] invalid development gate artifact")
if gate.get("analysis_stage") != "large_only":
    raise SystemExit("[FATAL] development gate is not the registered large-only gate")
if gate.get("gate_passed") is not True or gate.get("formal_test_authorized") is not True:
    raise SystemExit("[FATAL] development gate did not authorize formal work")
PY
  if [[ "$ARM" == "small" ]]; then
    LARGE_CONFIRM="${LARGE_CONFIRM:?small formal training requires LARGE_CONFIRM=path/to/large_confirm.json}"
    if [[ ! -f "$LARGE_CONFIRM" ]]; then
      echo "[FATAL] missing large-only confirmation artifact: $LARGE_CONFIRM" >&2
      exit 2
    fi
    "$PYTHON_BIN" - "$LARGE_CONFIRM" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    value.get("stage") != "complete"
    or value.get("kind") != "deepmath_formal_confirmation"
    or value.get("formal") is not True
    or value.get("analysis_stage") != "large_only"
):
    raise SystemExit("[FATAL] invalid large-only confirmation artifact")
if (value.get("claims") or {}).get("deepmath_distillation_improves_capability") is not True:
    raise SystemExit("[FATAL] large arm did not pass the registered +2pp/CI rule")
PY
  fi
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "[FATAL] missing TRAIN_DATA: $TRAIN_DATA" >&2
  exit 2
fi

BUILD_DIR="$(cd "$(dirname "$TRAIN_DATA")" && pwd)"
SCHEDULE_SUMMARY="${SCHEDULE_SUMMARY:-$BUILD_DIR/deepmath_schedule_summary.json}"
SCHEDULE_MANIFEST="${SCHEDULE_MANIFEST:-$BUILD_DIR/dm_${ARM}_schedule_manifest.jsonl}"
SCHEDULE_VALIDATION="${SCHEDULE_VALIDATION:-$BUILD_DIR/dm_${ARM}_schedule_validation.json}"
for required in "$SCHEDULE_SUMMARY" "$SCHEDULE_MANIFEST"; do
  if [[ ! -f "$required" ]]; then
    echo "[FATAL] missing registered schedule artifact: $required" >&2
    exit 2
  fi
done
"$PYTHON_BIN" "$SCRIPT_DIR/validate_deepmath_schedule.py" \
  --arm "$ARM" \
  --train-data "$TRAIN_DATA" \
  --schedule-manifest "$SCHEDULE_MANIFEST" \
  --summary "$SCHEDULE_SUMMARY" \
  --out "$SCHEDULE_VALIDATION"

rows="$(wc -l < "$TRAIN_DATA")"
if [[ "$MODE" == "smoke" ]]; then
  if (( rows < 1000 )); then
    echo "[FATAL] smoke requires at least 1000 schedule rows, got $rows" >&2
    exit 2
  fi
  N_TRAIN=1000
  # smoke 一律不上傳：硬釘不可被殘留的環境變數（如跑過 formal 後遺留 UPLOAD_MODEL=1）覆寫。
  export UPLOAD_RESULTS="0"
  export UPLOAD_MODEL="0"
  export CKPT_RESUME="0"
else
  if (( rows != 50000 )); then
    echo "[FATAL] development/formal schedule must contain exactly 50000 rows, got $rows" >&2
    exit 2
  fi
  N_TRAIN=50000
  export UPLOAD_RESULTS="${UPLOAD_RESULTS:-1}"
  export UPLOAD_MODEL="${UPLOAD_MODEL:-1}"
  for upload_value in "$UPLOAD_RESULTS" "$UPLOAD_MODEL"; do
    case "${upload_value,,}" in
      1|true|yes|on) ;;
      *)
        echo "[FATAL] development/formal DeepMath runs require model+result uploads for remote checkpoint recovery" >&2
        exit 2
        ;;
    esac
  done
  export CKPT_RESUME="1"
  export TRAIN_SAVE_STEPS="${TRAIN_SAVE_STEPS:-500}"
fi

export TRAIN_BS="${TRAIN_BS:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
if (( TRAIN_BS * GRAD_ACCUM != 8 )); then
  echo "[FATAL] effective batch must equal 8; TRAIN_BS=$TRAIN_BS GRAD_ACCUM=$GRAD_ACCUM" >&2
  exit 2
fi

export MODEL="google/gemma-3-4b-it"
export MODEL_REVISION="093f9f388b31de276ce2de164bdc2081324b9767"
export METHODS="lora"
export LRS_LORA="1e-4"
export EPOCHS="1"
export TRAIN_DATA
export SCHEDULE_VALIDATION
export N_TRAIN
export SEED
if [[ "$MODE" == "smoke" ]]; then
  export DATA_TAG="${DATA_TAG:-deepmath-smoke-${ARM}}"
else
  # Development seed42 is already the exact registered formal recipe.  Keeping
  # one stable tag lets formal seed42 reuse that verified adapter instead of
  # spending another identical training run.
  export DATA_TAG="${DATA_TAG:-deepmath-${ARM}}"
fi
export TRAIN_MAXLEN="2048"
export ALLOW_TRAIN_TRUNCATION="0"
export POST_TRAIN_EVAL="0"
export MATH_LEVELS="4,5"
export MAXNEW="4096"
export MAXLEN="1024"
export GRAD_CKPT="1"

echo "[DEEPMATH SFT] mode=$MODE arm=$ARM seed=$SEED rows=$N_TRAIN effective_batch=8"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_sft_math.py"
