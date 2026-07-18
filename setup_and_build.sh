#!/usr/bin/env bash
# 下載 HF 上已完成的 prepared corpus（~110MB，不抓 5GB source_rows），
# 在本機「重建」出 registered schedule，並驗 config_hash 完全吻合。
# 這一步 deterministic、不需 GPU；比傳 453MB schedule 輕。
set -euo pipefail
VENV="${VENV:-$HOME/dmtrain}"
DL="${DL:-$HOME/dm_prep}"
BUILD="${BUILD:-$HOME/dm_build}"
: "${HF_TOKEN:?export HF_TOKEN with the Yuivdldk token (env only, never hardcode)}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/code" && pwd)"
mkdir -p "$DL"

echo "== 1. 下載 prepared 輸入（rev f0bcf84…）=="
DL="$DL" HF_TOKEN="$HF_TOKEN" "$VENV/bin/python" - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
rev = "f0bcf84cb4d0b84f21d8a8dad08b38a68955fbb5"
files = [
    "deterministic_preparation/deepmath_prepared.jsonl",
    "deterministic_preparation/preparation_manifest.json",
    "prep/preparation_manifest.json",
    "prep/math_l45_replay_pool.jsonl",
    "prep/math_l45_internal_dev.jsonl",
]
for f in files:
    hf_hub_download("Yuivdldk/deepmath-prep-v1", f, repo_type="dataset",
                    revision=rev, local_dir=os.environ["DL"], token=os.environ["HF_TOKEN"])
    print("  downloaded", f)
PYEOF

echo "== 2. 重建 registered schedule（deterministic）=="
"$VENV/bin/python" "$CODE/build_deepmath_metadata_schedule.py" \
  --prepared "$DL/deterministic_preparation/deepmath_prepared.jsonl" \
  --preparation-manifest "$DL/deterministic_preparation/preparation_manifest.json" \
  --preparation-parent-manifest "$DL/prep/preparation_manifest.json" \
  --replay-pool "$DL/prep/math_l45_replay_pool.jsonl" \
  --internal-dev "$DL/prep/math_l45_internal_dev.jsonl" \
  --outdir "$BUILD" | tail -3

echo "== 3. 驗 config_hash 吻合 registered =="
EXPECT="21ea707b4d32660b3a28abee7e8a5e2979759965f1f722a88645e5d9d2f4814f"
GOT="$("$VENV/bin/python" -c "import json;print(json.load(open('$BUILD/deepmath_schedule_summary.json'))['config_hash'])")"
if [ "$GOT" = "$EXPECT" ]; then
  echo "[OK] schedule config_hash 吻合: $GOT"
else
  echo "[FATAL] config_hash 不吻合: got=$GOT expected=$EXPECT（重建與 registered 不一致，停）"; exit 1
fi
echo "BUILD_READY -> $BUILD/dm_large_train.jsonl  ($(wc -l < "$BUILD/dm_large_train.jsonl") rows)"
