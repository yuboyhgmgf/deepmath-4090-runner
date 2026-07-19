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

echo "== 4. 資料指紋 sha256 交叉檢查（advisory）=="
# config_hash 只證明「凍結常數」吻合，未證明重建語料與 registered 逐位元一致。這裡再比對
# 真正的資料指紋。設為 advisory（不 exit）：尚未在本機實測重建是否 byte-deterministic——
# 首建若全 [OK]，即可把此段升成硬閘；若 [WARN]，代表重建非決定性，先別拿來訓練並回報。
"$VENV/bin/python" - "$BUILD" <<'PYEOF'
import hashlib, os, sys
build = sys.argv[1]
expect = {
    "dm_large_train.jsonl": "d45cf97b9079d32a21b29d0751e9e6fbc0047a1ed9e58adc66058af4cb82c0ef",
    "dm_small_train.jsonl": "fdaa0bf60857707ec0b745c459c3600e48a6ec6f7edc3543d93f5a5c978472a8",
}
ok = True
for name, want in expect.items():
    p = os.path.join(build, name)
    if not os.path.isfile(p):
        print(f"[WARN] 缺 {name}，跳過指紋比對"); ok = False; continue
    hsh = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hsh.update(chunk)
    got = hsh.hexdigest()
    print(f"[OK] {name} sha256 吻合 registered" if got == want
          else f"[WARN] {name} sha256 不吻合: got={got} expected={want}")
    ok = ok and got == want
print("[OK] 資料指紋全部吻合（重建逐位元一致，可考慮把本段升成硬閘 exit 1）" if ok
      else "[WARN] 資料指紋未全吻合——重建可能非決定性，請先別用來訓練並回報")
PYEOF
echo "BUILD_READY -> $BUILD/dm_large_train.jsonl  ($(wc -l < "$BUILD/dm_large_train.jsonl") rows)"
