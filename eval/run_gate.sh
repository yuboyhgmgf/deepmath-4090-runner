#!/usr/bin/env bash
# 收尾：兩臂都上傳 HF 後，任一台跑這支。
# 從 HF 下載 base/large 的 eval_records.json → 跑 gate_deepmath_dev（large-base>=1pp）。
# 不需 GPU。
set -euo pipefail
VENV="${VENV:-$HOME/dmeval}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/code" && pwd)"
DEV="${DEV:-$HOME/dm_prep/prep/math_l45_internal_dev.jsonl}"
BASE_REPO="${BASE_REPO:-mathvl-l45-dmdev-base-g4b-mn4096bi-n500}"
LARGE_REPO="${LARGE_REPO:-mathvl-l45-dmdev-large-g4b-mn4096bi-n500}"
WORK="${WORK:-$HOME/dmeval_gate}"
: "${HF_TOKEN:?export HF_TOKEN}"
[ -f "$DEV" ] || { echo "[FATAL] internal-dev 不在 $DEV"; exit 1; }
mkdir -p "$WORK"

echo "== 從 HF 下載兩臂 eval_records =="
BASE_REPO="$BASE_REPO" LARGE_REPO="$LARGE_REPO" WORK="$WORK" \
  "$VENV/bin/python" - <<'PYEOF'
import os
from huggingface_hub import hf_hub_download
tok = os.environ["HF_TOKEN"]; work = os.environ["WORK"]
api_user = None
try:
    from huggingface_hub import HfApi
    api_user = HfApi(token=tok).whoami()["name"]
except Exception:
    api_user = "Yuivdldk"
for repo_short, dst in ((os.environ["BASE_REPO"], "base"), (os.environ["LARGE_REPO"], "large")):
    repo = repo_short if "/" in repo_short else f"{api_user}/{repo_short}"
    p = hf_hub_download(repo, "eval_records.json", repo_type="dataset", token=tok,
                        local_dir=os.path.join(work, dst))
    print(f"downloaded {repo} -> {p}")
PYEOF

echo "== dev gate（large - base >= 1pp）=="
set +e
"$VENV/bin/python" gate_deepmath_dev.py \
  --internal-dev "$DEV" \
  --baseline "$WORK/base/eval_records.json" \
  --large-seed42 "$WORK/large/eval_records.json" \
  --out "$WORK/dev_gate.json"
GATE_RC=$?
set -e
echo ""
echo "==================== 結果 ===================="
echo "dev_gate.json：$WORK/dev_gate.json"
if [ "$GATE_RC" = "0" ]; then
  echo "GATE：✅ 通過（large - base >= 1pp）→ DeepMath 有搞頭，值得再燒三 seed formal。"
else
  echo "GATE：❌ 未過（large - base < 1pp，gate exit=$GATE_RC）→ 就此止損。"
fi
echo "把 dev_gate.json 的 baseline.fallback_acc / large.fallback_acc / large.delta_vs_baseline 回報我。"
