#!/usr/bin/env bash
# 4090 官方 BI eval 環境。⚠️ 與訓練環境「分開」——別裝進 ~/dmtrain。
# 釘死 vLLM ruler stack（gate 會逐項驗，錯一個版本 records 就不被接受）：
#   vllm 0.24.0 / transformers 5.12.1 / torch 2.11.0+cu130 / math-verify 0.9.0
# ⚠️ 這一步是整包最可能要來回調的地方（cu130 stack 原本為 Blackwell G4 調，
#   在 Ada 4090 上首次安裝可能要微調）。任一步報錯，整段貼回來我修。
set -euo pipefail
VENV="${VENV:-$HOME/dmeval}"
echo "== eval venv: $VENV（與 ~/dmtrain 分開）=="
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

echo "== vLLM 0.24.0（先裝，會拉一版 torch）=="
"$VENV/bin/pip" install -q "vllm==0.24.0"

echo "== 覆蓋成釘死的 cu130 torch =="
"$VENV/bin/pip" install -q torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130

echo "== 釘死 transformers 5.12.1 + merge/scorer 需要的 =="
"$VENV/bin/pip" install -q "transformers==5.12.1" "math-verify[antlr4_13_2]==0.9.0" \
  "peft==0.19.1" datasets sentencepiece huggingface_hub

echo "== 驗證版本 + CC>=8（BI 真正門檻）+ bf16 =="
"$VENV/bin/python" - <<'PYEOF'
import vllm, transformers, torch
got = dict(vllm=vllm.__version__, transformers=transformers.__version__,
           torch=str(torch.__version__), torch_cuda=str(torch.version.cuda))
print("versions:", got)
want = {"vllm": "0.24.0", "transformers": "5.12.1", "torch": "2.11.0+cu130", "torch_cuda": "13.0"}
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
assert not bad, f"[FATAL] ruler stack 版本不符（gate 會拒 records）: {bad}"
assert torch.cuda.is_available(), "[FATAL] no CUDA GPU"
cc = torch.cuda.get_device_capability(0); name = torch.cuda.get_device_name(0)
print("GPU:", name, "cc", cc, "VRAM(GB)", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
assert cc[0] >= 8, f"[FATAL] batch-invariant 需要 CC>=8.0，got sm_{cc[0]}{cc[1]}（{name}）"
assert torch.cuda.is_bf16_supported(), "[FATAL] no bf16"
print("versions + CC>=8 + bf16 all OK")
PYEOF
echo "ENV_EVAL_READY -> $VENV/bin/python"
