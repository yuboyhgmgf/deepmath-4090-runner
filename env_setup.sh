#!/usr/bin/env bash
# 建 DeepMath 訓練 venv（4090/4070，Ada bf16，訓練不需 vLLM）。
# ⚠️ run_sft_math.py 會 fail-closed 檢查這幾個「精確」版本，別亂升：
#     transformers 4.50.0 / peft 0.19.1 / accelerate 1.13.0 / huggingface_hub 0.36.0
set -euo pipefail
VENV="${VENV:-$HOME/dmtrain}"
echo "== venv: $VENV =="
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip

echo "== torch (Ada sm_89 bf16；若你的驅動 CUDA 不同就換 cu 版本) =="
"$VENV/bin/pip" install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "== 釘死的訓練 runtime（精確版本，fail-closed）=="
"$VENV/bin/pip" install -q \
  "transformers==4.50.0" "peft==0.19.1" "accelerate==1.13.0" "huggingface_hub==0.36.0"

echo "== math_common / builder 需要的 =="
"$VENV/bin/pip" install -q "math-verify[antlr4_13_2]==0.9.0" "datasets==3.5.0" sentencepiece

echo "== 驗證版本 + bf16 =="
"$VENV/bin/python" - <<'PYEOF'
import transformers, peft, accelerate, huggingface_hub as hub, torch
want = {"transformers":"4.50.0","peft":"0.19.1","accelerate":"1.13.0","huggingface_hub":"0.36.0"}
got  = {"transformers":transformers.__version__,"peft":peft.__version__,
        "accelerate":accelerate.__version__,"huggingface_hub":hub.__version__}
print("versions:", got)
bad = {k:(got[k],v) for k,v in want.items() if got[k]!=v}
assert not bad, f"[FATAL] runtime version mismatch (run_sft_math will reject): {bad}"
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "no-cuda")
assert torch.cuda.is_available(), "[FATAL] no CUDA GPU visible"
assert torch.cuda.is_bf16_supported(), "[FATAL] GPU has no bf16 (need Ada/Ampere+; T4 not allowed)"
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print("GPU:", name, "cc", cc, "VRAM(GB)", round(torch.cuda.get_device_properties(0).total_memory/1e9,1))
PYEOF
echo "ENV_READY  ($VENV/bin/python)"
