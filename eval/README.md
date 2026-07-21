# DeepMath dev-gate eval —— 在 4090 上跑「官方 BI 尺」

DM-large 訓練完後,這包在**你自己的 4090**上跑官方 dev-gate:
base 與 DM-large 各評 500 題 internal-dev(vLLM **batch-invariant** 尺)→ `large − base ≥ 1pp` 才過。

> **為什麼 4090 能跑官方尺?** batch-invariance 的真實門檻是 **compute capability ≥ 8.0**(vLLM 對 SM80/Ampere 有專屬決定性分支),4090 是 CC 8.9 → 合格。原本 `vllm_gen_math.py`/`score_math_vllm.py` 裡寫成 `< 9` 是**離群 bug**(同專案其他腳本、連測試都用 `>=8.0`),已修。

## ⚠️ 這包和訓練包分開
- 訓練環境 = `~/dmtrain`(transformers 4.50)。
- **eval 環境 = `~/dmeval`**(vllm 0.24.0 / transformers 5.12.1 / torch 2.11.0+cu130)。**不要混裝。**
- **最可能要來回調的是 `env_setup_eval.sh`**:這個 cu130 stack 本為 Blackwell G4 調,在 Ada 4090 首裝可能要微調。**任一步報錯,整段貼回來。**

## 前置
```bash
export HF_TOKEN=<你的>
```
internal-dev 檔要在 `~/dm_prep/prep/math_l45_internal_dev.jsonl`(你跑訓練那包的 `setup_and_build.sh` 時已下載)。若不在,先補跑那步。

## 三步(在 eval/ 目錄)
```bash
bash env_setup_eval.sh     # 1. 建 ~/dmeval（驗版本 + CC>=8 + bf16）
bash smoke_eval.sh         # 2. 4 題 base-only smoke：證明 BI 尺能載入/生成/計分（不上傳）
bash run_dev_gate.sh       # 3. 全量：merge→base+large 各 500→score(上傳)→gate
```

## 判讀 / 回報
- **smoke** 印 `SMOKE_OK ✅` → 環境過,再跑全量。若掛在 gen 的 `cuda_runtime_library` 或版本檢查 → 回報我(多半是 cu130 lib 佈局要調)。
- **全量** 跑完看最後:`GATE ✅ 通過` 或 `❌ 未過`,並印 base/large fallback 與 delta。把 **`~/dmeval_run/dev_gate.json`** 裡的 `baseline.fallback_acc` / `large.fallback_acc` / `large.delta_vs_baseline` 回報我。
- 結果也會上傳到 HF:`Yuivdldk/mathvl-l45-dmdev-base-...` 與 `...-large-...`(private)。

## 注意
- **重跑全量前先 `rm -rf ~/dmeval_run ~/dmeval_merged_large`**(gen 會拒絕覆寫既有輸出)。
- 這是 dev gate(base vs large **兩臂都在同一台 4090**,自洽)。它不取代最終三 seed formal;但 `<1pp` 就代表不必再燒 formal。
- recipe/尺全部凍結,別改任何環境變數的乘積/版本(改了 records 會被 gate 拒)。
