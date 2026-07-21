# DeepMath dev-gate eval —— 在 4070/4090 上跑官方 BI 尺（可拆兩台）

DM-large 訓練完後,這包跑官方 dev-gate:base 與 DM-large 各評 500 題 internal-dev
(vLLM **batch-invariant** 尺)→ `large − base ≥ 1pp` 才過。

> **為什麼家用卡能跑官方尺?** batch-invariance 真實門檻是 **compute capability ≥ 8.0**
> (vLLM 對 SM80/Ampere 有專屬決定性分支);4070/4090 都是 CC 8.9 → 合格。原本
> `vllm_gen_math.py`/`score_math_vllm.py` 的 `< 9` 是離群 bug(其餘腳本＋測試都用 `>=8.0`),已修。

## 環境（每台都要,一次）
> eval 環境 `~/dmeval` 與訓練環境 `~/dmtrain` 分開,別混裝。
```bash
export HF_TOKEN=<你的>
cd eval
bash env_setup_eval.sh    # vllm0.24.0 / transformers5.12.1 / torch2.11.0+cu130；驗 CC>=8
bash smoke_eval.sh        # 4 題 base smoke，過了印 SMOKE_OK ✅ 再往下
```
internal-dev 需在 `~/dm_prep/prep/math_l45_internal_dev.jsonl`(訓練那包 setup_and_build 已下載)。

## 跑法 A：兩台 4070 各跑一臂（你要的）
```bash
# 4070 #1（base 臂）
ARM=base  bash run_arm.sh
# 4070 #2（large 臂，會先 merge DM-large）
ARM=large bash run_arm.sh
```
兩臂各自把結果上傳 HF(`Yuivdldk/mathvl-l45-dmdev-base-...` / `...-large-...`, private)。
兩臂都印 `ARM_DONE ✅` 後,**任一台**收尾(不需 GPU):
```bash
bash run_gate.sh          # 從 HF 抓兩臂 → 算 gate → 印通過/未過 + delta
```

## 跑法 B：一台跑到底（4090 建議）
```bash
bash run_dev_gate.sh      # merge→base+large 各500→score→gate，一次到底
```

## 4070 注意
- 12GB 較緊:`run_arm.sh` 預設 `GPU_UTIL=0.88`,若 gen 時 OOM → `GPU_UTIL=0.80 ARM=base bash run_arm.sh` 再試。
- 重跑某臂前先 `rm -rf ~/dmeval_base ~/dmeval_large ~/dmeval_merged_large`(gen 拒絕覆寫)。

## 回報我
- smoke 掛在 `cuda_runtime_library` 或版本 → 貼給我(多半 cu130 lib 佈局要調)。
- 收尾印 `GATE ✅/❌` + delta;把 `~/dmeval_gate/dev_gate.json` 的
  `baseline.fallback_acc` / `large.fallback_acc` / `large.delta_vs_baseline` 三個數回報。

## 給 AGY 的一句話
> 「clone 這個 repo,照 `eval/README.md`:先 `env_setup_eval.sh` 再 `smoke_eval.sh`(要看到 SMOKE_OK),
> 這台是 `ARM=<base 或 large> bash run_arm.sh`;跑完回報 `ARM_DONE` 與上傳的 repo 名。HF_TOKEN 已在環境。」
