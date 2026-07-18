# DeepMath 4090/4070 runner — DM-large 訓練

在**本機 24GB 4090 / 12GB 4070**上訓練 DeepMath 的 **DM-large**（50K rows, 4B LoRA, bf16）。
目的分兩段：先跑 **1K smoke** 量「50K 要多久 + 會不會 OOM」，過了再跑 **development gate** 的全量訓練。

> ⚠️ **這台只做訓練 + merge。** 正式 fallback eval 用 vLLM BI 尺、需 sm≥9.0（Colab G4），**4070/4090(sm_89) 跑不了 BI**，那步在別處做。訓練好的 adapter 會上 HF，由 G4 端下載評測。

---

## 硬事實
- 釘死訓練 runtime（`run_sft_math.py` fail-closed 檢查）：`transformers 4.50.0 / peft 0.19.1 / accelerate 1.13.0 / huggingface_hub 0.36.0`。
- recipe（凍結，勿動）：Gemma-3-4B-it `@093f9f38…`、LoRA r16/α32/dropout0.05/all-linear、lr 1e-4、cosine、epoch 1、**有效 batch 8**、TRAIN_MAXLEN 2048、seed 42。
- schedule：不放進 git（453MB）。用 `setup_and_build.sh` 從 HF prepared corpus **重建**（deterministic，驗 `config_hash=21ea707…`）。

## 0. 前置
```bash
export HF_TOKEN=<Yuivdldk 的 token>      # env-only，勿寫進檔案/命令歷史/repo
```

## 1. 建環境（一次）
```bash
bash env_setup.sh          # 出 ~/dmtrain venv；會驗版本+bf16，T4 會被擋
```

## 2. 下載 prepared + 重建 schedule（一次，~110MB，不需 GPU）
```bash
bash setup_and_build.sh    # 出 ~/dm_build/dm_large_train.jsonl 等；驗 config_hash 吻合
```

## 3. 1K smoke —— 量工時 + OOM（**先跑這個**）
```bash
cd code
MODE=smoke ARM=large SEED=42 \
  TRAIN_DATA=$HOME/dm_build/dm_large_train.jsonl \
  PYTHON_BIN=$HOME/dmtrain/bin/python \
  TRAIN_BS=1 GRAD_ACCUM=8 \
  bash run_deepmath_sft.sh
```
- smoke = 只訓前 **1000 rows**、不上傳。看它印出的 **train runtime** 與 **peak GB**。
- **工時換算**：`50K 全量 ≈ (runtime/1000) × 50000`。例：1000 rows 用 600s → 50K ≈ 30000s ≈ 8.3h。
- **OOM 判讀**：peak GB < 你的 VRAM = 過；若中途 OOM，代表這台這個 batch 塞不下。
- **batch 設定**：
  - **12GB 4070**：先用 `TRAIN_BS=1 GRAD_ACCUM=8`（最省）。仍爆就這台跑不了 4B，改 24GB 卡。
  - **24GB 4090**：可用預設 `TRAIN_BS=2 GRAD_ACCUM=4`（把上面兩個環境變數拿掉即可），較快。
  - 有效 batch 恆 8（BS×ACCUM=8），**只准改拆法、不准改乘積**（launcher 會擋）。

## 4. DM-large development 全量訓練
smoke 過了、工時可接受，再跑：
```bash
cd code
MODE=development ARM=large SEED=42 \
  TRAIN_DATA=$HOME/dm_build/dm_large_train.jsonl \
  PYTHON_BIN=$HOME/dmtrain/bin/python \
  TRAIN_BS=1 GRAD_ACCUM=8 \
  bash run_deepmath_sft.sh
```
- 全量 50K、**會上傳 adapter 到 HF**（`UPLOAD_MODEL=1`）、**每 500 步 checkpoint 可續跑**（`CKPT_RESUME=1`）。
- **每日時數不夠(如 4090 6-8h/天)沒關係**：跑到時間到就停,同指令重跑會從 HF/本地 checkpoint 續,不用重頭。
- 跑完把 HF 上的 adapter repo 名稱回報,G4 端下載 → `merge_lora.py` → vLLM BI eval 500 internal-dev → dev gate。

## 5. 交回 eval（不在這台）
adapter 上 HF 後：G4/box 端 `merge_lora.py`（bf16）→ `vllm_gen_math.py`(BI, MAXNEW4096) → `score_math_vllm.py` → `gate_deepmath_dev.py`。`large−base≥1pp` 才進 formal。

---
**這台的產物只有 adapter，不是研究成果數字。** strict/fallback/成敗一律由 G4 BI 尺 + 三 seed formal 決定。
