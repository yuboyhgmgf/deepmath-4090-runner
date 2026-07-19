#!/usr/bin/env python3
"""
MATH SFT/LoRA runner for Gemma-3（reasoner 蒸餾 A/B/control 臂）。
= run_sft_grid.py 的 MATH 版：把「尺」從 kd_sft_common(GSM8K,####) 換成 math_common(\boxed{},Math-Verify)。

與 GSM8K runner 的關鍵差異：
  * ruler = math_common（INSTR 要 \boxed{}、評分用 Math-Verify 符號等價、載 EleutherAI/hendrycks_math）。
  * eval 前跑 selftest 驗 Math-Verify（壞掉會顯示為假 ~0%，先擋）。
  * 長序列 SDPA 修正：Gemma-3 4B/12B 不把 attn_implementation="eager" 傳進巢狀 text model → 用 SDPA，
    長 MATH 序列會踩 "p.attn_bias_ptr is not correctly aligned"；強制 math SDES kernel（數值精確，只較慢）。
  * MAXNEW 預設 2048（MATH 生成長）；訓練 MAXLEN 另開 TRAIN_MAXLEN（B 教材=reasoning+content 很長）。
  * repo 命名 math-{level_tag}-results/model-...；test 用 L1-5 全量（MATH_LEVELS=1,2,3,4,5）。

臂靠 TRAIN_DATA + DATA_TAG 指定（build_math_train.py 產出的 data_rmathA/B/C.jsonl；tag=rmathA/rmathB/rmathC）。

用法（box）：
  MATH_LEVELS=1,2,3,4,5 MODEL=google/gemma-3-4b-it METHODS=lora LRS_LORA=1e-4 EPOCHS=2 \
  TRAIN_DATA=~/data_rmathB.jsonl DATA_TAG=rmathB TRAIN_MAXLEN=2048 MAXNEW=2048 EVAL_BS=24 \
  python run_sft_math.py
  DeepMath 訓練（正式尺另走 vLLM）: POST_TRAIN_EVAL=0 EPOCHS=1 ... python run_sft_math.py
  smoke: UPLOAD_RESULTS=0 N_TRAIN=32 N_TEST=40 ... （本地跑、不碰 HF）
"""
import os, sys, subprocess, gc, time, random, traceback, shutil, hashlib, json
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("MATH_LEVELS", "1,2,3,4,5")   # math_common 在 import 時讀
from datetime import datetime, timezone


class _GuardSkip(Exception):
    pass


DEEPMATH_CHECKPOINT_PREFIX = "last-checkpoint/"
DEEPMATH_CHECKPOINT_REQUIRED = {
    "adapter_config.json",
    "trainer_state.json",
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
}
DEEPMATH_CHECKPOINT_WEIGHT_FILES = {
    "adapter_model.safetensors",
    "adapter_model.bin",
}
DEEPMATH_LORA_R = 16
DEEPMATH_LORA_ALPHA = 32
DEEPMATH_LORA_DROPOUT = 0.05
DEEPMATH_LORA_TARGET_MODULES = "all-linear"
DEEPMATH_TECHNICAL_PREFLIGHT_MAX_ROWS = 64
DEEPMATH_EXPECTED_MODEL = "google/gemma-3-4b-it"
DEEPMATH_EXPECTED_MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
DEEPMATH_TRAINING_RUNTIME = {
    "transformers": "4.50.0",
    "peft": "0.19.1",
    "accelerate": "1.13.0",
    "huggingface_hub": "0.36.0",
}


def _validate_deepmath_technical_preflight(
    *,
    enabled,
    post_train_eval,
    model,
    model_revision,
    methods,
    epochs,
    n_train,
    train_bs,
    grad_accum,
    ckpt_resume,
    data_tag,
    public,
    upload_model,
    upload_results,
):
    """Keep the schedule-free T4 check small and impossible to cite as formal."""
    if not enabled:
        return
    if post_train_eval:
        raise RuntimeError("technical preflight must set POST_TRAIN_EVAL=0")
    if model != DEEPMATH_EXPECTED_MODEL or model_revision != DEEPMATH_EXPECTED_MODEL_REVISION:
        raise RuntimeError(
            "technical preflight requires the pinned Gemma-3-4B revision; "
            f"got model={model} revision={model_revision}"
        )
    if methods != ["lora"] or epochs != [1]:
        raise RuntimeError(
            f"technical preflight is one-epoch LoRA only; methods={methods} epochs={epochs}"
        )
    if not 1 <= n_train <= DEEPMATH_TECHNICAL_PREFLIGHT_MAX_ROWS:
        raise RuntimeError(
            "technical preflight row count must be between 1 and "
            f"{DEEPMATH_TECHNICAL_PREFLIGHT_MAX_ROWS}; got {n_train}"
        )
    if train_bs * grad_accum != 8:
        raise RuntimeError(
            "technical preflight must preserve effective batch 8; "
            f"got train_bs={train_bs} grad_accum={grad_accum}"
        )
    if ckpt_resume:
        raise RuntimeError("technical preflight may not use registered checkpoint resume")
    if not data_tag.startswith("deepmath-t4-preflight"):
        raise RuntimeError(
            "technical preflight DATA_TAG must start with deepmath-t4-preflight"
        )
    if public:
        raise RuntimeError("technical preflight must remain private")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_obj(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compact_deepmath_data_tag(data_tag, technical_preflight=False):
    if technical_preflight:
        return "t4pf"
    tag = data_tag.removeprefix("deepmath-") or "run"
    tag = "".join(char if (char.isalnum() or char in "._-") else "-" for char in tag)
    tag = tag.strip("-.") or "run"
    if len(tag) > 20:
        tag = f"{tag[:11]}-{hashlib.sha256(tag.encode()).hexdigest()[:8]}"
    return tag


def _build_run_slug(
    *, level_tag, short, method, lr, epochs, seed, n_train, maxnew, n_test,
    post_train_eval, deepmath_run, technical_preflight, train_data_sha256,
    data_tag,
):
    if deepmath_run or technical_preflight:
        tag = _compact_deepmath_data_tag(data_tag, technical_preflight)
        slug = (
            f"dm-{tag}-{level_tag}-{short}-{method}-lr{lr:g}-ep{epochs:g}"
            f"-s{seed}-n{n_train}-d{train_data_sha256[:10]}"
        )
        if len(slug) > 64:
            raise RuntimeError(f"DeepMath run slug unexpectedly exceeds 64 chars: {slug}")
        return slug
    return (
        f"{level_tag}-{short}-{method}-lr{lr:g}-ep{epochs:g}-s{seed}"
        f"-ntr{n_train}-mn{maxnew}-n{n_test}"
        + ("" if post_train_eval else "-pte0")
        + (("-" + data_tag) if data_tag else "")
    )


def _deepmath_final_model_ready(repo_files, method):
    """An intermediate Trainer push is not a completed DeepMath adapter."""
    files = set(repo_files)
    if "training_provenance.json" not in files:
        return False
    if method == "lora":
        return "adapter_config.json" in files and bool(
            DEEPMATH_CHECKPOINT_WEIGHT_FILES & files
        )
    return "config.json" in files


def _deepmath_remote_checkpoint_ready(repo_files):
    """Fail closed on partial `last-checkpoint/` uploads; return False if absent."""
    relative = {
        path[len(DEEPMATH_CHECKPOINT_PREFIX):]
        for path in repo_files
        if path.startswith(DEEPMATH_CHECKPOINT_PREFIX)
    }
    if not relative:
        return False
    missing = sorted(DEEPMATH_CHECKPOINT_REQUIRED - relative)
    if missing or not (DEEPMATH_CHECKPOINT_WEIGHT_FILES & relative):
        raise RuntimeError(
            "incomplete DeepMath Hub last-checkpoint: "
            f"missing={missing} weights={sorted(DEEPMATH_CHECKPOINT_WEIGHT_FILES & relative)}"
        )
    return True


def _deepmath_local_checkpoint_ready(path):
    if not path or not os.path.isdir(path):
        return False
    files = {
        name for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
    }
    missing = sorted(DEEPMATH_CHECKPOINT_REQUIRED - files)
    if missing or not (DEEPMATH_CHECKPOINT_WEIGHT_FILES & files):
        raise RuntimeError(
            f"incomplete local DeepMath checkpoint {path}: "
            f"missing={missing} weights={sorted(DEEPMATH_CHECKPOINT_WEIGHT_FILES & files)}"
        )
    return True


def _deepmath_local_checkpoint_safe(path):
    """Non-raising variant of _deepmath_local_checkpoint_ready: True iff the dir
    has every required file, else False. Used to *discard* a half-written
    checkpoint (power-loss / kill during a save) instead of hard-failing resume."""
    try:
        return _deepmath_local_checkpoint_ready(path)
    except RuntimeError:
        return False


def _deepmath_checkpoint_step(path):
    state_path = os.path.join(path, "trainer_state.json")
    try:
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid DeepMath checkpoint state: {state_path}") from exc
    step = state.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise RuntimeError(f"invalid DeepMath checkpoint global_step: {step!r}")
    return step


def _validate_deepmath_training_recipe(methods, observed_runtime):
    if methods != ["lora"]:
        raise RuntimeError(f"registered DeepMath training is LoRA-only; got methods={methods}")
    if observed_runtime != DEEPMATH_TRAINING_RUNTIME:
        raise RuntimeError(
            "DeepMath training runtime drift: "
            f"observed={observed_runtime} expected={DEEPMATH_TRAINING_RUNTIME}"
        )


def _select_deepmath_checkpoint(paths):
    candidates = [path for path in paths if path]
    return max(candidates, key=_deepmath_checkpoint_step) if candidates else None


def _pip():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "transformers==4.50.0", "peft==0.19.1", "accelerate==1.13.0",
         "datasets==4.0.0", "safetensors", "huggingface_hub==0.36.0",
         "math-verify[antlr4_13_2]==0.9.0"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)


def _shutdown(reason, success):
    flag = "AUTO_SHUTDOWN_ON_SUCCESS" if success else "AUTO_SHUTDOWN_ON_ERROR"
    if os.environ.get(flag, "0").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[GUARD] {reason}; shutdown in 20s", flush=True)
        time.sleep(20)
        subprocess.Popen(["sudo", "shutdown", "-h", "now"])


def main():
    _pip()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import torch
    # 長 MATH 序列的 SDPA alignment 修正（同 eval_baseline_math）：關 flash/mem-efficient、留 math kernel（精確）。
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    import accelerate
    import huggingface_hub
    import peft
    import transformers
    from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                              Trainer, DataCollatorForSeq2Seq)
    from transformers.trainer_utils import get_last_checkpoint
    from huggingface_hub import HfApi, login, create_repo, hf_hub_download, snapshot_download
    from peft import LoraConfig, get_peft_model, PeftModel
    import math_common as C

    token = C.read_token()
    if not token:
        print("[FATAL] no HF token", flush=True)
        _shutdown("no token", False)
        raise SystemExit(2)
    login(token)
    api = HfApi(token=token)
    user = api.whoami()["name"]

    # ---- Math-Verify selftest：壞掉的 scorer 會顯示為假 ~0%，先擋 ----
    if not C.selftest():
        print("[FATAL] Math-Verify selftest FAILED — 修好 math-verify 安裝再訓練/評測。", flush=True)
        _shutdown("selftest failed", False)
        raise SystemExit(3)

    MODEL = os.environ.get("MODEL", "google/gemma-3-4b-it")
    MODEL_REVISION = os.environ.get("MODEL_REVISION", "").strip() or None
    short = C.model_short(MODEL)
    SEED = int(os.environ.get("SEED", "42"))
    N_TRAIN = int(os.environ.get("N_TRAIN", "100000"))   # 蒸餾集全量（過濾後 <7500），大值=不截斷
    N_TEST = int(os.environ.get("N_TEST", "0"))          # 0=全量 L1-5 test（~5000）
    TRAIN_MAXLEN = int(os.environ.get("TRAIN_MAXLEN", "2048"))  # B 教材長，訓練截斷要大
    MAXLEN = int(os.environ.get("MAXLEN", "1024"))       # eval prompt 截斷
    MAXNEW = int(os.environ.get("MAXNEW", "2048"))       # MATH 生成長
    PUBLIC = os.environ.get("PUBLIC", "1").strip().lower() not in {"0", "false", "no"}
    UPLOAD_MODEL = os.environ.get("UPLOAD_MODEL", "1").strip().lower() not in {"0", "false", "no"}
    UPLOAD_RESULTS = os.environ.get("UPLOAD_RESULTS", "1").strip().lower() not in {"0", "false", "no"}
    ALLOW_TRAIN_TRUNCATION = os.environ.get("ALLOW_TRAIN_TRUNCATION", "0").strip().lower() in {"1", "true", "yes"}
    # Backward compatible: historical runs still do their transformers diagnostic eval.
    # DeepMath uses POST_TRAIN_EVAL=0 so training never touches formal test and final numbers
    # come only from the pinned vLLM 4096/1024/5136 ruler after bf16 merge.
    POST_TRAIN_EVAL = os.environ.get("POST_TRAIN_EVAL", "1").strip().lower() not in {"0", "false", "no"}
    TECHNICAL_PREFLIGHT = os.environ.get(
        "DEEPMATH_TECHNICAL_PREFLIGHT", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    TRAIN_ONLY = not POST_TRAIN_EVAL
    DEEPMATH_RUN = TRAIN_ONLY and not TECHNICAL_PREFLIGHT
    if DEEPMATH_RUN or TECHNICAL_PREFLIGHT:
        if MODEL != DEEPMATH_EXPECTED_MODEL or MODEL_REVISION != DEEPMATH_EXPECTED_MODEL_REVISION:
            raise SystemExit(
                "DeepMath training requires the pinned Gemma-3-4B model revision "
                f"{DEEPMATH_EXPECTED_MODEL_REVISION}; got model={MODEL} revision={MODEL_REVISION}"
            )
    GRAD_CKPT = os.environ.get("GRAD_CKPT", "1").strip().lower() not in {"0", "false", "no"}
    CKPT_RESUME = os.environ.get("CKPT_RESUME", "0").strip().lower() in {"1", "true", "yes", "on"}
    default_save_steps = "500" if DEEPMATH_RUN else "200"
    try:
        TRAIN_SAVE_STEPS = int(os.environ.get("TRAIN_SAVE_STEPS", default_save_steps))
    except ValueError as exc:
        raise SystemExit("TRAIN_SAVE_STEPS must be an integer") from exc
    if CKPT_RESUME and TRAIN_SAVE_STEPS < 1:
        raise SystemExit("TRAIN_SAVE_STEPS must be positive when checkpoint resume is enabled")
    DATA_TAG = os.environ.get("DATA_TAG", "").strip()

    methods = [m.strip() for m in os.environ.get("METHODS", "lora").split(",") if m.strip()]
    lrs_lora = [float(x) for x in os.environ.get("LRS_LORA", "1e-4").split(",")]
    lrs_full = [float(x) for x in os.environ.get("LRS_FULL", "5e-6,1e-5").split(",")]
    epochs = [int(x) for x in os.environ.get("EPOCHS", "2").split(",")]
    if short == "g12b":
        methods = [m for m in methods if m == "lora"] or ["lora"]
    if DEEPMATH_RUN or TECHNICAL_PREFLIGHT:
        observed_training_runtime = {
            "transformers": str(transformers.__version__),
            "peft": str(peft.__version__),
            "accelerate": str(accelerate.__version__),
            "huggingface_hub": str(huggingface_hub.__version__),
        }
        _validate_deepmath_training_recipe(methods, observed_training_runtime)

    def bs_ga():  # effective batch 8；B 長序列可用 TRAIN_BS=2 GRAD_ACCUM=4 覆蓋
        return int(os.environ.get("TRAIN_BS", "8")), int(os.environ.get("GRAD_ACCUM", "1"))

    preflight_bs, preflight_ga = bs_ga()
    _validate_deepmath_technical_preflight(
        enabled=TECHNICAL_PREFLIGHT,
        post_train_eval=POST_TRAIN_EVAL,
        model=MODEL,
        model_revision=MODEL_REVISION,
        methods=methods,
        epochs=epochs,
        n_train=N_TRAIN,
        train_bs=preflight_bs,
        grad_accum=preflight_ga,
        ckpt_resume=CKPT_RESUME,
        data_tag=DATA_TAG,
        public=PUBLIC,
        upload_model=UPLOAD_MODEL,
        upload_results=UPLOAD_RESULTS,
    )

    EVAL_BS = int(os.environ.get("EVAL_BS", "0")) or {"g1b": 48, "g4b": 24, "g12b": 8}.get(short, 8)

    torch.backends.cuda.matmul.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[GPU] {gpu_name} {gpu_gb:.1f}GiB | MODEL={MODEL} short={short} seed={SEED} "
          f"levels={C.MATH_LEVEL_LABEL} tag={DATA_TAG or '(none)'}", flush=True)

    # ---- data ----
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot is None or eot == tok.unk_token_id:
        eot = tok.eos_token_id

    def encode_parts(ex):
        p = tok.apply_chat_template(
            [{"role": "user", "content": C.INSTR.format(q=ex["question"])}],
            add_generation_prompt=True, tokenize=True,
        )
        if isinstance(p, dict):
            p = p["input_ids"]
        if p and isinstance(p[0], list):
            p = p[0]
        c = tok(ex["answer"].strip(), add_special_tokens=False)["input_ids"] + [eot]
        return p, c

    def build_from_parts(p, c):
        ids = (p + c)[:TRAIN_MAXLEN]
        labels = ([-100] * len(p) + c)[:TRAIN_MAXLEN]
        return {"input_ids": ids, "labels": labels}

    TRAIN_DATA = os.environ.get("TRAIN_DATA", "").strip()
    if not TRAIN_DATA:
        print("[FATAL] MATH runner 需要 TRAIN_DATA（reasoner 蒸餾 jsonl）", flush=True)
        _shutdown("no train data", False)
        raise SystemExit(2)
    import json as _json
    with open(TRAIN_DATA, encoding="utf-8") as handle:
        train_raw = [_json.loads(line) for line in handle if line.strip()][:N_TRAIN]
    train_data_sha256 = _sha256_file(TRAIN_DATA)
    schedule_validation = None
    schedule_validation_sha256 = None
    if DEEPMATH_RUN:
        validation_path = os.environ.get("SCHEDULE_VALIDATION", "").strip()
        if not validation_path or not os.path.isfile(validation_path):
            raise SystemExit("DeepMath training requires SCHEDULE_VALIDATION from the fail-closed launcher")
        with open(validation_path, encoding="utf-8") as handle:
            schedule_validation = _json.load(handle)
        schedule_validation_sha256 = _sha256_file(validation_path)
        if (schedule_validation.get("stage"), schedule_validation.get("kind")) != (
            "complete", "deepmath_schedule_validation"
        ):
            raise SystemExit("invalid DeepMath schedule validation artifact")
        if schedule_validation.get("train_data_sha256") != train_data_sha256:
            raise SystemExit("TRAIN_DATA hash differs from the validated DeepMath schedule")
    train_questions = [x["question"] for x in train_raw]
    print(
        f"[DATA] TRAIN_DATA={TRAIN_DATA} rows={len(train_raw)} sha256={train_data_sha256}",
        flush=True,
    )

    # Exact audit: use the actual chat-template prompt plus completion tokens that
    # will be passed to Trainer. The old answer_len+40 proxy undercounted long prompts.
    encoded = [encode_parts(x) for x in train_raw]
    trunc_rows = [i for i, (p, c) in enumerate(encoded) if len(p) + len(c) > TRAIN_MAXLEN]
    ntrunc = len(trunc_rows)
    boxed_tail_lost = 0
    for i in trunc_rows:
        p, c = encoded[i]
        keep = max(0, TRAIN_MAXLEN - len(p))
        retained = tok.decode(c[:keep], skip_special_tokens=True)
        if C.last_boxed_only_string(train_raw[i]["answer"]) is not None and C.last_boxed_only_string(retained) is None:
            boxed_tail_lost += 1
    train_data = [build_from_parts(p, c) for p, c in encoded]
    if ntrunc:
        print(f"[FATAL] exact truncation audit: {ntrunc}/{len(train_raw)} completions exceed "
              f"TRAIN_MAXLEN={TRAIN_MAXLEN}; boxed tail lost={boxed_tail_lost}", flush=True)
        print(f"[FATAL] first affected row indices: {trunc_rows[:20]}", flush=True)
        if not ALLOW_TRAIN_TRUNCATION:
            raise SystemExit("Increase TRAIN_MAXLEN; formal runs may not truncate target completions")
        print("[WARN] ALLOW_TRAIN_TRUNCATION=1: exploratory contaminated run; summary will record it", flush=True)

    test = None
    if POST_TRAIN_EVAL:
        test = C.load_math("test")
        if N_TEST and N_TEST < len(test):
            test = test.select(range(N_TEST))
        # contamination：train 的 question vs test 的 problem
        overlap = len(set(train_questions) & set(test["problem"]))
        assert overlap == 0, f"train/test contamination! overlap={overlap}"
    else:
        print("[DATA] POST_TRAIN_EVAL=0: skip loading formal MATH test; use merged-model vLLM eval", flush=True)
    coll = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)
    test_count = len(test) if test is not None else 0
    print(f"[DATA] train={len(train_data)} test={test_count} (levels {C.MATH_LEVEL_TAG})", flush=True)

    def setseed(s):
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

    cells = [(mtd, lr, ep) for ep in epochs for mtd in methods
             for lr in (lrs_lora if mtd == "lora" else lrs_full)]
    print(f"[GRID] {len(cells)} cells: {cells}", flush=True)

    ran = skipped = failed = deferred = 0
    failures = []
    for (method, lr, ep) in cells:
        run_slug = _build_run_slug(
            level_tag=C.MATH_LEVEL_TAG,
            short=short,
            method=method,
            lr=lr,
            epochs=ep,
            seed=SEED,
            n_train=len(train_data),
            maxnew=MAXNEW,
            n_test=test_count,
            post_train_eval=POST_TRAIN_EVAL,
            deepmath_run=DEEPMATH_RUN,
            technical_preflight=TECHNICAL_PREFLIGHT,
            train_data_sha256=train_data_sha256,
            data_tag=DATA_TAG,
        )
        if TECHNICAL_PREFLIGHT:
            results_repo = f"{user}/t4-res-{train_data_sha256[:10]}"
            model_repo = f"{user}/t4-mdl-{train_data_sha256[:10]}"
        else:
            results_repo = f"{user}/math-results-{run_slug}"
            model_repo = f"{user}/math-model-{run_slug}"
        for repo_id in (results_repo, model_repo):
            if len(repo_id) > 96:
                raise RuntimeError(f"Hugging Face repo id exceeds 96 characters: {repo_id}")
        done = C.cell_done(results_repo, token) if UPLOAD_RESULTS else False
        if done and DEEPMATH_RUN:
            try:
                prior_summary_path = hf_hub_download(
                    repo_id=results_repo, filename="summary.json", repo_type="dataset", token=token
                )
                with open(prior_summary_path, encoding="utf-8") as handle:
                    prior_summary = _json.load(handle)
                done = (
                    prior_summary.get("train_data_sha256") == train_data_sha256
                    and prior_summary.get("model_revision") == MODEL_REVISION
                    and prior_summary.get("schedule_validation_sha256") == schedule_validation_sha256
                )
            except Exception:
                done = False
        if UPLOAD_RESULTS and done:
            print(f"[SKIP done] {results_repo}", flush=True)
            skipped += 1
            continue
        try:
            print(f"\n===== CELL {method} lr={lr:g} ep={ep} -> {results_repo} =====", flush=True)
            if UPLOAD_RESULTS:
                create_repo(results_repo, repo_type="dataset", private=not PUBLIC, exist_ok=True, token=token)

            base = "/content" if os.path.isdir("/content") else "/tmp"
            out_dir = f"{base}/{run_slug}"
            res_dir = os.path.join(out_dir, "results")
            mdl_dir = os.path.join(out_dir, "model")
            trn_dir = os.path.join(out_dir, "trainer")
            os.makedirs(res_dir, exist_ok=True); os.makedirs(mdl_dir, exist_ok=True)

            bs, ga = bs_ga()
            checkpoint_hub_enabled = bool(
                DEEPMATH_RUN and CKPT_RESUME and UPLOAD_MODEL and UPLOAD_RESULTS
            )
            if DEEPMATH_RUN and CKPT_RESUME and not checkpoint_hub_enabled:
                raise RuntimeError(
                    "DeepMath checkpoint resume requires both model and result uploads"
                )
            checkpoint_contract = None
            checkpoint_contract_sha256 = None
            if DEEPMATH_RUN:
                checkpoint_contract = {
                    "stage": "complete",
                    "kind": "deepmath_training_checkpoint_contract",
                    "run_slug": run_slug,
                    "base_model": MODEL,
                    "base_model_revision": MODEL_REVISION,
                    "method": method,
                    "lr": lr,
                    "epochs": ep,
                    "seed": SEED,
                    "n_train": len(train_data),
                    "train_data_sha256": train_data_sha256,
                    "schedule_validation_sha256": schedule_validation_sha256,
                    "train_maxlen": TRAIN_MAXLEN,
                    "train_bs": bs,
                    "grad_accum": ga,
                    "effective_batch": bs * ga,
                    "save_steps": TRAIN_SAVE_STEPS if CKPT_RESUME else None,
                    "optimizer": "adamw_torch",
                    "lr_scheduler": "cosine",
                    "warmup_ratio": 0.03,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "gradient_checkpointing": GRAD_CKPT,
                    "training_runtime": observed_training_runtime,
                    "lora": (
                        {
                            "r": DEEPMATH_LORA_R,
                            "alpha": DEEPMATH_LORA_ALPHA,
                            "dropout": DEEPMATH_LORA_DROPOUT,
                            "target_modules": DEEPMATH_LORA_TARGET_MODULES,
                        }
                        if method == "lora" else None
                    ),
                }
                checkpoint_contract_sha256 = _sha256_obj(checkpoint_contract)
            contract_name = "training_checkpoint_contract.json"
            contract_path = os.path.join(res_dir, contract_name)
            if DEEPMATH_RUN and CKPT_RESUME:
                existing_local_checkpoint = (
                    get_last_checkpoint(trn_dir) if os.path.isdir(trn_dir) else None
                )
                if os.path.isfile(contract_path):
                    with open(contract_path, encoding="utf-8") as handle:
                        local_contract = _json.load(handle)
                    if local_contract != checkpoint_contract:
                        raise RuntimeError(
                            "local checkpoint belongs to a different training contract"
                        )
                elif existing_local_checkpoint:
                    raise RuntimeError(
                        "local checkpoint exists without a training checkpoint contract"
                    )
                C.save_json(checkpoint_contract, contract_path)

            adapter_files = ("adapter_config.json", "adapter_model.safetensors")
            local_files = {
                name for name in os.listdir(mdl_dir)
                if os.path.isfile(os.path.join(mdl_dir, name))
            }
            local_ready = (
                _deepmath_final_model_ready(local_files, method)
                if DEEPMATH_RUN else (
                    all(name in local_files for name in adapter_files)
                    if method == "lora" else "config.json" in local_files
                )
            )
            hf_ready = False
            hf_files = set()
            if (not local_ready) and UPLOAD_MODEL and api.repo_exists(model_repo, repo_type="model"):
                hf_files = set(api.list_repo_files(model_repo, repo_type="model"))
                hf_ready = (
                    _deepmath_final_model_ready(hf_files, method)
                    if DEEPMATH_RUN else (
                        all(name in hf_files for name in adapter_files)
                        if method == "lora" else "config.json" in hf_files
                    )
                )
            remote_resume_checkpoint = None
            if not local_ready and not hf_ready and checkpoint_hub_enabled:
                create_repo(
                    model_repo, repo_type="model", private=not PUBLIC,
                    exist_ok=True, token=token,
                )
                hf_files = set(api.list_repo_files(model_repo, repo_type="model"))
                if contract_name in hf_files:
                    remote_contract_path = hf_hub_download(
                        repo_id=model_repo, filename=contract_name,
                        repo_type="model", token=token, force_download=True,
                    )
                    with open(remote_contract_path, encoding="utf-8") as handle:
                        remote_contract = _json.load(handle)
                    if remote_contract != checkpoint_contract:
                        raise RuntimeError(
                            "existing Hub checkpoint belongs to a different training contract"
                        )
                else:
                    if any(path.startswith(DEEPMATH_CHECKPOINT_PREFIX) for path in hf_files):
                        raise RuntimeError(
                            "Hub last-checkpoint exists without a training checkpoint contract"
                        )
                    commit = api.upload_file(
                        path_or_fileobj=contract_path,
                        path_in_repo=contract_name,
                        repo_id=model_repo,
                        repo_type="model",
                        commit_message=f"Bind checkpoint contract {run_slug}",
                        token=token,
                    )
                    remote_contract_path = hf_hub_download(
                        repo_id=model_repo, filename=contract_name,
                        repo_type="model", revision=commit.oid,
                        token=token, force_download=True,
                    )
                    with open(remote_contract_path, encoding="utf-8") as handle:
                        remote_contract = _json.load(handle)
                    if remote_contract != checkpoint_contract:
                        raise RuntimeError("Hub checkpoint contract readback mismatch")
                    hf_files.add(contract_name)
                if _deepmath_remote_checkpoint_ready(hf_files):
                    restore_dir = os.path.join(
                        out_dir, ".hub_checkpoint_restore",
                        checkpoint_contract_sha256[:12],
                    )
                    snapshot_download(
                        repo_id=model_repo,
                        repo_type="model",
                        token=token,
                        allow_patterns=[f"{DEEPMATH_CHECKPOINT_PREFIX}*"],
                        local_dir=restore_dir,
                    )
                    downloaded_checkpoint = os.path.join(restore_dir, "last-checkpoint")
                    _deepmath_local_checkpoint_ready(downloaded_checkpoint)
                    step = _deepmath_checkpoint_step(downloaded_checkpoint)
                    os.makedirs(trn_dir, exist_ok=True)
                    normalized_checkpoint = os.path.join(trn_dir, f"checkpoint-{step}")
                    if os.path.isdir(normalized_checkpoint):
                        _deepmath_local_checkpoint_ready(normalized_checkpoint)
                        if _deepmath_checkpoint_step(normalized_checkpoint) != step:
                            raise RuntimeError("local/Hub checkpoint step collision")
                    else:
                        os.replace(downloaded_checkpoint, normalized_checkpoint)
                        _deepmath_local_checkpoint_ready(normalized_checkpoint)
                        if _deepmath_checkpoint_step(normalized_checkpoint) != step:
                            raise RuntimeError("Hub checkpoint changed while normalizing locally")
                    remote_resume_checkpoint = normalized_checkpoint
                    print(
                        f"[RESUME] restored verified Hub last-checkpoint step={step}: "
                        f"{model_repo}",
                        flush=True,
                    )
            tm = None
            if local_ready:
                trained_dir = mdl_dir
                print(f"[RESUME] local model present, skip TRAIN: {mdl_dir}", flush=True)
            elif hf_ready:
                trained_dir = snapshot_download(model_repo, repo_type="model", token=token)
                print(f"[RESUME] HF model exists, skip TRAIN: {model_repo}", flush=True)
            else:
                _min_full = {"g4b": 78, "g1b": 20}
                if method == "full" and gpu_gb < _min_full.get(short, 0):
                    raise _GuardSkip(f"{short} full FT needs >={_min_full[short]}GB; have {gpu_gb:.0f}GB.")
                setseed(SEED)
                torch.cuda.reset_peak_memory_stats()
                train_dtype = torch.float32 if method == "full" else torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL, revision=MODEL_REVISION, torch_dtype=train_dtype,
                    attn_implementation="eager").to("cuda")
                if method == "lora":
                    model = get_peft_model(model, LoraConfig(
                        r=DEEPMATH_LORA_R,
                        lora_alpha=DEEPMATH_LORA_ALPHA,
                        lora_dropout=DEEPMATH_LORA_DROPOUT,
                        target_modules=DEEPMATH_LORA_TARGET_MODULES,
                        task_type="CAUSAL_LM"))
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                if GRAD_CKPT:
                    model.gradient_checkpointing_enable()
                model.config.use_cache = False
                tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
                tot = sum(p.numel() for p in model.parameters())
                print(f"[TRAINABLE] {tr:,}/{tot:,} ({100 * tr / max(1, tot):.3f}%) dtype={train_dtype}", flush=True)
                args = TrainingArguments(
                    output_dir=trn_dir, per_device_train_batch_size=bs, gradient_accumulation_steps=ga,
                    num_train_epochs=ep, learning_rate=lr, weight_decay=0.0, warmup_ratio=0.03,
                    lr_scheduler_type="cosine", logging_steps=10,
                    save_strategy=("steps" if CKPT_RESUME else "no"),
                    save_steps=TRAIN_SAVE_STEPS, save_total_limit=1,
                    report_to=[], bf16=True, fp16=False, remove_unused_columns=False,
                    max_grad_norm=1.0, optim="adamw_torch", seed=SEED,
                    push_to_hub=checkpoint_hub_enabled,
                    hub_model_id=(model_repo if checkpoint_hub_enabled else None),
                    hub_strategy=("checkpoint" if checkpoint_hub_enabled else "every_save"),
                    hub_private_repo=not PUBLIC,
                    hub_always_push=False,
                )
                trainer = Trainer(model=model, args=args, train_dataset=train_data, data_collator=coll)
                local_last_ckpt = (
                    get_last_checkpoint(trn_dir)
                    if CKPT_RESUME and os.path.isdir(trn_dir) else None
                )
                if DEEPMATH_RUN and local_last_ckpt:
                    # A power-loss / kill during a checkpoint write can leave the
                    # highest-numbered local checkpoint half-written. Do NOT hard-fail
                    # (that wedges every subsequent same-command resume): discard the
                    # incomplete dir and fall back to the next-best local checkpoint or
                    # the verified Hub checkpoint restored above, so resume still works.
                    _discarded = set()
                    while local_last_ckpt and not _deepmath_local_checkpoint_safe(local_last_ckpt):
                        if local_last_ckpt in _discarded:
                            local_last_ckpt = None
                            break
                        _discarded.add(local_last_ckpt)
                        print(f"[RESUME] discarding incomplete local checkpoint {local_last_ckpt}", flush=True)
                        shutil.rmtree(local_last_ckpt, ignore_errors=True)
                        local_last_ckpt = (
                            get_last_checkpoint(trn_dir)
                            if os.path.isdir(trn_dir) else None
                        )
                candidates = [
                    path for path in (local_last_ckpt, remote_resume_checkpoint) if path
                ]
                last_ckpt = (
                    _select_deepmath_checkpoint(candidates)
                    if DEEPMATH_RUN and candidates else (
                        local_last_ckpt if not DEEPMATH_RUN else None
                    )
                )
                if last_ckpt:
                    print(
                        f"[RESUME] selected checkpoint step={_deepmath_checkpoint_step(last_ckpt)} "
                        f"path={last_ckpt}",
                        flush=True,
                    )
                t0 = time.perf_counter()
                res = trainer.train(resume_from_checkpoint=last_ckpt)
                tm = dict(res.metrics)
                tm["wall_clock_seconds"] = time.perf_counter() - t0
                tm["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
                tm["trainable_params"] = int(tr); tm["train_bs"] = bs; tm["grad_accum"] = ga
                C.save_json(tm, os.path.join(res_dir, "train_metrics.json"))
                C.save_json(trainer.state.log_history, os.path.join(res_dir, "trainer_log_history.json"))
                print("[TRAIN]", tm, flush=True)
                if method == "full":
                    model.to(torch.bfloat16)
                model.save_pretrained(mdl_dir); tok.save_pretrained(mdl_dir)
                if DEEPMATH_RUN:
                    training_provenance = {
                        "stage": "complete",
                        "kind": "deepmath_training_provenance",
                        "base_model": MODEL,
                        "base_model_revision": MODEL_REVISION,
                        "method": method,
                        "lr": lr,
                        "epochs": ep,
                        "seed": SEED,
                        "train_data_sha256": train_data_sha256,
                        "schedule_validation_sha256": schedule_validation_sha256,
                        "schedule_validation": schedule_validation,
                        "checkpoint_contract": checkpoint_contract,
                        "checkpoint_contract_sha256": checkpoint_contract_sha256,
                        "checkpoint_hub_strategy": (
                            "checkpoint" if checkpoint_hub_enabled else None
                        ),
                        "train_metrics": tm,
                    }
                    C.save_json(training_provenance, os.path.join(mdl_dir, "training_provenance.json"))
                del model, trainer; gc.collect(); torch.cuda.empty_cache()
                trained_dir = mdl_dir
                if UPLOAD_MODEL and UPLOAD_RESULTS:
                    C.save_json({"repo": model_repo}, os.path.join(res_dir, "model_repo.json"))
                    create_repo(model_repo, repo_type="model", private=not PUBLIC, exist_ok=True, token=token)
                    model_commit = api.upload_folder(
                        folder_path=mdl_dir,
                        repo_id=model_repo,
                        repo_type="model",
                        commit_message=f"model {run_slug}",
                    )
                    if DEEPMATH_RUN:
                        committed_files = set(api.list_repo_files(
                            model_repo,
                            repo_type="model",
                            revision=model_commit.oid,
                            token=token,
                        ))
                        if not _deepmath_final_model_ready(committed_files, method):
                            raise RuntimeError(
                                "final DeepMath model commit lacks weights and provenance"
                            )
                        remote_provenance_path = hf_hub_download(
                            repo_id=model_repo,
                            filename="training_provenance.json",
                            repo_type="model",
                            revision=model_commit.oid,
                            token=token,
                            force_download=True,
                        )
                        local_provenance_path = os.path.join(
                            mdl_dir, "training_provenance.json"
                        )
                        if _sha256_file(remote_provenance_path) != _sha256_file(
                            local_provenance_path
                        ):
                            raise RuntimeError(
                                "final DeepMath training provenance HF readback mismatch"
                            )
                    print(
                        "[UPLOAD model]",
                        "https://huggingface.co/" + model_repo,
                        f"revision={model_commit.oid}",
                        flush=True,
                    )

            if DEEPMATH_RUN:
                provenance_path = os.path.join(trained_dir, "training_provenance.json")
                if not os.path.isfile(provenance_path):
                    raise RuntimeError("DeepMath adapter is missing training_provenance.json")
                with open(provenance_path, encoding="utf-8") as handle:
                    trained_provenance = _json.load(handle)
                expected_provenance = {
                    "base_model": MODEL,
                    "base_model_revision": MODEL_REVISION,
                    "method": method,
                    "lr": lr,
                    "epochs": ep,
                    "seed": SEED,
                    "train_data_sha256": train_data_sha256,
                    "schedule_validation_sha256": schedule_validation_sha256,
                    "checkpoint_contract_sha256": checkpoint_contract_sha256,
                }
                provenance_mismatch = {
                    key: (trained_provenance.get(key), value)
                    for key, value in expected_provenance.items()
                    if trained_provenance.get(key) != value
                }
                if provenance_mismatch:
                    raise RuntimeError(f"DeepMath adapter provenance mismatch: {provenance_mismatch}")

            # ---- optional legacy transformers diagnostic eval ----
            m = None
            if POST_TRAIN_EVAL:
                if method == "lora":
                    base_m = AutoModelForCausalLM.from_pretrained(
                        MODEL, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
                        attn_implementation="eager").to("cuda")
                    eval_model = PeftModel.from_pretrained(base_m, trained_dir).to("cuda")
                else:
                    base_m = None
                    eval_model = AutoModelForCausalLM.from_pretrained(
                        trained_dir, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda")
                eval_model.config.use_cache = True
                rec_path = os.path.join(res_dir, "finetuned_eval_records.json")
                records = C.evaluate(eval_model, tok, test, EVAL_BS, MAXLEN, MAXNEW,
                                     f"math-{short}-{method}-lr{lr:g}-ep{ep}", records_path=rec_path)
                m = C.metrics(records)
                print("[FINETUNED]", m, flush=True)
                del eval_model
                if base_m is not None:
                    del base_m
                gc.collect(); torch.cuda.empty_cache()
            else:
                print("[POST_TRAIN_EVAL] skipped by registered DeepMath protocol", flush=True)

            summary = {
                "stage": "complete",
                "kind": (
                    "technical-preflight"
                    if TECHNICAL_PREFLIGHT
                    else ("finetuned" if POST_TRAIN_EVAL else "trained-only")
                ),
                "run_regime": (
                    "technical_preflight"
                    if TECHNICAL_PREFLIGHT
                    else ("registered_deepmath" if DEEPMATH_RUN else "legacy_eval")
                ),
                "research_result_eligible": bool(DEEPMATH_RUN),
                "dataset": f"hendrycks_math_{C.MATH_LEVEL_TAG}",
                "model": MODEL, "model_revision": MODEL_REVISION, "short": short,
                "scorer": "math-verify==0.9.0 symbolic-equivalence",
                "ruler_version": C.RULER_VERSION, "instr_sha": C.INSTR_SHA, "transformers": "4.50.0",
                "fallback_contract": C.FALLBACK_CONTRACT,
                "method": method, "lr": lr, "epochs": ep, "seed": SEED, "data_tag": DATA_TAG or None,
                "math_levels": list(C.MATH_LEVELS), "math_level_tag": C.MATH_LEVEL_TAG,
                "n_train": len(train_data), "n_test": test_count, "train_maxlen": TRAIN_MAXLEN,
                "train_truncated_count": ntrunc, "train_boxed_tail_lost": boxed_tail_lost,
                "allow_train_truncation": ALLOW_TRAIN_TRUNCATION,
                "train_data_sha256": train_data_sha256,
                "schedule_validation_sha256": schedule_validation_sha256,
                "schedule_validation": schedule_validation,
                "checkpoint_contract_sha256": checkpoint_contract_sha256,
                "checkpoint_resume_enabled": CKPT_RESUME,
                "checkpoint_save_steps": (TRAIN_SAVE_STEPS if CKPT_RESUME else None),
                "checkpoint_hub_strategy": (
                    "checkpoint" if checkpoint_hub_enabled else None
                ),
                "post_train_eval": POST_TRAIN_EVAL,
                "maxlen": MAXLEN, "maxnew": MAXNEW, "eval_bs": EVAL_BS, "gpu": gpu_name,
                "results_repo": results_repo, "model_repo": (model_repo if UPLOAD_MODEL else None),
                "train_metrics": tm, "finetuned": m,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            C.save_json(summary, os.path.join(res_dir, "summary.json"))
            if UPLOAD_RESULTS:
                C.upload_results(api, res_dir, results_repo, token, public=PUBLIC, msg=f"results {run_slug}")
                print("[UPLOAD results]", "https://huggingface.co/datasets/" + results_repo, flush=True)
                shutil.rmtree(out_dir, ignore_errors=True)
            else:
                print(f"[LOCAL-ONLY] UPLOAD_RESULTS=0 -> metrics above, artifacts in {res_dir}", flush=True)
            ran += 1
        except _GuardSkip as e:
            print(f"[DEFER] {run_slug}: {e}", flush=True); deferred += 1; continue
        except Exception as e:
            traceback.print_exc()
            failures.append(f"{run_slug}: {type(e).__name__}: {e}")
            failed += 1
            try: gc.collect(); torch.cuda.empty_cache()
            except Exception: pass
            continue

    print(f"\n[GRID DONE] ran={ran} skipped={skipped} failed={failed} deferred={deferred} total={len(cells)}", flush=True)
    if failures:
        print("[FAILURES]", failures, flush=True)
    return failed, deferred


if __name__ == "__main__":
    try:
        nfail, ndef = main()
    except Exception:
        traceback.print_exc(); _shutdown("grid failed", False); raise
    else:
        _shutdown("grid succeeded", True)
        sys.exit(0 if (not nfail and not ndef) else (3 if nfail else 4))
