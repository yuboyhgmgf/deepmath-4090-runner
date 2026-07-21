#!/usr/bin/env python3
"""Serially score a manifest-verified formal vLLM MATH generation."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math_common as C
from huggingface_hub import HfApi, create_repo, login


MAXNEW = int(os.environ.get("MAXNEW", "4096"))
MAXLEN = int(os.environ.get("MAXLEN", "1024"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "5136"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    eval_path = Path(os.path.expanduser(os.environ["EVAL_SET"]))
    gen_path = Path(os.path.expanduser(os.environ["GEN"]))
    manifest_path = Path(
        os.path.expanduser(os.environ.get("GEN_MANIFEST", str(gen_path) + ".manifest.json"))
    )
    kind = os.environ.get("KIND", "baseline")
    repo = os.environ.get("REPO", f"local-{kind}")
    model_label = os.environ.get("MODEL_LABEL", "")
    public = os.environ.get("PUBLIC", "1").strip().lower() not in {"0", "false", "no"}
    upload = os.environ.get("UPLOAD", "1").strip().lower() not in {"0", "false", "no"}
    extra = json.loads(os.environ.get("EXTRA", "{}"))
    if not isinstance(extra, dict):
        raise SystemExit("EXTRA must be a JSON object")
    if (MAXNEW, MAXLEN, MAX_MODEL_LEN) != (4096, 1024, 5136):
        raise SystemExit(
            f"formal MATH ruler requires 4096/1024/5136, got {MAXNEW}/{MAXLEN}/{MAX_MODEL_LEN}"
        )
    if not C.selftest():
        raise SystemExit("Math-Verify selftest FAILED")
    if not manifest_path.is_file():
        raise SystemExit(f"generation manifest is required: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config") or {}
    expected = {
        "kind": "math_vllm_generation",
        "vllm": "0.24.0",
        "transformers": "5.12.1",
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "vllm_use_flashinfer_sampler": "0",
        "temperature": 0.0,
        "maxnew": 4096,
        "maxlen": 1024,
        "max_model_len": 5136,
        "prompt_sha": C.INSTR_SHA,
        "batch_invariant": True,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"generation manifest is not the final ruler: {mismatches}")
    if not config.get("cuda_runtime_library"):
        raise SystemExit("generation manifest is missing CUDA runtime provenance")
    capability = config.get("gpu_compute_capability") or [0, 0]
    if int(capability[0]) < 8:
        raise SystemExit(f"generation manifest records an ineligible BI GPU: CC={capability}")
    if manifest.get("stage") != "complete":
        raise SystemExit("generation manifest is not complete")
    if manifest.get("output_sha256") != sha256_file(gen_path):
        raise SystemExit("generation JSONL hash does not match its manifest")
    eval_sha256 = sha256_file(eval_path)
    if manifest.get("eval_set_sha256") != eval_sha256 or config.get("eval_set_sha256") != eval_sha256:
        raise SystemExit("eval set hash does not match generation manifest")
    config_hash = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("config_hash") != config_hash:
        raise SystemExit("generation config hash is invalid")
    generation_manifest_sha256 = sha256_file(manifest_path)

    rows = [json.loads(line) for line in eval_path.open(encoding="utf-8") if line.strip()]
    row_indices = [row["idx"] for row in rows]
    if len(row_indices) != len(set(row_indices)):
        raise SystemExit("duplicate idx in eval set")
    generations = {}
    for line in gen_path.open(encoding="utf-8"):
        if not line.strip():
            continue
        generation = json.loads(line)
        idx = generation["idx"]
        if idx in generations:
            raise SystemExit(f"duplicate idx in generation: {idx}")
        if generation.get("generation_config_hash") != config_hash:
            raise SystemExit(f"generation config hash mismatch at idx={idx}")
        if generation.get("eval_set_sha256") != eval_sha256:
            raise SystemExit(f"generation eval hash mismatch at idx={idx}")
        generations[idx] = generation
    if set(row_indices) != set(generations):
        raise SystemExit(
            f"eval/gen idx mismatch: missing={list(set(row_indices)-set(generations))[:10]} "
            f"extra={list(set(generations)-set(row_indices))[:10]}"
        )

    records = []
    for row in rows:
        generation = generations[row["idx"]]
        problem_sha256 = hashlib.sha256(str(row["problem"]).encode()).hexdigest()
        if generation.get("problem_sha256") != problem_sha256:
            raise SystemExit(f"problem hash mismatch at idx={row['idx']}")
        text = generation["text"]
        gold = C.parse_gold(row["solution"])
        if not gold:
            raise SystemExit(f"unparseable gold at idx={row['idx']}")
        strict = C.parse_strict(text)
        fallback = C.parse_fallback(text)
        has_boxed = C.last_boxed_only_string(text) is not None
        record = {
            "idx": row["idx"],
            "problem": row["problem"],
            "level": row["level"],
            "type": row["type"],
            "gold": C.boxed_inner(row["solution"]),
            "strict_pred": C.boxed_inner(text),
            "strict_correct": int(has_boxed and C.is_correct(gold, strict)),
            "fallback_correct": int(C.is_correct(gold, fallback)),
            "fallback_source": C.fallback_source(text),
            "parse_fail": int(not has_boxed),
            "generated_tokens_approx": int(generation["gen_tokens"]),
            "hit_maxnew_approx": int(generation["finish_reason"] == "length"),
            "text": text,
            "ruler_version": C.RULER_VERSION,
            "instr_sha": C.INSTR_SHA,
            "vllm_version": config["vllm"],
            "transformers_version": config["transformers"],
            "torch_version": config["torch"],
            "torch_cuda": config["torch_cuda"],
            "vllm_use_flashinfer_sampler": config["vllm_use_flashinfer_sampler"],
            "cuda_runtime_library": config["cuda_runtime_library"],
            "batch_invariant": config["batch_invariant"],
            "maxnew": config["maxnew"],
            "maxlen": config["maxlen"],
            "max_model_len": config["max_model_len"],
            "eval_set_sha256": eval_sha256,
            "ruler_config_hash": config_hash,
            "generation_manifest_sha256": generation_manifest_sha256,
        }
        if record["strict_correct"] and not record["fallback_correct"]:
            raise SystemExit(f"strict=>fallback contract broken at idx={row['idx']}")
        records.append(record)

    levels = sorted({str(row["level"]) for row in rows})
    level_nums = "".join(level.rsplit(" ", 1)[-1] for level in levels)
    dataset_tag = f"hendrycks_math_l{level_nums}"
    metrics = C.metrics(records)
    reserved = {
        "stage", "kind", "engine", "dataset", "ruler_version", "instr_sha",
        "fallback_contract", "scorer", "n_test", "maxlen", "maxnew",
        "max_model_len", "eval_set_sha256", "generation_sha256",
        "generation_manifest_sha256", "generation_config_hash",
    }
    overrides = reserved.intersection(extra)
    if overrides:
        raise SystemExit(f"EXTRA cannot override provenance fields: {sorted(overrides)}")
    token = C.read_token()
    repo_full = repo
    api = None
    if upload:
        if not token:
            raise SystemExit("HF token is required for upload")
        login(token)
        api = HfApi(token=token)
        user = api.whoami()["name"]
        repo_full = f"{user}/{repo}" if "/" not in repo else repo

    summary = {
        "stage": "complete",
        "kind": kind,
        "engine": "vllm-0.24.0-greedy-tokenids-bi",
        "dataset": dataset_tag,
        "ruler_version": C.RULER_VERSION,
        "instr_sha": C.INSTR_SHA,
        "fallback_contract": C.FALLBACK_CONTRACT,
        "scorer": "math-verify==0.9.0 symbolic-equivalence",
        "model": model_label,
        "model_revision": config.get("model_revision"),
        "merge_manifest_sha256": config.get("merge_manifest_sha256"),
        "results_repo": repo_full,
        "math_levels": levels,
        "n_test": len(records),
        "maxlen": MAXLEN,
        "maxnew": MAXNEW,
        "max_model_len": MAX_MODEL_LEN,
        "batch_invariant": True,
        "gpu_name": config.get("gpu_name"),
        "gpu_compute_capability": capability,
        "eval_set_sha256": eval_sha256,
        "generation_sha256": sha256_file(gen_path),
        "generation_manifest_sha256": generation_manifest_sha256,
        "generation_config_hash": config_hash,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        kind: metrics,
    }
    summary.update(extra)
    result_dir = Path(os.path.expanduser(f"~/vres_{repo.replace('/', '_')}"))
    C.save_json(summary, str(result_dir / "summary.json"))
    C.save_json(records, str(result_dir / "eval_records.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if upload:
        create_repo(repo_full, repo_type="dataset", private=not public, exist_ok=True, token=token)
        C.upload_results(api, str(result_dir), repo_full, token, public=public, msg=f"vllm eval {repo}")
        print("[UPLOAD]", "https://huggingface.co/datasets/" + repo_full, flush=True)


if __name__ == "__main__":
    main()
