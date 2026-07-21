#!/usr/bin/env python3
"""Generate formal MATH outputs with vLLM and a verifiable ruler manifest."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


INSTR = (
    "Solve the math problem. Please reason step by step, and put your final answer "
    "within \\boxed{{}}.\n\n{q}"
)
INSTR_SHA = hashlib.sha1(INSTR.encode()).hexdigest()[:12]
assert INSTR_SHA == "28557dc5b760", "INSTR drift vs math_common!"
EXPECTED_GEMMA_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
MAXNEW = int(os.environ.get("MAXNEW", "4096"))
MAXLEN = int(os.environ.get("MAXLEN", "1024"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "5136"))


def _prepare_vllm_cuda_runtime() -> str | None:
    candidates = [Path(root) / "nvidia" / "cu13" / "lib" for root in sys.path]
    for lib in candidates:
        if (lib / "libcudart.so.13").is_file():
            current = os.environ.get("LD_LIBRARY_PATH", "")
            entries = [value for value in current.split(":") if value]
            if str(lib) not in entries:
                os.environ["LD_LIBRARY_PATH"] = ":".join([str(lib), *entries])
                script = globals().get("__file__")
                if script and Path(script).is_file() and not os.environ.get(
                    "DEEPMATH_VLLM_CUDA13_REEXEC"
                ):
                    env = os.environ.copy()
                    env["DEEPMATH_VLLM_CUDA13_REEXEC"] = "1"
                    os.execve(sys.executable, [sys.executable, script, *sys.argv[1:]], env)
            return str(lib)
    return None


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hashes(path: Path) -> dict[str, str]:
    return {
        file.relative_to(path).as_posix(): sha256_file(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name != "merge_manifest.json"
    }


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    if (MAXNEW, MAXLEN, MAX_MODEL_LEN) != (4096, 1024, 5136):
        raise SystemExit(
            "formal MATH ruler requires 4096/1024/5136, got "
            f"{MAXNEW}/{MAXLEN}/{MAX_MODEL_LEN}"
        )
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise SystemExit("formal MATH ruler requires VLLM_BATCH_INVARIANT=1")
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise SystemExit("formal MATH ruler requires VLLM_USE_FLASHINFER_SAMPLER=0")

    cuda13_lib = _prepare_vllm_cuda_runtime()
    import torch
    import vllm
    import transformers
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if str(vllm.__version__) != "0.24.0":
        raise SystemExit(f"formal MATH ruler requires vllm==0.24.0, got {vllm.__version__}")
    if str(transformers.__version__) != "5.12.1":
        raise SystemExit(f"formal MATH ruler requires transformers==5.12.1, got {transformers.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("formal MATH ruler requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = list(torch.cuda.get_device_capability(0))
    if gpu_capability[0] < 8:
        raise SystemExit(
            f"VLLM_BATCH_INVARIANT=1 requires compute capability >=8.0; "
            f"got {gpu_name} CC={gpu_capability}"
        )

    raw_model = os.environ["MODEL"]
    model_path = Path(os.path.expanduser(raw_model))
    model = str(model_path) if model_path.exists() else raw_model
    model_revision = os.environ.get("MODEL_REVISION", "").strip() or None
    sets = [Path(os.path.expanduser(value.strip())) for value in os.environ["EVAL_SETS"].split(",") if value.strip()]
    prefix = os.path.expanduser(os.environ.get("OUT_PREFIX", "~/gen"))
    gpu_util = float(os.environ.get("GPU_UTIL", "0.90"))
    tokenizer_source = os.environ.get("TOK_FROM", model)
    tokenizer_path = Path(os.path.expanduser(tokenizer_source))
    tokenizer_source = str(tokenizer_path) if tokenizer_path.exists() else tokenizer_source

    merge_manifest = None
    merge_manifest_sha256 = None
    if model_path.exists():
        merge_path = model_path / "merge_manifest.json"
        if not merge_path.is_file():
            raise SystemExit(f"formal local model is missing merge_manifest.json: {model_path}")
        merge_manifest = json.loads(merge_path.read_text(encoding="utf-8"))
        merge_manifest_sha256 = sha256_file(merge_path)
        if merge_manifest.get("stage") != "complete" or merge_manifest.get("kind") != "lora_bf16_merge":
            raise SystemExit("local model merge manifest is not complete")
        if merge_manifest.get("output_hashes") != directory_hashes(model_path):
            raise SystemExit("local merged-model files do not match merge_manifest.json")
        merged_base_revision = merge_manifest.get("base_model_revision")
        if model_revision is not None and model_revision != merged_base_revision:
            raise SystemExit(
                f"MODEL_REVISION disagrees with merge manifest: {model_revision} != {merged_base_revision}"
            )
        model_revision = merged_base_revision
    elif raw_model == "google/gemma-3-4b-it" and model_revision != EXPECTED_GEMMA_REVISION:
        raise SystemExit(
            f"formal Gemma-3-4B baseline requires MODEL_REVISION={EXPECTED_GEMMA_REVISION}"
        )

    tokenizer_revision = None if tokenizer_path.exists() else model_revision
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, revision=tokenizer_revision)
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    stop_ids = [int(tokenizer.eos_token_id)]
    if eot is not None and eot != getattr(tokenizer, "unk_token_id", None) and int(eot) not in stop_ids:
        stop_ids.append(int(eot))

    llm_kwargs = {
        "model": model,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": gpu_util,
        "seed": 42,
    }
    if not model_path.exists():
        llm_kwargs.update({"revision": model_revision, "tokenizer_revision": model_revision})
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        # Greedy generation has no sampling RNG; keep the registered profile
        # convention (engine seed fixed, greedy sampling seed unset).
        temperature=0.0, max_tokens=MAXNEW, stop_token_ids=stop_ids, seed=None
    )

    for eval_path in sets:
        rows = [json.loads(line) for line in eval_path.open(encoding="utf-8") if line.strip()]
        indices = [row["idx"] for row in rows]
        if len(indices) != len(set(indices)):
            raise SystemExit(f"duplicate idx in eval set: {eval_path}")
        eval_sha256 = sha256_file(eval_path)
        prompts = []
        prompt_meta = []
        for row in rows:
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": INSTR.format(q=row["problem"])}],
                add_generation_prompt=True,
                tokenize=False,
            )
            all_ids = [int(value) for value in tokenizer(chat, add_special_tokens=False)["input_ids"]]
            used_ids = all_ids[-MAXLEN:]
            bos_count = used_ids.count(int(tokenizer.bos_token_id))
            # Preserve the registered left-truncation behavior exactly. A
            # prompt above MAXLEN legitimately loses its leading BOS;
            # prompt_token_ids still prevents vLLM from adding a second one.
            # Untruncated prompts must contain exactly one, and no prompt may
            # contain more than one.
            expected_bos_count = 0 if len(all_ids) > MAXLEN else 1
            if bos_count != expected_bos_count:
                raise SystemExit(
                    f"unexpected BOS count at idx={row['idx']}: bos_count={bos_count} "
                    f"expected={expected_bos_count} truncated={len(all_ids) > MAXLEN}"
                )
            prompts.append({"prompt_token_ids": used_ids})
            prompt_meta.append({
                "prompt_tokens_before_truncation": len(all_ids),
                "prompt_tokens_used": len(used_ids),
                "prompt_truncated": len(all_ids) > MAXLEN,
                "prompt_bos_count": bos_count,
                "prompt_token_ids_sha256": sha256_bytes(canonical_json(used_ids).encode()),
            })

        config = {
            "kind": "math_vllm_generation",
            "model": raw_model,
            "model_revision": model_revision,
            "merge_manifest_sha256": merge_manifest_sha256,
            "merge_training_provenance": (
                merge_manifest.get("training_provenance") if merge_manifest else None
            ),
            "tokenizer": tokenizer_source,
            "tokenizer_revision": tokenizer_revision,
            "vllm": str(vllm.__version__),
            "transformers": str(transformers.__version__),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "vllm_use_flashinfer_sampler": os.environ["VLLM_USE_FLASHINFER_SAMPLER"],
            "cuda_runtime_library": cuda13_lib,
            "dtype": "bfloat16",
            "temperature": 0.0,
            "engine_seed": 42,
            "sampling_seed": None,
            "maxnew": MAXNEW,
            "maxlen": MAXLEN,
            "max_model_len": MAX_MODEL_LEN,
            "stop_token_ids": stop_ids,
            "prompt_sha": INSTR_SHA,
            "batch_invariant": True,
            "gpu_name": gpu_name,
            "gpu_compute_capability": gpu_capability,
            "gpu_memory_utilization": gpu_util,
            "eval_set_sha256": eval_sha256,
        }
        config_hash = sha256_bytes(canonical_json(config).encode())
        tag = eval_path.name.replace("eval_", "").replace(".jsonl", "")
        out_path = Path(f"{prefix}_{tag}.jsonl")
        if out_path.exists():
            raise SystemExit(f"refusing to overwrite an existing formal generation: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        outputs = llm.generate(prompts, sampling)
        if len(outputs) != len(rows):
            raise RuntimeError(f"vLLM returned {len(outputs)} requests for {len(rows)} inputs")
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row, meta, request_output in zip(rows, prompt_meta, outputs):
                if len(request_output.outputs) != 1:
                    raise RuntimeError(f"idx={row['idx']} returned {len(request_output.outputs)} outputs")
                completion = request_output.outputs[0]
                record = {
                    "idx": row["idx"],
                    "problem_sha256": sha256_bytes(str(row["problem"]).encode()),
                    "text": completion.text,
                    "token_ids": [int(value) for value in completion.token_ids],
                    "gen_tokens": len(completion.token_ids),
                    "finish_reason": str(completion.finish_reason),
                    "generation_config_hash": config_hash,
                    "eval_set_sha256": eval_sha256,
                    **meta,
                }
                handle.write(canonical_json(record) + "\n")
        manifest = {
            "stage": "complete",
            "config": config,
            "config_hash": config_hash,
            "eval_set_path": str(eval_path.resolve()),
            "eval_set_sha256": eval_sha256,
            "output_path": str(out_path.resolve()),
            "output_sha256": sha256_file(out_path),
            "n_questions": len(rows),
            "elapsed_seconds": time.time() - started,
        }
        write_json(Path(str(out_path) + ".manifest.json"), manifest)
        print(
            f"[gen] {tag}: n={len(rows)} {(time.time() - started) / 60:.1f}min "
            f"config={config_hash} -> {out_path}",
            flush=True,
        )
    print("[gen DONE]", flush=True)


if __name__ == "__main__":
    main()
