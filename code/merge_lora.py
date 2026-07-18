#!/usr/bin/env python3
"""Merge a LoRA adapter into bf16 weights and emit a cryptographic manifest."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = os.environ["BASE"]
BASE_REVISION = os.environ.get("BASE_REVISION", "").strip() or None
ADAPTER = os.environ["ADAPTER"]
ADAPTER_REVISION = os.environ.get("ADAPTER_REVISION", "").strip() or None
OUT = Path(os.path.expanduser(os.environ.get("OUT", "~/merged_tmp")))
EXPECTED_GEMMA_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
REQUIRE_TRAINING_PROVENANCE = os.environ.get(
    "REQUIRE_TRAINING_PROVENANCE", "0"
).strip().lower() in {"1", "true", "yes"}


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


if BASE == "google/gemma-3-4b-it" and BASE_REVISION != EXPECTED_GEMMA_REVISION:
    raise SystemExit(
        f"Gemma-3-4B merge requires BASE_REVISION={EXPECTED_GEMMA_REVISION}; "
        f"got {BASE_REVISION}"
    )
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(f"OUT must be absent or empty to prevent stale merged shards: {OUT}")
OUT.mkdir(parents=True, exist_ok=True)

adapter_path = Path(os.path.expanduser(ADAPTER))
adapter_repo_sha = None
adapter_input_hashes = None
if adapter_path.exists():
    adapter_input_hashes = directory_hashes(adapter_path)
else:
    adapter_repo_sha = HfApi().model_info(ADAPTER, revision=ADAPTER_REVISION).sha

print(
    f"[merge] base={BASE}@{BASE_REVISION} adapter={ADAPTER}@{adapter_repo_sha or ADAPTER_REVISION}",
    flush=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE, revision=BASE_REVISION, torch_dtype=torch.bfloat16, device_map="cpu"
)
resolved_adapter_revision = adapter_repo_sha or ADAPTER_REVISION
model = PeftModel.from_pretrained(model, ADAPTER, revision=resolved_adapter_revision)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION).save_pretrained(OUT)

training_provenance = None
if adapter_path.exists() and (adapter_path / "training_provenance.json").is_file():
    training_provenance = json.loads(
        (adapter_path / "training_provenance.json").read_text(encoding="utf-8")
    )
elif adapter_repo_sha:
    try:
        provenance_path = hf_hub_download(
            repo_id=ADAPTER,
            filename="training_provenance.json",
            revision=adapter_repo_sha,
        )
        training_provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    except Exception as exc:
        if REQUIRE_TRAINING_PROVENANCE:
            raise SystemExit(f"remote adapter is missing training_provenance.json: {exc}") from exc
if REQUIRE_TRAINING_PROVENANCE and training_provenance is None:
    raise SystemExit("DeepMath merge requires training_provenance.json")
manifest = {
    "stage": "complete",
    "kind": "lora_bf16_merge",
    "base_model": BASE,
    "base_model_revision": BASE_REVISION,
    "adapter": ADAPTER,
    "adapter_revision": resolved_adapter_revision,
    "adapter_input_hashes": adapter_input_hashes,
    "training_provenance": training_provenance,
    "dtype": "bfloat16",
    "output_hashes": directory_hashes(OUT),
}
tmp = OUT / "merge_manifest.json.tmp"
tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, OUT / "merge_manifest.json")
print(f"[merge] -> {OUT} files={len(manifest['output_hashes'])}", flush=True)
