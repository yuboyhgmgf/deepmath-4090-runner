#!/usr/bin/env python3
"""Serially score DeepMath profiles and build nested, token-matched schedules.

Math-Verify runs only in this process' main thread.  The builder refuses missing
or duplicate generations, test/development leakage, non-nested subsets, unequal
row counts, distribution drift, and assistant-token budget drift above 2%.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import deepmath_common as D


GENERATION_PAIR_PROVENANCE_FIELDS = (
    "model", "vllm", "transformers", "torch", "torch_cuda",
    "vllm_use_flashinfer_sampler", "cuda_runtime_library", "dtype",
    "maxnew", "maxlen", "max_model_len", "prompt_sha",
    "difficulty_min", "difficulty_max", "stop_token_ids", "batch_invariant",
    "enable_prefix_caching", "vllm_cache_root", "compile_cache_id",
    "compile_cache_phase", "request_batch_size", "gpu_memory_utilization",
    "tokenizer", "model_revision", "tokenizer_revision", "gpu_name",
    "gpu_compute_capability", "prepared_sha256",
)

REPLAY_DEDUP_POLICY = "question_then_template_input_order_v1"


def generation_pair_provenance_mismatches(
    greedy_config: dict[str, Any], sampled_config: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Return every ruler/runtime field that differs across the two passes."""
    return {
        key: (greedy_config.get(key), sampled_config.get(key))
        for key in GENERATION_PAIR_PROVENANCE_FIELDS
        if greedy_config.get(key) != sampled_config.get(key)
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True)
    p.add_argument("--greedy", required=True)
    p.add_argument("--sampled", required=True)
    p.add_argument(
        "--profile-orchestrator-manifest",
        help="required in formal mode; proves both passes used the validated warm compile cache",
    )
    p.add_argument("--replay-pool", required=True)
    p.add_argument("--internal-dev", required=True)
    p.add_argument(
        "--preparation-parent-manifest",
        help=(
            "required for a compression-supplement prepared corpus; proves the "
            "inherited replay/internal-dev artifacts without rewriting the immutable "
            "supplement manifest"
        ),
    )
    p.add_argument("--outdir", required=True)
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--small-size", type=int)
    p.add_argument("--large-size", type=int)
    p.add_argument("--dm-exposures", type=int, default=40_000)
    p.add_argument("--replay-exposures", type=int, default=10_000)
    p.add_argument("--schedule-seed", type=int, default=42)
    return p.parse_args()


def resolve_preparation_artifact_manifest(
    preparation_manifest: dict[str, Any],
    parent_manifest_path: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Return the manifest that owns replay/internal-dev artifact hashes.

    The deterministic compression supplement intentionally wrote a new prepared
    artifact but referenced the immutable base preparation by SHA.  Its legacy
    manifest therefore does not duplicate the base replay/internal-dev entries.
    Formal scoring must follow and verify that parent link instead of accepting
    unbound files or failing after the expensive profile has completed.
    """
    kind = preparation_manifest.get("kind")
    if kind == "deepmath_preparation":
        return preparation_manifest, None
    if kind != "deepmath_preparation_with_compression_supplement":
        raise SystemExit(f"unsupported preparation manifest kind: {kind!r}")
    if not parent_manifest_path:
        raise SystemExit(
            "compression-supplement scoring requires --preparation-parent-manifest"
        )
    parent_path = Path(parent_manifest_path)
    try:
        parent_manifest = json.loads(parent_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid preparation parent manifest: {parent_path}") from error
    expected_parent_sha = (
        (preparation_manifest.get("parent") or {}).get("preparation_manifest_sha256")
    )
    actual_parent_sha = D.sha256_file(parent_path)
    if expected_parent_sha != actual_parent_sha:
        raise SystemExit(
            "preparation parent manifest hash mismatch: "
            f"expected={expected_parent_sha} actual={actual_parent_sha}"
        )
    if (
        parent_manifest.get("stage"), parent_manifest.get("kind")
    ) != ("complete", "deepmath_preparation"):
        raise SystemExit("preparation parent manifest is not a complete base preparation")
    parent_config = parent_manifest.get("config") or {}
    if parent_manifest.get("config_hash") != D.config_hash(parent_config):
        raise SystemExit("preparation parent config hash is invalid")
    normalized_parent_config = dict(parent_config)
    supplement_config = dict(preparation_manifest.get("config") or {})
    supplement_config.pop("compression_supplement", None)
    if parent_config.get("student_model_revision") is None:
        supplement_config.pop("student_model_revision", None)
        normalized_parent_config.pop("student_model_revision", None)
    if supplement_config != normalized_parent_config:
        raise SystemExit("supplement preparation config does not inherit its parent config")
    parent_prepared_sha = (
        (parent_manifest.get("artifacts") or {}).get("prepared", {}).get("sha256")
    )
    recorded_base_sha = (
        (preparation_manifest.get("parent") or {}).get("base_prepared_sha256")
    )
    if parent_prepared_sha != recorded_base_sha:
        raise SystemExit("supplement base-prepared hash does not match its parent manifest")
    for key in ("replay_pool", "internal_dev"):
        inherited_sha = (
            (parent_manifest.get("artifacts") or {}).get(key, {}).get("sha256")
        )
        if not isinstance(inherited_sha, str) or len(inherited_sha) != 64:
            raise SystemExit(f"preparation parent lacks a valid {key} artifact hash")
    return parent_manifest, actual_parent_sha


def _one_per_hash(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = str(row["row_hash"])
        if key in result:
            raise ValueError(f"duplicate {label} row_hash: {key}")
        result[key] = row
    return result


def _generation_map(rows: Sequence[dict[str, Any]], mode: str, n: int):
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    config_hashes = set()
    for row in rows:
        if row.get("mode") != mode:
            raise ValueError(f"expected mode={mode}, got {row.get('mode')!r}")
        row_hash, sample_id = str(row["row_hash"]), int(row["sample_id"])
        if sample_id in grouped[row_hash]:
            raise ValueError(f"duplicate {mode} completion: row={row_hash} sample={sample_id}")
        grouped[row_hash][sample_id] = row
        config_hashes.add(row.get("generation_config_hash"))
    if len(config_hashes) != 1 or None in config_hashes:
        raise ValueError(f"{mode} generation must have exactly one non-null config hash")
    expected = set(range(n))
    for row_hash, samples in grouped.items():
        if set(samples) != expected:
            raise ValueError(
                f"{mode} row {row_hash} completion closure failed: "
                f"missing={sorted(expected - set(samples))} extra={sorted(set(samples) - expected)}"
            )
    return grouped, next(iter(config_hashes))


def score_profiles(
    prepared: Sequence[dict[str, Any]],
    greedy_rows: Sequence[dict[str, Any]],
    sampled_rows: Sequence[dict[str, Any]],
    C,
):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Math-Verify scoring must run in the main thread")
    prepared_by_hash = _one_per_hash(prepared, "prepared")
    greedy, greedy_cfg = _generation_map(greedy_rows, "greedy", 1)
    sampled, sampled_cfg = _generation_map(sampled_rows, "sampled", D.SAMPLED_K)
    if set(greedy) != set(sampled):
        raise ValueError(
            f"greedy/sample question mismatch: missing_sampled={len(set(greedy)-set(sampled))} "
            f"missing_greedy={len(set(sampled)-set(greedy))}"
        )
    unknown = set(greedy) - set(prepared_by_hash)
    if unknown:
        raise ValueError(f"generation contains {len(unknown)} rows not present in prepared corpus")

    difficulty_rows, completion_rows = [], []
    for row_hash in sorted(greedy):
        source = prepared_by_hash[row_hash]
        gold = C.parse_gold(D.gold_boxed_text(source["final_answer"]))
        if not gold:
            raise ValueError(f"prepared row has unparseable gold: {row_hash}")
        scored = []
        for mode, samples in (("greedy", greedy[row_hash]), ("sampled", sampled[row_hash])):
            for sample_id in sorted(samples):
                generation = samples[sample_id]
                if generation.get("question_hash") != source.get("question_hash"):
                    raise ValueError(f"question hash mismatch for {row_hash}/{mode}/{sample_id}")
                text = str(generation.get("text") or "")
                strict = C.parse_strict(text)
                fallback = C.parse_fallback(text)
                strict_correct = int(C.is_correct(gold, strict))
                fallback_correct = int(C.is_correct(gold, fallback))
                if strict_correct and not fallback_correct:
                    raise AssertionError(f"strict=>fallback contract broken for {row_hash}/{mode}/{sample_id}")
                item = {
                    **generation,
                    "strict_correct": strict_correct,
                    "fallback_correct": fallback_correct,
                    "format_gap": fallback_correct - strict_correct,
                    "fallback_source": C.fallback_source(text),
                }
                completion_rows.append(item)
                scored.append(item)
        greedy_score = next(x for x in scored if x["mode"] == "greedy")
        sample_scores = [x for x in scored if x["mode"] == "sampled"]
        correct_at_8 = sum(x["fallback_correct"] for x in sample_scores)
        if greedy_score["fallback_correct"]:
            difficulty_bin = "anchor"
        elif correct_at_8:
            difficulty_bin = "frontier"
        else:
            difficulty_bin = "hard"
        difficulty_rows.append({
            **source,
            "difficulty_bin": difficulty_bin,
            "greedy_strict_correct": greedy_score["strict_correct"],
            "greedy_fallback_correct": greedy_score["fallback_correct"],
            "greedy_format_gap": greedy_score["format_gap"],
            "greedy_gen_tokens": int(greedy_score["gen_tokens"]),
            "greedy_finish_reason": greedy_score["finish_reason"],
            "sampled_correct_at_8": correct_at_8,
            "sampled_strict_correct_at_8": sum(x["strict_correct"] for x in sample_scores),
            "sampled_gen_tokens": [int(x["gen_tokens"]) for x in sample_scores],
            "sampled_finish_reasons": [x["finish_reason"] for x in sample_scores],
            "greedy_generation_config_hash": greedy_cfg,
            "sampled_generation_config_hash": sampled_cfg,
            "split": "difficulty_profile",
        })
    deciles = D.assistant_token_deciles(difficulty_rows)
    prompt_deciles = D.token_deciles(difficulty_rows, "prompt_tokens")
    for row in difficulty_rows:
        row["solution_token_decile"] = deciles[row["row_hash"]]
        row["prompt_token_decile"] = prompt_deciles[row["row_hash"]]
        row["difficulty_bucket"] = D.difficulty_bucket(row["difficulty"])
    return difficulty_rows, completion_rows


def _proportional_nested_take(rows: Sequence[dict[str, Any]], n: int, fields: Sequence[str], salt: str):
    if n > len(rows):
        raise ValueError(f"cannot take nested {n} from {len(rows)}")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    allocations: dict[tuple[Any, ...], int] = {}
    remainders = []
    for key, group in groups.items():
        exact = n * len(group) / len(rows)
        base = math.floor(exact)
        allocations[key] = base
        remainders.append((exact - base, D.canonical_json(key), key))
    remaining = n - sum(allocations.values())
    for _, _, key in sorted(remainders, reverse=True)[:remaining]:
        allocations[key] += 1
    selected = []
    for key in sorted(groups, key=D.canonical_json):
        selected.extend(D.stable_order(groups[key], f"{salt}:{D.canonical_json(key)}")[:allocations[key]])
    if len(selected) != n:
        raise AssertionError(f"nested allocation produced {len(selected)} != {n}")
    return D.stable_order(selected, salt + ":final")


def select_master(rows: Sequence[dict[str, Any]], target: int):
    by_bin = {name: [r for r in rows if r["difficulty_bin"] == name]
              for name in ("anchor", "frontier", "hard")}
    hard_cap = math.floor(target * 0.20)
    counts = {
        "hard": min(hard_cap, len(by_bin["hard"])),
        "frontier": min(math.floor(target * 0.60), len(by_bin["frontier"])),
        "anchor": min(target - math.floor(target * 0.60) - hard_cap, len(by_bin["anchor"])),
    }
    remaining = target - sum(counts.values())
    for name in ("frontier", "anchor"):
        add = min(remaining, len(by_bin[name]) - counts[name])
        counts[name] += add
        remaining -= add
    if remaining:
        raise ValueError(
            f"cannot fill {target} without exceeding hard<=20%: counts_available="
            f"{ {k: len(v) for k, v in by_bin.items()} } missing={remaining}"
        )
    fields = ("topic", "solution_token_decile", "prompt_token_decile", "difficulty_bucket")
    selected = []
    for name in ("anchor", "frontier", "hard"):
        selected.extend(_proportional_nested_take(by_bin[name], counts[name], fields, f"master-{name}"))
    if sum(r["difficulty_bin"] == "hard" for r in selected) > hard_cap:
        raise AssertionError("hard proportion exceeded 20%")
    return D.stable_order(selected, "master-final"), counts


def _expand_exposures(rows: Sequence[dict[str, Any]], total: int, source: str):
    if not rows or total % len(rows):
        raise ValueError(f"{source} exposures={total} must be divisible by unique rows={len(rows)}")
    repeats = total // len(rows)
    out = []
    for repeat_id in range(repeats):
        for row in rows:
            out.append({
                "source": source,
                "source_row_hash": row["row_hash"],
                "repeat_id": repeat_id,
                "question": row.get("question", row.get("problem")),
                "answer": row.get("chosen_solution", row.get("answer", row.get("solution"))),
                "prompt_tokens": int(row["prompt_tokens"]),
                "assistant_tokens": int(row["assistant_tokens"]),
                "train_tokens": int(row["train_tokens"]),
            })
    return out


def deduplicate_replay_pool(
    replay_pool: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int | str]]:
    """Deterministically remove exact and template-family replay duplicates.

    Input order is part of the pinned replay-pool artifact hash.  Keep the first
    row for each normalized question and then the first row for each template
    family.  The resulting audit is written into both arm schedules so repeated
    *exposures* cannot hide duplicated unique replay sources.
    """
    if not replay_pool:
        raise ValueError("empty replay pool")
    seen_row_hashes: set[str] = set()
    seen_questions: set[str] = set()
    seen_families: set[str] = set()
    kept: list[dict[str, Any]] = []
    dropped_exact = 0
    dropped_family = 0
    for index, row in enumerate(replay_pool):
        row_hash = str(row.get("row_hash") or "")
        question = str(row.get("question") or "")
        question_hash = str(row.get("question_hash") or "")
        if not row_hash or not question or not question_hash:
            raise ValueError(f"replay row {index} lacks row/question/question_hash identity")
        if row_hash in seen_row_hashes:
            raise ValueError(f"duplicate replay row_hash: {row_hash}")
        seen_row_hashes.add(row_hash)
        expected_question_hash = D.normalized_question_hash(question)
        if question_hash != expected_question_hash:
            raise ValueError(
                f"replay question_hash drift at row {index}: "
                f"recorded={question_hash} expected={expected_question_hash}"
            )
        family_hash = D.template_hash(question)
        if question_hash in seen_questions:
            dropped_exact += 1
            continue
        if family_hash in seen_families:
            dropped_family += 1
            continue
        seen_questions.add(question_hash)
        seen_families.add(family_hash)
        kept.append({**row, "replay_family_hash": family_hash})
    if not kept:
        raise ValueError("replay deduplication removed every row")
    audit: dict[str, int | str] = {
        "replay_dedup_policy": REPLAY_DEDUP_POLICY,
        "replay_input_rows": len(replay_pool),
        "replay_kept_rows": len(kept),
        "replay_dropped_exact": dropped_exact,
        "replay_dropped_family": dropped_family,
        "replay_unique_question_hashes": len(seen_questions),
        "replay_unique_family_hashes": len(seen_families),
    }
    if len(kept) + dropped_exact + dropped_family != len(replay_pool):
        raise AssertionError("replay deduplication count closure failed")
    return kept, audit


def _replay_exposures(replay_pool: Sequence[dict[str, Any]], total: int):
    if not replay_pool:
        raise ValueError("empty replay pool")
    ordered = D.stable_order(replay_pool, "replay-v1")
    out = []
    for exposure_id in range(total):
        row = ordered[exposure_id % len(ordered)]
        out.append({
            "source": "math_replay",
            "source_row_hash": row["row_hash"],
            "source_question_hash": row["question_hash"],
            "source_family_hash": row["replay_family_hash"],
            "repeat_id": exposure_id // len(ordered),
            "question": row["question"],
            "answer": row["answer"],
            "prompt_tokens": int(row["prompt_tokens"]),
            "assistant_tokens": int(row["assistant_tokens"]),
            "train_tokens": int(row["train_tokens"]),
        })
    return out


def build_schedules(
    master: Sequence[dict[str, Any]],
    small_size: int,
    dm_exposures: int,
    replay_pool: Sequence[dict[str, Any]],
    replay_exposure_count: int,
    schedule_seed: int,
    small_selector=None,
):
    fields = (
        "difficulty_bin", "topic", "solution_token_decile",
        "prompt_token_decile", "difficulty_bucket",
    )
    if small_selector is None:
        small = _proportional_nested_take(master, small_size, fields, "small-nested-v1")
    else:
        small = small_selector(master, small_size)
    if not {r["row_hash"] for r in small}.issubset({r["row_hash"] for r in master}):
        raise AssertionError("small corpus is not nested in master")
    gaps = {field: D.max_distribution_gap(small, master, field) for field in fields}
    if any(value > 0.03 + 1e-12 for value in gaps.values()):
        raise ValueError(f"small/master stratification drift exceeds 3%: {gaps}")

    unique_replay, replay_audit = deduplicate_replay_pool(replay_pool)
    common_replay = _replay_exposures(unique_replay, replay_exposure_count)
    small_schedule = _expand_exposures(small, dm_exposures, "deepmath_small") + [dict(x) for x in common_replay]
    large_schedule = _expand_exposures(master, dm_exposures, "deepmath_large") + [dict(x) for x in common_replay]
    if len(small_schedule) != len(large_schedule):
        raise AssertionError("small/large schedule row counts differ")
    small_tokens = sum(row["assistant_tokens"] for row in small_schedule)
    large_tokens = sum(row["assistant_tokens"] for row in large_schedule)
    token_gap = abs(small_tokens - large_tokens) / max(1, max(small_tokens, large_tokens))
    if token_gap > 0.02 + 1e-12:
        raise ValueError(
            f"assistant token budget gap {token_gap:.4%} exceeds 2% "
            f"(small={small_tokens}, large={large_tokens})"
        )
    small_prompt_tokens = sum(row["prompt_tokens"] for row in small_schedule)
    large_prompt_tokens = sum(row["prompt_tokens"] for row in large_schedule)
    prompt_token_gap = abs(small_prompt_tokens - large_prompt_tokens) / max(
        1, max(small_prompt_tokens, large_prompt_tokens)
    )
    small_train_tokens = sum(row["train_tokens"] for row in small_schedule)
    large_train_tokens = sum(row["train_tokens"] for row in large_schedule)
    train_token_gap = abs(small_train_tokens - large_train_tokens) / max(
        1, max(small_train_tokens, large_train_tokens)
    )
    if prompt_token_gap > 0.02 + 1e-12 or train_token_gap > 0.02 + 1e-12:
        raise ValueError(
            "training-compute token budget gap exceeds 2%: "
            f"prompt={prompt_token_gap:.4%} train={train_token_gap:.4%} "
            f"small_prompt={small_prompt_tokens} large_prompt={large_prompt_tokens} "
            f"small_train={small_train_tokens} large_train={large_train_tokens}"
        )
    random.Random(schedule_seed).shuffle(small_schedule)
    random.Random(schedule_seed).shuffle(large_schedule)
    for arm, schedule in (("dm_small", small_schedule), ("dm_large", large_schedule)):
        for schedule_idx, row in enumerate(schedule):
            row["schedule_idx"] = schedule_idx
            row["split"] = "train"
            row["arm"] = arm
    return small, small_schedule, large_schedule, {
        "small_assistant_tokens": small_tokens,
        "large_assistant_tokens": large_tokens,
        "assistant_token_gap_fraction": token_gap,
        "small_prompt_tokens": small_prompt_tokens,
        "large_prompt_tokens": large_prompt_tokens,
        "prompt_token_gap_fraction": prompt_token_gap,
        "small_train_tokens": small_train_tokens,
        "large_train_tokens": large_train_tokens,
        "train_token_gap_fraction": train_token_gap,
        "distribution_gaps": gaps,
        **replay_audit,
    }


def _write_training(outdir: Path, name: str, schedule: Sequence[dict[str, Any]]):
    train_path = outdir / f"{name}_train.jsonl"
    manifest_path = outdir / f"{name}_schedule_manifest.jsonl"
    D.write_jsonl(train_path, ({"question": r["question"], "answer": r["answer"]} for r in schedule))
    D.write_jsonl(manifest_path, schedule)
    return train_path, manifest_path


def _read_generation_manifest(path: str):
    manifest_path = Path(path + ".manifest.json")
    return json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None


def validate_profile_orchestrator(
    path: str,
    *,
    prepared_path: str,
    greedy_path: str,
    sampled_path: str,
    greedy_manifest: dict[str, Any],
    sampled_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Verify the log-audited closure manifest for the two formal passes."""
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid formal profile orchestrator manifest: {manifest_path}") from error
    if not isinstance(value, dict) or (
        value.get("stage"), value.get("kind")
    ) != ("complete", "deepmath_formal_profile_orchestrator"):
        raise SystemExit("formal profile orchestrator manifest is not complete")
    if value.get("formal_profile_will_not_start_another_stage") is not True:
        raise SystemExit("formal profile orchestrator manifest lacks its closure marker")
    prepared_sha = D.sha256_file(prepared_path)
    if value.get("prepared_sha256") != prepared_sha:
        raise SystemExit("formal profile orchestrator points to a different prepared corpus")
    cache_id = value.get("compile_cache_id")
    cache_root = value.get("vllm_cache_root")
    if not cache_id or not cache_root or value.get("compile_cache_phase") != "formal_measurement":
        raise SystemExit("formal profile orchestrator lacks validated compile-cache identity")
    modes = value.get("modes")
    if not isinstance(modes, dict):
        raise SystemExit("formal profile orchestrator lacks per-mode validation")
    local = {
        "greedy": (greedy_path, greedy_manifest, 1),
        "sampled": (sampled_path, sampled_manifest, D.SAMPLED_K),
    }
    question_counts = set()
    for mode, (output_path, generation_manifest, completions_per_question) in local.items():
        audit = modes.get(mode)
        if not isinstance(audit, dict) or audit.get("mode") != mode:
            raise SystemExit(f"formal profile orchestrator lacks {mode} audit")
        generation_manifest_path = Path(output_path + ".manifest.json")
        expected = {
            "output_sha256": D.sha256_file(output_path),
            "manifest_sha256": D.sha256_file(generation_manifest_path),
            "n_questions": generation_manifest.get("n_questions"),
            "n_completions": generation_manifest.get("n_completions"),
            "compile_count": 0,
            "cold_marker_count": 0,
        }
        drift = {
            key: (audit.get(key), wanted)
            for key, wanted in expected.items()
            if audit.get(key) != wanted
        }
        if drift:
            raise SystemExit(f"formal profile orchestrator {mode} audit mismatch: {drift}")
        if not isinstance(audit.get("load_count"), int) or audit["load_count"] < 1:
            raise SystemExit(f"formal profile orchestrator {mode} lacks direct-load evidence")
        n_questions = generation_manifest.get("n_questions")
        if not isinstance(n_questions, int) or n_questions < 1:
            raise SystemExit(f"formal {mode} manifest has invalid question count")
        if generation_manifest.get("n_completions") != n_questions * completions_per_question:
            raise SystemExit(f"formal {mode} generation count closure failed")
        config = generation_manifest.get("config") or {}
        if config.get("compile_cache_id") != cache_id or config.get("vllm_cache_root") != cache_root:
            raise SystemExit(f"formal {mode} generation does not match orchestrator compile cache")
        question_counts.add(n_questions)
    if len(question_counts) != 1:
        raise SystemExit("formal profile orchestrator greedy/sampled question counts differ")
    return value


def main() -> None:
    args = parse_args()
    if not args.fixture and any(v is not None for v in (args.small_size, args.large_size)):
        raise SystemExit("formal corpus sizes are decided by the registered 4K/20K or 2K/10K rule")
    if not args.fixture and (args.dm_exposures, args.replay_exposures, args.schedule_seed) != (40_000, 10_000, 42):
        raise SystemExit("formal exposures/schedule seed are pinned to 40K/10K/42")

    import math_common as C

    if not C.selftest():
        raise SystemExit("Math-Verify selftest failed; refusing to score")
    prepared = D.read_jsonl(args.prepared)
    greedy_rows = D.read_jsonl(args.greedy)
    sampled_rows = D.read_jsonl(args.sampled)
    replay_pool = D.read_jsonl(args.replay_pool)
    internal_dev = D.read_jsonl(args.internal_dev)
    if {r["question_hash"] for r in replay_pool} & {r["question_hash"] for r in internal_dev}:
        raise SystemExit("internal dev leaked into replay pool")
    dev_question_hashes = {r["question_hash"] for r in internal_dev}
    if {r["question_hash"] for r in prepared} & dev_question_hashes:
        raise SystemExit("internal dev exact match leaked into prepared DeepMath pool")
    dev_family_hashes = {D.template_hash(r["problem"]) for r in internal_dev}
    if {r["family_hash"] for r in prepared} & dev_family_hashes:
        raise SystemExit("internal dev template match leaked into prepared DeepMath pool")

    greedy_manifest = _read_generation_manifest(args.greedy)
    sampled_manifest = _read_generation_manifest(args.sampled)
    preparation_manifest_sha = None
    preparation_parent_manifest_sha = None
    if not args.fixture and (greedy_manifest is None or sampled_manifest is None):
        raise SystemExit("formal scoring requires both generation manifests")
    if not args.fixture and not args.profile_orchestrator_manifest:
        raise SystemExit("formal scoring requires --profile-orchestrator-manifest")
    if not args.fixture:
        preparation_manifest_path = Path(args.prepared).parent / "preparation_manifest.json"
        if not preparation_manifest_path.exists():
            raise SystemExit("formal scoring requires preparation_manifest.json beside prepared JSONL")
        preparation_manifest = json.loads(preparation_manifest_path.read_text(encoding="utf-8"))
        preparation_manifest_sha = D.sha256_file(preparation_manifest_path)
        allowed_prep_kinds = {
            "deepmath_preparation",
            "deepmath_preparation_with_compression_supplement",
        }
        if (
            preparation_manifest.get("stage") != "complete"
            or preparation_manifest.get("kind") not in allowed_prep_kinds
        ):
            raise SystemExit(
                "prepared corpus manifest is not a complete registered preparation: "
                f"stage={preparation_manifest.get('stage')} kind={preparation_manifest.get('kind')}"
            )
        expected_prepared_sha = (
            preparation_manifest.get("artifacts", {}).get("prepared", {}).get("sha256")
        )
        if expected_prepared_sha != D.sha256_file(args.prepared):
            raise SystemExit("prepared JSONL does not match preparation_manifest.json")
        prep_cfg = preparation_manifest.get("config", {})
        required_prep = {
            "formal": True,
            "deepmath_dataset": D.DEEPMATH_DATASET,
            "deepmath_revision": D.DEEPMATH_REVISION,
            "student_model": D.STUDENT_MODEL,
            "train_maxlen": D.TRAIN_MAXLEN,
            "internal_dev_size": 500,
            "decontaminate_against_internal_dev": True,
            "char_5gram_threshold": 0.75,
            "embedding_threshold": 0.88,
            "embedding_model": D.EMBEDDING_MODEL,
            "embedding_revision": D.EMBEDDING_REVISION,
            "embedding_skipped": False,
            "prompt_sha": D.INSTR_SHA,
            "math_verify": "0.9.0",
        }
        prep_mismatches = {k: (prep_cfg.get(k), v) for k, v in required_prep.items() if prep_cfg.get(k) != v}
        if prep_mismatches:
            raise SystemExit(f"prepared corpus is not the pinned formal build: {prep_mismatches}")
        recorded_student_revision = prep_cfg.get("student_model_revision")
        if recorded_student_revision not in (None, D.STUDENT_REVISION):
            raise SystemExit(
                "prepared corpus records an unexpected student-model revision: "
                f"{recorded_student_revision}"
            )
        if preparation_manifest.get("config_hash") != D.config_hash(prep_cfg):
            raise SystemExit("preparation config hash is invalid")
        recorded_prepared_sha = preparation_manifest.get("artifacts", {}).get("prepared", {}).get("sha256")
        if recorded_prepared_sha != D.sha256_file(args.prepared):
            raise SystemExit("prepared JSONL hash does not match preparation manifest")
        artifact_manifest, preparation_parent_manifest_sha = (
            resolve_preparation_artifact_manifest(
                preparation_manifest,
                args.preparation_parent_manifest,
            )
        )
        for key, path in (("replay_pool", args.replay_pool), ("internal_dev", args.internal_dev)):
            recorded = artifact_manifest.get("artifacts", {}).get(key, {}).get("sha256")
            if recorded != D.sha256_file(path):
                raise SystemExit(f"{key} hash does not match preparation manifest")
    if greedy_manifest and sampled_manifest:
        gcfg, scfg = greedy_manifest["config"], sampled_manifest["config"]
        mismatches = generation_pair_provenance_mismatches(gcfg, scfg)
        if mismatches:
            raise SystemExit(f"greedy/sampled generation provenance mismatch: {mismatches}")
        if (gcfg["maxnew"], gcfg["maxlen"], gcfg["max_model_len"], gcfg["prompt_sha"]) != (
            D.MAXNEW, D.MAXLEN, D.MAX_MODEL_LEN, D.INSTR_SHA
        ):
            raise SystemExit("generation does not use the final MATH ruler")
        if (gcfg.get("model"), gcfg.get("vllm"), gcfg.get("batch_invariant")) != (
            D.STUDENT_MODEL, "0.24.0", True
        ):
            raise SystemExit("generation is not frozen Gemma-3-4B on vLLM 0.24.0 with BI")
        if gcfg.get("transformers") != D.VLLM_TRANSFORMERS_VERSION:
            raise SystemExit(
                f"generation uses unexpected vLLM tokenizer runtime: {gcfg.get('transformers')}"
            )
        if (gcfg.get("torch"), gcfg.get("torch_cuda")) != (
            D.VLLM_TORCH_VERSION, D.VLLM_TORCH_CUDA
        ):
            raise SystemExit("generation does not use the pinned Torch CUDA-13 runtime")
        if gcfg.get("vllm_use_flashinfer_sampler") != D.VLLM_USE_FLASHINFER_SAMPLER:
            raise SystemExit("generation does not use the registered FlashInfer sampler setting")
        if not gcfg.get("cuda_runtime_library") or not scfg.get("cuda_runtime_library"):
            raise SystemExit("generation manifest does not record the CUDA runtime library")
        if gcfg.get("dtype") != "bfloat16":
            raise SystemExit(f"generation uses unexpected dtype: {gcfg.get('dtype')!r}")
        if gcfg.get("enable_prefix_caching") is not False:
            raise SystemExit("formal generation must use the registered prefix-caching-disabled ruler")
        if not gcfg.get("vllm_cache_root") or not gcfg.get("compile_cache_id"):
            raise SystemExit("formal generation does not identify its validated compile cache")
        if gcfg.get("compile_cache_phase") != "formal_measurement":
            raise SystemExit(
                "formal generation was not produced in the registered warm-cache measurement phase"
            )
        if (gcfg.get("model_revision"), gcfg.get("tokenizer_revision")) != (
            D.STUDENT_REVISION, D.STUDENT_REVISION
        ):
            raise SystemExit("generation does not use the pinned Gemma-3-4B model/tokenizer revision")
        capability = gcfg.get("gpu_compute_capability") or [0, 0]
        if int(capability[0]) < 9:
            raise SystemExit(f"generation BI GPU is ineligible: CC={capability}")
        if not args.fixture and (not gcfg.get("formal") or not scfg.get("formal")):
            raise SystemExit("smoke generations may not be promoted into formal schedules")
        for path, manifest, mode in (
            (args.greedy, greedy_manifest, "greedy"),
            (args.sampled, sampled_manifest, "sampled"),
        ):
            if manifest.get("output_sha256") != D.sha256_file(path):
                raise SystemExit(f"{mode} output hash does not match generation manifest")
            if manifest.get("prepared_sha256") != D.sha256_file(args.prepared):
                raise SystemExit(f"{mode} manifest points to a different prepared corpus")
            if manifest.get("config_hash") != D.config_hash(manifest.get("config", {})):
                raise SystemExit(f"{mode} generation config hash is invalid")
        orchestrator_manifest = None
        if not args.fixture:
            orchestrator_manifest = validate_profile_orchestrator(
                args.profile_orchestrator_manifest,
                prepared_path=args.prepared,
                greedy_path=args.greedy,
                sampled_path=args.sampled,
                greedy_manifest=greedy_manifest,
                sampled_manifest=sampled_manifest,
            )
        profile_range = (float(gcfg["difficulty_min"]), float(gcfg["difficulty_max"]))
    else:
        orchestrator_manifest = None
        profile_range = (5.0, 7.0)

    difficulty_rows, completions = score_profiles(prepared, greedy_rows, sampled_rows, C)
    if greedy_manifest and completions:
        actual_greedy_cfg = {r["generation_config_hash"] for r in completions if r["mode"] == "greedy"}
        actual_sampled_cfg = {r["generation_config_hash"] for r in completions if r["mode"] == "sampled"}
        if actual_greedy_cfg != {greedy_manifest["config_hash"]}:
            raise SystemExit("greedy row config hashes do not match manifest")
        if actual_sampled_cfg != {sampled_manifest["config_hash"]}:
            raise SystemExit("sampled row config hashes do not match manifest")
    expected_profile = {
        row["row_hash"] for row in prepared
        if profile_range[0] <= float(row["difficulty"]) < profile_range[1]
    }
    actual_profile = {row["row_hash"] for row in difficulty_rows}
    if not args.fixture and actual_profile != expected_profile:
        raise SystemExit(
            f"profile input is not full-range closed: missing={len(expected_profile-actual_profile)} "
            f"extra={len(actual_profile-expected_profile)}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    completion_path = outdir / "deepmath_difficulty_completions.jsonl"
    difficulty_path = outdir / "deepmath_difficulty_rows.jsonl"
    D.write_jsonl(completion_path, completions)
    D.write_jsonl(difficulty_path, difficulty_rows)

    n = len(difficulty_rows)
    if args.fixture:
        if args.large_size is None or args.small_size is None:
            raise SystemExit("fixture mode requires --small-size and --large-size")
        small_size, large_size, status = args.small_size, args.large_size, "ready_fixture"
    elif n >= 20_000:
        small_size, large_size, status = 4_000, 20_000, "ready_4k_20k"
    elif profile_range == (5.0, 7.0):
        small_size = large_size = 0
        status = "need_expanded_4_8_profile"
    elif n >= 10_000:
        small_size, large_size, status = 2_000, 10_000, "ready_2k_10k"
    else:
        small_size = large_size = 0
        status = "insufficient_under_10k_stop_training"

    summary: dict[str, Any] = {
        "stage": "complete",
        "kind": "deepmath_difficulty_and_schedule",
        "formal": not args.fixture,
        "status": status,
        "profile_range": list(profile_range),
        "counts": {
            "profiled": n,
            "completion_rows": len(completions),
            "anchor": sum(r["difficulty_bin"] == "anchor" for r in difficulty_rows),
            "frontier": sum(r["difficulty_bin"] == "frontier" for r in difficulty_rows),
            "hard": sum(r["difficulty_bin"] == "hard" for r in difficulty_rows),
        },
        "inputs": {
            "prepared_sha256": D.sha256_file(args.prepared),
            "greedy_sha256": D.sha256_file(args.greedy),
            "sampled_sha256": D.sha256_file(args.sampled),
            "replay_pool_sha256": D.sha256_file(args.replay_pool),
            "internal_dev_sha256": D.sha256_file(args.internal_dev),
            "preparation_manifest_sha256": preparation_manifest_sha,
            "preparation_parent_manifest_sha256": preparation_parent_manifest_sha,
            "profile_orchestrator_manifest_sha256": (
                D.sha256_file(args.profile_orchestrator_manifest)
                if orchestrator_manifest is not None else None
            ),
        },
        "artifacts": {
            "difficulty_completions": {"path": completion_path.name, "sha256": D.sha256_file(completion_path)},
            "difficulty_rows": {"path": difficulty_path.name, "sha256": D.sha256_file(difficulty_path)},
        },
    }
    if large_size:
        try:
            master, master_counts = select_master(difficulty_rows, large_size)
        except ValueError as exc:
            summary["status"] = "insufficient_balanced_rows_stop_training"
            summary["stop_reason"] = str(exc)
        else:
            small, small_schedule, large_schedule, audits = build_schedules(
                master, small_size, args.dm_exposures, replay_pool,
                args.replay_exposures, args.schedule_seed,
            )
            master_path = outdir / f"deepmath_master_{large_size}.jsonl"
            small_path = outdir / f"deepmath_nested_{small_size}.jsonl"
            small_hashes = {row["row_hash"] for row in small}
            D.write_jsonl(master_path, (
                {**row, "split": "master_large", "in_nested_small": row["row_hash"] in small_hashes}
                for row in master
            ))
            D.write_jsonl(small_path, ({**row, "split": "nested_small"} for row in small))
            small_train, small_schedule_manifest = _write_training(outdir, "dm_small", small_schedule)
            large_train, large_schedule_manifest = _write_training(outdir, "dm_large", large_schedule)
            if len(small_schedule) != args.dm_exposures + args.replay_exposures:
                raise AssertionError("small schedule row count is not registered total")
            if len(large_schedule) != args.dm_exposures + args.replay_exposures:
                raise AssertionError("large schedule row count is not registered total")
            summary.update({
                "small_size": small_size,
                "large_size": large_size,
                "master_bin_counts": master_counts,
                "dm_exposures": args.dm_exposures,
                "replay_exposures": args.replay_exposures,
                "total_rows_per_arm": len(small_schedule),
                "epochs": 1,
                "schedule_seed": args.schedule_seed,
                "audits": audits,
            })
            for key, path in {
                "master": master_path, "nested_small": small_path,
                "dm_small_train": small_train, "dm_small_schedule": small_schedule_manifest,
                "dm_large_train": large_train, "dm_large_schedule": large_schedule_manifest,
            }.items():
                summary["artifacts"][key] = {"path": path.name, "sha256": D.sha256_file(path)}

    schedule_config = {
        "profile_range": summary["profile_range"],
        "small_size": summary.get("small_size"),
        "large_size": summary.get("large_size"),
        "dm_exposures": args.dm_exposures,
        "replay_exposures": args.replay_exposures,
        "schedule_seed": args.schedule_seed,
        "prompt_sha": D.INSTR_SHA,
        "student_model_revision": D.STUDENT_REVISION,
        "math_verify": "0.9.0",
        "replay_dedup_policy": REPLAY_DEDUP_POLICY,
    }
    summary["config"] = schedule_config
    summary["config_hash"] = D.config_hash(schedule_config)
    D.write_json(outdir / "deepmath_schedule_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
