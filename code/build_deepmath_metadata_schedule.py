#!/usr/bin/env python3
"""Build the registered DeepMath 2K/10K schedules from dataset metadata only.

This builder intentionally performs no frozen-model generation.  DeepMath's
dataset difficulty is used only to balance the two nested training corpora; it
is not treated as a Gemma capability measurement or a reported outcome.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import deepmath_common as D
from score_build_deepmath import (
    REPLAY_DEDUP_POLICY,
    _proportional_nested_take,
    _write_training,
    build_schedules,
    resolve_preparation_artifact_manifest,
)


PROTOCOL_ID = "deepmath-metadata-stratified-v1"
SELECTION_RANGE = (4.0, 8.0)
SELECTION_FIELDS = (
    "topic",
    "difficulty_bucket",
    "solution_token_decile",
    "prompt_token_decile",
)
SMALL_SIZE = 2_000
LARGE_SIZE = 10_000
DM_EXPOSURES = 40_000
REPLAY_EXPOSURES = 10_000
SCHEDULE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--replay-pool", required=True)
    parser.add_argument("--internal-dev", required=True)
    parser.add_argument(
        "--preparation-manifest",
        help="defaults to preparation_manifest.json beside --prepared",
    )
    parser.add_argument(
        "--preparation-parent-manifest",
        help="required when the prepared corpus is a compression supplement",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--small-size", type=int)
    parser.add_argument("--large-size", type=int)
    parser.add_argument("--dm-exposures", type=int)
    parser.add_argument("--replay-exposures", type=int)
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid preparation manifest: {path}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"preparation manifest is not a JSON object: {path}")
    return value


def validate_preparation_inputs(
    prepared_path: str,
    replay_path: str,
    internal_dev_path: str,
    manifest_path: str,
    parent_manifest_path: str | None,
    fixture: bool,
) -> dict[str, Any]:
    """Validate the immutable preparation chain and bound input artifacts."""
    manifest_file = Path(manifest_path)
    manifest = _load_manifest(manifest_file)
    if manifest.get("stage") != "complete" or manifest.get("kind") not in {
        "deepmath_preparation",
        "deepmath_preparation_with_compression_supplement",
    }:
        raise SystemExit(
            "prepared corpus manifest is not complete: "
            f"stage={manifest.get('stage')} kind={manifest.get('kind')}"
        )
    config = manifest.get("config") or {}
    if manifest.get("config_hash") != D.config_hash(config):
        raise SystemExit("preparation config hash is invalid")

    if not fixture:
        required = {
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
        mismatches = {
            key: (config.get(key), expected)
            for key, expected in required.items()
            if config.get(key) != expected
        }
        if mismatches:
            raise SystemExit(f"prepared corpus is not the pinned formal build: {mismatches}")
        if config.get("student_model_revision") not in (None, D.STUDENT_REVISION):
            raise SystemExit("prepared corpus records an unexpected student-model revision")

    prepared_sha = D.sha256_file(prepared_path)
    if (manifest.get("artifacts") or {}).get("prepared", {}).get("sha256") != prepared_sha:
        raise SystemExit("prepared JSONL does not match preparation manifest")
    artifact_manifest, parent_sha = resolve_preparation_artifact_manifest(
        manifest, parent_manifest_path
    )
    for key, path in (("replay_pool", replay_path), ("internal_dev", internal_dev_path)):
        expected = (artifact_manifest.get("artifacts") or {}).get(key, {}).get("sha256")
        if expected != D.sha256_file(path):
            raise SystemExit(f"{key} hash does not match preparation manifest")
    return {
        "preparation_manifest_sha256": D.sha256_file(manifest_file),
        "preparation_parent_manifest_sha256": parent_sha,
        "prepared_sha256": prepared_sha,
        "replay_pool_sha256": D.sha256_file(replay_path),
        "internal_dev_sha256": D.sha256_file(internal_dev_path),
        "preparation_kind": manifest["kind"],
        "preparation_config_hash": manifest["config_hash"],
    }


def metadata_rows(
    prepared: Sequence[dict[str, Any]], internal_dev: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate every prepared row and attach metadata-only strata."""
    required = {
        "row_hash",
        "question_hash",
        "family_hash",
        "question",
        "chosen_solution",
        "topic",
        "difficulty",
        "prompt_tokens",
        "assistant_tokens",
        "train_tokens",
    }
    row_hashes: set[str] = set()
    question_hashes: set[str] = set()
    family_hashes: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, source in enumerate(prepared):
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f"prepared row {index} lacks required metadata: {missing}")
        row_hash = str(source["row_hash"])
        question_hash = str(source["question_hash"])
        family_hash = str(source["family_hash"])
        if row_hash in row_hashes:
            raise ValueError(f"duplicate prepared row_hash: {row_hash}")
        if question_hash in question_hashes:
            raise ValueError(f"duplicate prepared question_hash: {question_hash}")
        if family_hash in family_hashes:
            raise ValueError(f"duplicate prepared family_hash: {family_hash}")
        row_hashes.add(row_hash)
        question_hashes.add(question_hash)
        family_hashes.add(family_hash)
        if D.normalized_question_hash(str(source["question"])) != question_hash:
            raise ValueError(f"prepared question_hash drift at row {index}")
        if D.template_hash(str(source["question"])) != family_hash:
            raise ValueError(f"prepared family_hash drift at row {index}")
        difficulty = float(source["difficulty"])
        token_values = {
            key: int(source[key])
            for key in ("prompt_tokens", "assistant_tokens", "train_tokens")
        }
        if any(value <= 0 for value in token_values.values()):
            raise ValueError(f"non-positive token count at prepared row {index}: {token_values}")
        if token_values["train_tokens"] > D.TRAIN_MAXLEN:
            raise ValueError(f"prepared row exceeds TRAIN_MAXLEN at row {index}")
        if SELECTION_RANGE[0] <= difficulty < SELECTION_RANGE[1]:
            validated.append({
                **source,
                "difficulty": difficulty,
                "difficulty_bucket": D.difficulty_bucket(difficulty),
                "selection_basis": "dataset_metadata_only",
                "difficulty_bin": "metadata_only",
            })

    dev_questions = {str(row["question_hash"]) for row in internal_dev}
    dev_families = {D.template_hash(str(row["problem"])) for row in internal_dev}
    leaked_questions = question_hashes & dev_questions
    leaked_families = family_hashes & dev_families
    if leaked_questions or leaked_families:
        raise ValueError(
            "internal dev leaked into prepared corpus: "
            f"exact={len(leaked_questions)} family={len(leaked_families)}"
        )
    solution_deciles = D.assistant_token_deciles(validated)
    prompt_deciles = D.token_deciles(validated, "prompt_tokens")
    for row in validated:
        row["solution_token_decile"] = solution_deciles[row["row_hash"]]
        row["prompt_token_decile"] = prompt_deciles[row["row_hash"]]
    return validated, {
        "prepared_total": len(prepared),
        "eligible_metadata_range": len(validated),
        "unique_row_hashes": len(row_hashes),
        "unique_question_hashes": len(question_hashes),
        "unique_family_hashes": len(family_hashes),
    }


def select_metadata_master(rows: Sequence[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Select a stable metadata-stratified corpus without ability labels."""
    if len(rows) < target:
        raise ValueError(f"metadata range has {len(rows)} rows; need at least {target}")
    return _proportional_nested_take(
        rows, target, SELECTION_FIELDS, "metadata-master-v1"
    )


def select_metadata_nested(master: Sequence[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Allocate nested rows while keeping every registered metadata marginal balanced.

    The joint allocation is the stable baseline.  Integer rounding can still
    move a marginal by a few percent, so deterministic swaps repair only the
    difficulty bucket.  Swaps prefer identical topic and token-decile strata,
    preserving the other marginals while making the difficulty quota exact.
    """
    if target > len(master):
        raise ValueError(f"cannot take nested {target} from {len(master)}")
    selected = _proportional_nested_take(
        master,
        target,
        ("difficulty_bin",) + SELECTION_FIELDS,
        "metadata-small-v1",
    )
    target_rows = _proportional_nested_take(
        master, target, ("difficulty_bucket",), "metadata-small-difficulty-target"
    )
    target_counts = {}
    for row in target_rows:
        bucket = str(row["difficulty_bucket"])
        target_counts[bucket] = target_counts.get(bucket, 0) + 1
    selected_hashes = {str(row["row_hash"]) for row in selected}
    for _ in range(target):
        current_counts = {}
        for row in selected:
            bucket = str(row["difficulty_bucket"])
            current_counts[bucket] = current_counts.get(bucket, 0) + 1
        add_bucket = next(
            (bucket for bucket in sorted(target_counts)
             if current_counts.get(bucket, 0) < target_counts[bucket]),
            None,
        )
        if add_bucket is None:
            break
        remove_bucket = next(
            (bucket for bucket in sorted(current_counts)
             if current_counts.get(bucket, 0) > target_counts.get(bucket, 0)),
            None,
        )
        if remove_bucket is None:
            raise AssertionError("metadata difficulty quotas cannot be repaired")

        add_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        remove_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in master:
            if str(row["row_hash"]) not in selected_hashes and str(row["difficulty_bucket"]) == add_bucket:
                key = (row["topic"], row["solution_token_decile"], row["prompt_token_decile"])
                add_groups.setdefault(key, []).append(row)
        for row in selected:
            if str(row["difficulty_bucket"]) == remove_bucket:
                key = (row["topic"], row["solution_token_decile"], row["prompt_token_decile"])
                remove_groups.setdefault(key, []).append(row)
        exact_keys = sorted(set(add_groups) & set(remove_groups), key=D.canonical_json)
        if exact_keys:
            key = exact_keys[0]
            add_row = D.stable_order(add_groups[key], "metadata-swap-add")[0]
            remove_row = D.stable_order(remove_groups[key], "metadata-swap-remove")[0]
        else:
            add_by_topic: dict[Any, list[dict[str, Any]]] = {}
            remove_by_topic: dict[Any, list[dict[str, Any]]] = {}
            for key, group in add_groups.items():
                add_by_topic.setdefault(key[0], []).extend(group)
            for key, group in remove_groups.items():
                remove_by_topic.setdefault(key[0], []).extend(group)
            topics = sorted(set(add_by_topic) & set(remove_by_topic), key=str)
            if not topics:
                raise AssertionError(
                    f"metadata difficulty swap has no shared topic: add={add_bucket} remove={remove_bucket}"
                )
            topic = topics[0]
            add_row = D.stable_order(add_by_topic[topic], "metadata-swap-add-topic")[0]
            remove_row = D.stable_order(remove_by_topic[topic], "metadata-swap-remove-topic")[0]
        selected.remove(remove_row)
        selected.append(add_row)
        selected_hashes.remove(str(remove_row["row_hash"]))
        selected_hashes.add(str(add_row["row_hash"]))
    if len(selected) != target:
        raise AssertionError(f"metadata nested allocation produced {len(selected)} != {target}")
    return D.stable_order(selected, "metadata-small-final")


def _validate_replay_and_dev(
    replay_pool: Sequence[dict[str, Any]], internal_dev: Sequence[dict[str, Any]]
) -> None:
    if len(internal_dev) != 500:
        raise ValueError(f"registered internal dev must contain 500 rows, got {len(internal_dev)}")
    dev_hashes = {str(row["question_hash"]) for row in internal_dev}
    replay_hashes = {str(row["question_hash"]) for row in replay_pool}
    if dev_hashes & replay_hashes:
        raise ValueError("internal dev exact match leaked into replay pool")


def main() -> None:
    args = parse_args()
    small_size = args.small_size if args.fixture else SMALL_SIZE
    large_size = args.large_size if args.fixture else LARGE_SIZE
    dm_exposures = args.dm_exposures if args.fixture else DM_EXPOSURES
    replay_exposures = args.replay_exposures if args.fixture else REPLAY_EXPOSURES
    if None in (small_size, large_size, dm_exposures, replay_exposures):
        raise SystemExit("fixture mode requires explicit sizes and exposure counts")
    if not args.fixture and any(
        value is not None
        for value in (args.small_size, args.large_size, args.dm_exposures, args.replay_exposures)
    ):
        raise SystemExit("formal metadata schedule sizes and exposures are pinned")

    manifest_path = args.preparation_manifest or str(
        Path(args.prepared).parent / "preparation_manifest.json"
    )
    provenance = validate_preparation_inputs(
        args.prepared,
        args.replay_pool,
        args.internal_dev,
        manifest_path,
        args.preparation_parent_manifest,
        args.fixture,
    )
    prepared = D.read_jsonl(args.prepared)
    replay_pool = D.read_jsonl(args.replay_pool)
    internal_dev = D.read_jsonl(args.internal_dev)
    _validate_replay_and_dev(replay_pool, internal_dev)
    eligible, counts = metadata_rows(prepared, internal_dev)
    master = select_metadata_master(eligible, int(large_size))
    small, small_schedule, large_schedule, audits = build_schedules(
        master,
        int(small_size),
        int(dm_exposures),
        replay_pool,
        int(replay_exposures),
        SCHEDULE_SEED,
        small_selector=select_metadata_nested,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    master_path = outdir / f"deepmath_master_{large_size}.jsonl"
    small_path = outdir / f"deepmath_nested_{small_size}.jsonl"
    small_hashes = {row["row_hash"] for row in small}
    D.write_jsonl(master_path, (
        {**row, "split": "master_large", "in_nested_small": row["row_hash"] in small_hashes}
        for row in master
    ))
    D.write_jsonl(small_path, ({**row, "split": "nested_small"} for row in small))
    small_train, small_manifest = _write_training(outdir, "dm_small", small_schedule)
    large_train, large_manifest = _write_training(outdir, "dm_large", large_schedule)

    config = {
        "protocol_id": PROTOCOL_ID,
        "selection_basis": "dataset_metadata_only",
        "selection_range": list(SELECTION_RANGE),
        "selection_fields": list(SELECTION_FIELDS),
        "ability_interpretation": False,
        "small_size": int(small_size),
        "large_size": int(large_size),
        "dm_exposures": int(dm_exposures),
        "replay_exposures": int(replay_exposures),
        "schedule_seed": SCHEDULE_SEED,
        "prompt_sha": D.INSTR_SHA,
        "student_model_revision": D.STUDENT_REVISION,
        "math_verify": "0.9.0",
        "replay_dedup_policy": REPLAY_DEDUP_POLICY,
    }
    summary: dict[str, Any] = {
        "stage": "complete",
        "kind": "deepmath_metadata_schedule",
        "formal": not args.fixture,
        "status": "ready_metadata_2k_10k" if not args.fixture else "ready_metadata_fixture",
        "selection_outcome_eligible": False,
        "counts": counts,
        "small_size": int(small_size),
        "large_size": int(large_size),
        "dm_exposures": int(dm_exposures),
        "replay_exposures": int(replay_exposures),
        "total_rows_per_arm": len(small_schedule),
        "epochs": 1,
        "schedule_seed": SCHEDULE_SEED,
        "audits": audits,
        "inputs": provenance,
        "artifacts": {},
        "config": config,
        "config_hash": D.config_hash(config),
    }
    for key, path in {
        "master": master_path,
        "nested_small": small_path,
        "dm_small_train": small_train,
        "dm_small_schedule": small_manifest,
        "dm_large_train": large_train,
        "dm_large_schedule": large_manifest,
    }.items():
        summary["artifacts"][key] = {"path": path.name, "sha256": D.sha256_file(path)}
    summary_path = outdir / "deepmath_schedule_summary.json"
    D.write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
