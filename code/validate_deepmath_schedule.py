#!/usr/bin/env python3
"""Fail-closed validation of a DeepMath training schedule before GPU use."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import deepmath_common as D
from build_deepmath_metadata_schedule import (
    LARGE_SIZE,
    PROTOCOL_ID,
    SELECTION_FIELDS,
    SELECTION_RANGE,
    SMALL_SIZE,
)
from score_build_deepmath import REPLAY_DEDUP_POLICY


def validate_registered_summary(summary: dict) -> tuple[int, int]:
    """Validate the immutable formal recipe encoded by a schedule summary."""
    if (
        summary.get("stage") != "complete"
        or summary.get("kind") != "deepmath_metadata_schedule"
        or summary.get("formal") is not True
        or summary.get("status") != "ready_metadata_2k_10k"
    ):
        raise SystemExit(
            f"schedule is not a formal ready build: formal={summary.get('formal')} "
            f"kind={summary.get('kind')} status={summary.get('status')}"
        )
    if summary.get("selection_outcome_eligible") is not False:
        raise SystemExit("metadata selection must be permanently ineligible as a model outcome")
    if summary.get("total_rows_per_arm") != 50_000 or summary.get("epochs") != 1:
        raise SystemExit("registered schedule must contain 50,000 rows per arm and one Trainer epoch")
    if summary.get("dm_exposures") != 40_000 or summary.get("replay_exposures") != 10_000:
        raise SystemExit("registered exposure counts drifted from 40K DeepMath + 10K replay")
    expected_small, expected_large = SMALL_SIZE, LARGE_SIZE
    config = summary.get("config")
    if not isinstance(config, dict) or summary.get("config_hash") != D.config_hash(config):
        raise SystemExit("schedule config hash is missing or invalid")
    registered_config = {
        "protocol_id": PROTOCOL_ID,
        "selection_basis": "dataset_metadata_only",
        "selection_range": list(SELECTION_RANGE),
        "selection_fields": list(SELECTION_FIELDS),
        "ability_interpretation": False,
        "small_size": expected_small,
        "large_size": expected_large,
        "dm_exposures": 40_000,
        "replay_exposures": 10_000,
        "schedule_seed": 42,
        "prompt_sha": D.INSTR_SHA,
        "student_model_revision": D.STUDENT_REVISION,
        "math_verify": "0.9.0",
        "replay_dedup_policy": REPLAY_DEDUP_POLICY,
    }
    if config != registered_config:
        raise SystemExit(
            f"schedule config differs from the registered recipe: "
            f"observed={config!r} expected={registered_config!r}"
        )
    if (summary.get("small_size"), summary.get("large_size")) != (
        expected_small, expected_large
    ):
        raise SystemExit("schedule corpus sizes disagree with the registered status")
    counts = summary.get("counts") or {}
    if int(counts.get("eligible_metadata_range", 0)) < expected_large:
        raise SystemExit("metadata selection range does not contain 10,000 eligible rows")
    return expected_small, expected_large


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("small", "large"))
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--schedule-manifest", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_path = Path(args.train_data)
    schedule_path = Path(args.schedule_manifest)
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_registered_summary(summary)

    artifact_prefix = f"dm_{args.arm}"
    artifacts = summary.get("artifacts") or {}
    expected_train_sha = (artifacts.get(f"{artifact_prefix}_train") or {}).get("sha256")
    expected_schedule_sha = (artifacts.get(f"{artifact_prefix}_schedule") or {}).get("sha256")
    actual_train_sha = D.sha256_file(train_path)
    actual_schedule_sha = D.sha256_file(schedule_path)
    if expected_train_sha != actual_train_sha:
        raise SystemExit("train JSONL hash does not match deepmath_schedule_summary.json")
    if expected_schedule_sha != actual_schedule_sha:
        raise SystemExit("schedule manifest hash does not match deepmath_schedule_summary.json")

    source_counts: Counter[str] = Counter()
    assistant_tokens = prompt_tokens = train_tokens = 0
    rows = 0
    replay_sources: dict[str, tuple[str, str]] = {}
    replay_question_hashes: set[str] = set()
    replay_family_hashes: set[str] = set()
    with train_path.open(encoding="utf-8") as train_handle, schedule_path.open(
        encoding="utf-8"
    ) as schedule_handle:
        for schedule_idx, pair in enumerate(
            itertools.zip_longest(train_handle, schedule_handle), start=0
        ):
            train_line, schedule_line = pair
            if train_line is None or schedule_line is None:
                raise SystemExit("train JSONL and schedule manifest have different row counts")
            train_row = json.loads(train_line)
            schedule_row = json.loads(schedule_line)
            if schedule_row.get("schedule_idx") != schedule_idx:
                raise SystemExit(f"schedule_idx mismatch at row {schedule_idx}")
            if schedule_row.get("arm") != f"dm_{args.arm}":
                raise SystemExit(f"wrong arm at row {schedule_idx}: {schedule_row.get('arm')}")
            if train_row != {
                "question": schedule_row.get("question"),
                "answer": schedule_row.get("answer"),
            }:
                raise SystemExit(f"training projection differs from manifest at row {schedule_idx}")
            source_counts[str(schedule_row.get("source"))] += 1
            if schedule_row.get("source") == "math_replay":
                source_row_hash = str(schedule_row.get("source_row_hash") or "")
                source_question_hash = str(schedule_row.get("source_question_hash") or "")
                source_family_hash = str(schedule_row.get("source_family_hash") or "")
                if not source_row_hash or not source_question_hash or not source_family_hash:
                    raise SystemExit(
                        f"replay provenance identity is missing at row {schedule_idx}"
                    )
                identity = (source_question_hash, source_family_hash)
                previous = replay_sources.setdefault(source_row_hash, identity)
                if previous != identity:
                    raise SystemExit(
                        f"replay source identity drift at row {schedule_idx}: {source_row_hash}"
                    )
                replay_question_hashes.add(source_question_hash)
                replay_family_hashes.add(source_family_hash)
            assistant_tokens += int(schedule_row["assistant_tokens"])
            prompt_tokens += int(schedule_row["prompt_tokens"])
            train_tokens += int(schedule_row["train_tokens"])
            rows += 1

    expected_sources = {f"deepmath_{args.arm}": 40_000, "math_replay": 10_000}
    if rows != 50_000 or dict(source_counts) != expected_sources:
        raise SystemExit(f"schedule composition mismatch: rows={rows} sources={dict(source_counts)}")
    audits = summary.get("audits") or {}
    replay_audit = {
        key: audits.get(key)
        for key in (
            "replay_dedup_policy",
            "replay_input_rows",
            "replay_kept_rows",
            "replay_dropped_exact",
            "replay_dropped_family",
            "replay_unique_question_hashes",
            "replay_unique_family_hashes",
        )
    }
    if replay_audit["replay_dedup_policy"] != REPLAY_DEDUP_POLICY:
        raise SystemExit("replay deduplication policy is missing or drifted")
    numeric_replay_audit = {
        key: replay_audit[key]
        for key in replay_audit
        if key != "replay_dedup_policy"
    }
    if any(not isinstance(value, int) or value < 0 for value in numeric_replay_audit.values()):
        raise SystemExit(f"replay deduplication audit is invalid: {replay_audit}")
    if (
        replay_audit["replay_kept_rows"]
        + replay_audit["replay_dropped_exact"]
        + replay_audit["replay_dropped_family"]
        != replay_audit["replay_input_rows"]
    ):
        raise SystemExit("replay deduplication audit count closure failed")
    observed_replay_unique = {
        "replay_kept_rows": len(replay_sources),
        "replay_unique_question_hashes": len(replay_question_hashes),
        "replay_unique_family_hashes": len(replay_family_hashes),
    }
    for key, observed_value in observed_replay_unique.items():
        if replay_audit.get(key) != observed_value:
            raise SystemExit(
                f"{key} differs from schedule: recorded={replay_audit.get(key)} "
                f"observed={observed_value}"
            )
    observed = {
        "assistant_tokens": assistant_tokens,
        "prompt_tokens": prompt_tokens,
        "train_tokens": train_tokens,
    }
    for metric, value in observed.items():
        recorded = audits.get(f"{args.arm}_{metric}")
        if recorded != value:
            raise SystemExit(
                f"{metric} total differs from summary: recorded={recorded} observed={value}"
            )
    for gap_name in (
        "assistant_token_gap_fraction",
        "prompt_token_gap_fraction",
        "train_token_gap_fraction",
    ):
        if float(audits.get(gap_name, 1.0)) > 0.02 + 1e-12:
            raise SystemExit(f"{gap_name} exceeds the registered 2% ceiling")

    result = {
        "stage": "complete",
        "kind": "deepmath_schedule_validation",
        "arm": args.arm,
        "status": summary["status"],
        "rows": rows,
        "source_counts": dict(source_counts),
        "token_totals": observed,
        "train_data_sha256": actual_train_sha,
        "schedule_manifest_sha256": actual_schedule_sha,
        "schedule_summary_sha256": D.sha256_file(summary_path),
        "schedule_config_hash": summary.get("config_hash"),
        "replay_unique_sources": observed_replay_unique,
    }
    D.write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
