#!/usr/bin/env python3
"""Apply the registered seed-42 internal-dev gate without touching MATH test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import deepmath_common as D
from analyze_deepmath_confirm import (
    by_idx,
    exact_mcnemar,
    read_records,
    record_metrics,
    validate_final_ruler_records,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--internal-dev", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--large-seed42", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = D.read_jsonl(args.internal_dev)
    if len(dev) != 500:
        raise SystemExit(f"registered internal dev must contain 500 rows, got {len(dev)}")
    dev_indices = {row["idx"] for row in dev}
    baseline_rows = read_records(args.baseline)
    large_rows = read_records(args.large_seed42)
    provenance = {
        "baseline": validate_final_ruler_records(baseline_rows, "dev baseline"),
        "large_seed42": validate_final_ruler_records(large_rows, "dev large seed42"),
    }
    eval_hashes = {item["eval_set_sha256"] for item in provenance.values()}
    if len(eval_hashes) != 1:
        raise SystemExit(f"dev gate inputs used different eval sets: {sorted(eval_hashes)}")
    baseline = by_idx(baseline_rows, "dev baseline")
    large = by_idx(large_rows, "dev large seed42")
    if set(baseline) != dev_indices or set(large) != dev_indices:
        raise SystemExit("baseline/large idx sets must exactly equal the frozen internal dev")
    dev_by_idx = {row["idx"]: row for row in dev}
    for label, records in (("baseline", baseline), ("large", large)):
        mismatched = [
            idx for idx in dev_indices
            if D.normalized_question_hash(dev_by_idx[idx]["problem"])
            != D.normalized_question_hash(str(records[idx].get("problem") or records[idx].get("question") or ""))
        ]
        if mismatched:
            raise SystemExit(f"{label} questions do not match frozen internal dev: {mismatched[:10]}")
    indices = sorted(dev_indices, key=str)
    b = [int(baseline[idx]["fallback_correct"]) for idx in indices]
    l = [int(large[idx]["fallback_correct"]) for idx in indices]
    base_acc, large_acc = (sum(values) / len(values) for values in (b, l))
    passed = large_acc - base_acc >= 0.01
    result = {
        "stage": "complete",
        "kind": "deepmath_internal_dev_gate",
        "analysis_stage": "large_only",
        "seed": 42,
        "n": len(indices),
        "baseline": {"fallback_acc": base_acc, "metrics": record_metrics([baseline[i] for i in indices])},
        "large": {
            "fallback_acc": large_acc,
            "delta_vs_baseline": large_acc - base_acc,
            "mcnemar_vs_baseline": exact_mcnemar(b, l),
            "metrics": record_metrics([large[i] for i in indices]),
        },
        "gate_passed": passed,
        "gate_rule": "large-base >=1pp on frozen 500-question internal dev",
        "formal_test_authorized": passed,
        "small_ablation_authorized": False,
        "ruler_provenance": provenance,
        "input_sha256": {
            "internal_dev": D.sha256_file(args.internal_dev),
            "baseline": D.sha256_file(args.baseline),
            "large_seed42": D.sha256_file(args.large_seed42),
        },
    }
    D.write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
