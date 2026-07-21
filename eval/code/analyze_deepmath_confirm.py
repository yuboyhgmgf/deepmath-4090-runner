#!/usr/bin/env python3
"""Registered staged formal analysis for DeepMath versus frozen baseline."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import deepmath_common as D


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, help="formal MATH L4-L5 baseline eval records")
    p.add_argument("--small", action="append", default=[], metavar="SEED=PATH")
    p.add_argument("--large", action="append", default=[], metavar="SEED=PATH")
    p.add_argument(
        "--large-confirm",
        help="required before the small-arm diversity ablation; must be a passing large-only result",
    )
    p.add_argument("--bootstrap-reps", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=20260714)
    p.add_argument("--out", required=True)
    p.add_argument("--fixture", action="store_true")
    return p.parse_args()


def read_records(path: str) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").lstrip()
    if not text:
        raise ValueError(f"empty records file: {path}")
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"expected list in {path}")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_seed_paths(values: Sequence[str]) -> dict[int, str]:
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected SEED=PATH, got {value!r}")
        raw_seed, path = value.split("=", 1)
        seed = int(raw_seed)
        if seed in out:
            raise ValueError(f"duplicate seed {seed}")
        out[seed] = path
    return out


def by_idx(rows: Sequence[dict[str, Any]], label: str) -> dict[Any, dict[str, Any]]:
    out = {}
    required = {
        "idx", "strict_correct", "fallback_correct", "generated_tokens_approx",
        "hit_maxnew_approx", "text",
    }
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"missing required eval-record fields in {label}: {missing}")
        if not (row.get("problem") or row.get("question")):
            raise ValueError(f"missing problem/question text in {label}: idx={row.get('idx')}")
        idx = row["idx"]
        if idx in out:
            raise ValueError(f"duplicate idx in {label}: {idx}")
        if int(row["fallback_correct"]) not in (0, 1):
            raise ValueError(f"non-binary fallback_correct in {label}: idx={idx}")
        if int(row["strict_correct"]) not in (0, 1):
            raise ValueError(f"non-binary strict_correct in {label}: idx={idx}")
        out[idx] = row
    return out


def validate_final_ruler_records(
    rows: Sequence[dict[str, Any]], label: str
) -> dict[str, Any]:
    required = {
        "ruler_version", "instr_sha", "vllm_version", "transformers_version",
        "torch_version", "torch_cuda",
        "vllm_use_flashinfer_sampler", "cuda_runtime_library", "batch_invariant",
        "maxnew", "maxlen", "max_model_len", "eval_set_sha256",
        "ruler_config_hash", "generation_manifest_sha256",
    }
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"missing final-ruler provenance in {label}: {missing}")
    expected = {
        "ruler_version": "math-mathverify-terminal-v2",
        "instr_sha": D.INSTR_SHA,
        "vllm_version": "0.24.0",
        "transformers_version": D.VLLM_TRANSFORMERS_VERSION,
        "torch_version": D.VLLM_TORCH_VERSION,
        "torch_cuda": D.VLLM_TORCH_CUDA,
        "vllm_use_flashinfer_sampler": D.VLLM_USE_FLASHINFER_SAMPLER,
        "batch_invariant": True,
        "maxnew": D.MAXNEW,
        "maxlen": D.MAXLEN,
        "max_model_len": D.MAX_MODEL_LEN,
    }
    mismatches = {
        key: sorted({str(row.get(key)) for row in rows})
        for key, value in expected.items()
        if any(row.get(key) != value for row in rows)
    }
    if mismatches:
        raise ValueError(f"{label} is not the registered final ruler: {mismatches}")
    eval_hashes = {str(row["eval_set_sha256"]) for row in rows}
    config_hashes = {str(row["ruler_config_hash"]) for row in rows}
    manifest_hashes = {str(row["generation_manifest_sha256"]) for row in rows}
    if len(eval_hashes) != 1 or len(config_hashes) != 1 or len(manifest_hashes) != 1:
        raise ValueError(
            f"non-uniform ruler provenance in {label}: eval={len(eval_hashes)} "
            f"config={len(config_hashes)} manifest={len(manifest_hashes)}"
        )
    if any(not value or value == "None" for value in eval_hashes | config_hashes | manifest_hashes):
        raise ValueError(f"null ruler provenance in {label}")
    if any(not str(row.get("cuda_runtime_library") or "").strip() for row in rows):
        raise ValueError(f"null CUDA runtime provenance in {label}")
    return {
        "eval_set_sha256": next(iter(eval_hashes)),
        "ruler_config_hash": next(iter(config_hashes)),
        "generation_manifest_sha256": next(iter(manifest_hashes)),
    }


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    top = max(values)
    return top + math.log(sum(math.exp(value - top) for value in values))


def exact_mcnemar(base: Sequence[int], treatment: Sequence[int]) -> dict[str, Any]:
    improved = sum(b == 0 and t == 1 for b, t in zip(base, treatment))
    regressed = sum(b == 1 and t == 0 for b, t in zip(base, treatment))
    discordant = improved + regressed
    if discordant == 0:
        p = 1.0
    else:
        tail = min(improved, regressed)
        logs = [
            math.lgamma(discordant + 1) - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1) - discordant * math.log(2)
            for k in range(tail + 1)
        ]
        p = min(1.0, 2.0 * math.exp(_logsumexp(logs)))
    return {
        "improved": improved,
        "regressed": regressed,
        "discordant": discordant,
        "exact_two_sided_p": p,
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty quantile input")
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def clustered_question_bootstrap(
    per_question_seed_deltas: Sequence[Sequence[float]], reps: int, seed: int
) -> dict[str, float]:
    if not per_question_seed_deltas:
        raise ValueError("empty bootstrap data")
    cluster_values = [sum(values) / len(values) for values in per_question_seed_deltas]
    rng = random.Random(seed)
    n = len(cluster_values)
    draws = []
    for _ in range(reps):
        draws.append(sum(cluster_values[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    return {
        "estimate": sum(cluster_values) / n,
        "ci95_low": _quantile(draws, 0.025),
        "ci95_high": _quantile(draws, 0.975),
        "reps": reps,
        "seed": seed,
        "cluster": "question_idx",
        "seed_aggregation_within_cluster": "mean",
    }


def compare_arm(
    baseline: dict[Any, dict[str, Any]], runs: dict[int, dict[Any, dict[str, Any]]], label: str
):
    base_indices = set(baseline)
    per_seed = {}
    for seed, run in sorted(runs.items()):
        if set(run) != base_indices:
            raise ValueError(
                f"{label} seed {seed} idx mismatch: missing={len(base_indices-set(run))} "
                f"extra={len(set(run)-base_indices)}"
            )
        mismatched_questions = [
            idx for idx in base_indices
            if D.normalized_question_hash(str(baseline[idx].get("problem") or baseline[idx].get("question") or ""))
            != D.normalized_question_hash(str(run[idx].get("problem") or run[idx].get("question") or ""))
        ]
        if mismatched_questions:
            raise ValueError(f"{label} seed {seed} question mismatch at idx={mismatched_questions[:10]}")
        base_values = [int(baseline[idx]["fallback_correct"]) for idx in sorted(base_indices, key=str)]
        run_values = [int(run[idx]["fallback_correct"]) for idx in sorted(base_indices, key=str)]
        delta = sum(t - b for b, t in zip(base_values, run_values)) / len(base_values)
        per_seed[str(seed)] = {
            "baseline_fallback": sum(base_values) / len(base_values),
            "arm_fallback": sum(run_values) / len(run_values),
            "delta_fallback": delta,
            "mcnemar": exact_mcnemar(base_values, run_values),
            "metrics": record_metrics([run[idx] for idx in sorted(base_indices, key=str)]),
        }
    return per_seed


def record_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    n = len(rows)
    if not n:
        raise ValueError("cannot summarize empty records")
    strict = [int(row["strict_correct"]) for row in rows]
    fallback = [int(row["fallback_correct"]) for row in rows]
    tokens = [int(row["generated_tokens_approx"]) for row in rows]
    hit = [int(row.get("hit_maxnew_approx", 0)) for row in rows]
    loops = [int(D.has_repetition_loop(str(row.get("text") or ""))) for row in rows]
    if any(s and not f for s, f in zip(strict, fallback)):
        raise ValueError("strict=>fallback contract violation in formal eval records")
    return {
        "n": n,
        "strict_acc": sum(strict) / n,
        "fallback_acc": sum(fallback) / n,
        "format_gap": (sum(fallback) - sum(strict)) / n,
        "mean_generated_tokens": sum(tokens) / n,
        "median_generated_tokens": float(statistics.median(tokens)),
        "hit_maxnew_rate": sum(hit) / n,
        "repetition_loop_rate": sum(loops) / n,
    }


def validate_large_confirmation(
    path: str,
    baseline_path: str,
    large_paths: dict[int, str],
) -> dict[str, Any]:
    """Validate the hash-bound passing large-only artifact that unlocks small."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid large confirmation artifact: {path}") from error
    if (
        value.get("stage") != "complete"
        or value.get("kind") != "deepmath_formal_confirmation"
        or value.get("formal") is not True
        or value.get("analysis_stage") != "large_only"
    ):
        raise SystemExit("small ablation requires a complete formal large-only confirmation")
    if (value.get("claims") or {}).get("deepmath_distillation_improves_capability") is not True:
        raise SystemExit("large-only confirmation did not pass the registered distillation rule")
    if value.get("baseline_sha256") != D.sha256_file(baseline_path):
        raise SystemExit("large-only confirmation uses a different frozen baseline")
    expected_large = {
        str(seed): D.sha256_file(path_value)
        for seed, path_value in sorted(large_paths.items())
    }
    observed_large = ((value.get("input_sha256") or {}).get("large") or {})
    if observed_large != expected_large:
        raise SystemExit("large-only confirmation uses different large-arm inputs")
    if value.get("seeds") != sorted(large_paths):
        raise SystemExit("large-only confirmation seed set differs from the ablation")
    return value


def main() -> None:
    args = parse_args()
    small_paths, large_paths = parse_seed_paths(args.small), parse_seed_paths(args.large)
    if not large_paths:
        raise SystemExit("at least one large-arm seed is required")
    if small_paths and set(small_paths) != set(large_paths):
        raise SystemExit("small and large arms must have identical seeds")
    if not args.fixture and set(large_paths) != {42, 43, 44}:
        raise SystemExit("formal confirmation requires seeds 42/43/44")
    if not args.fixture and (args.bootstrap_reps, args.bootstrap_seed) != (10_000, 20260714):
        raise SystemExit("formal bootstrap recipe is pinned to 10,000 reps and seed 20260714")
    if not args.fixture and small_paths and not args.large_confirm:
        raise SystemExit("formal small ablation requires --large-confirm")
    if args.large_confirm and not small_paths:
        raise SystemExit("--large-confirm is only valid when the small arm is present")
    if args.bootstrap_reps < 1000:
        raise SystemExit("bootstrap requires at least 1000 resamples")

    baseline_rows = read_records(args.baseline)
    baseline_provenance = None
    if not args.fixture:
        baseline_provenance = validate_final_ruler_records(baseline_rows, "baseline")
    baseline = by_idx(baseline_rows, "baseline")
    if not args.fixture:
        levels = {str(row.get("level")) for row in baseline_rows}
        missing_problem = sum(not (row.get("problem") or row.get("question")) for row in baseline_rows)
        if (not levels.issubset({"Level 4", "Level 5"})
                or len(baseline) != D.MATH_L45_TEST_ROWS or missing_problem):
            raise SystemExit(f"baseline is not the full MATH L4-L5 set: n={len(baseline)} levels={levels}")
    large_rows = {seed: read_records(path) for seed, path in large_paths.items()}
    small_rows = {seed: read_records(path) for seed, path in small_paths.items()}
    if not args.fixture:
        all_provenance = [baseline_provenance]
        all_provenance.extend(
            validate_final_ruler_records(rows, f"large seed {seed}")
            for seed, rows in sorted(large_rows.items())
        )
        all_provenance.extend(
            validate_final_ruler_records(rows, f"small seed {seed}")
            for seed, rows in sorted(small_rows.items())
        )
        eval_hashes = {item["eval_set_sha256"] for item in all_provenance if item is not None}
        if len(eval_hashes) != 1:
            raise SystemExit(f"formal runs used different eval sets: {sorted(eval_hashes)}")
    large_runs = {seed: by_idx(rows, f"large seed {seed}") for seed, rows in large_rows.items()}
    small_runs = {seed: by_idx(rows, f"small seed {seed}") for seed, rows in small_rows.items()}

    large_per_seed = compare_arm(baseline, large_runs, "large")
    indices = sorted(baseline, key=str)
    seeds = sorted(large_runs)
    large_vs_base_clusters = []
    for idx in indices:
        base = int(baseline[idx]["fallback_correct"])
        large_vs_base_clusters.append([int(large_runs[s][idx]["fallback_correct"]) - base for s in seeds])
    large_boot = clustered_question_bootstrap(
        large_vs_base_clusters, args.bootstrap_reps, args.bootstrap_seed
    )
    large_deltas = [large_per_seed[str(seed)]["delta_fallback"] for seed in seeds]
    distillation_success = (
        all(delta > 0 for delta in large_deltas)
        and sum(large_deltas) / len(large_deltas) >= 0.02
        and large_boot["ci95_low"] > 0
    )
    analysis_stage = "diversity_ablation" if small_runs else "large_only"
    small_result = None
    diversity_result = None
    diversity_success = None
    large_confirm_sha = None
    if small_runs:
        if not args.fixture:
            validate_large_confirmation(args.large_confirm, args.baseline, large_paths)
            large_confirm_sha = D.sha256_file(args.large_confirm)
        small_per_seed = compare_arm(baseline, small_runs, "small")
        small_vs_base_clusters = []
        large_vs_small_clusters = []
        for idx in indices:
            base = int(baseline[idx]["fallback_correct"])
            small_vs_base_clusters.append([
                int(small_runs[seed][idx]["fallback_correct"]) - base for seed in seeds
            ])
            large_vs_small_clusters.append([
                int(large_runs[seed][idx]["fallback_correct"])
                - int(small_runs[seed][idx]["fallback_correct"])
                for seed in seeds
            ])
        small_boot = clustered_question_bootstrap(
            small_vs_base_clusters, args.bootstrap_reps, args.bootstrap_seed + 1
        )
        diversity_boot = clustered_question_bootstrap(
            large_vs_small_clusters, args.bootstrap_reps, args.bootstrap_seed + 2
        )
        small_deltas = [small_per_seed[str(seed)]["delta_fallback"] for seed in seeds]
        large_vs_small_per_seed = {}
        for seed in seeds:
            small_values = [int(small_runs[seed][idx]["fallback_correct"]) for idx in indices]
            large_values = [int(large_runs[seed][idx]["fallback_correct"]) for idx in indices]
            large_vs_small_per_seed[str(seed)] = {
                "delta_fallback": sum(l - s for s, l in zip(small_values, large_values)) / len(indices),
                "mcnemar": exact_mcnemar(small_values, large_values),
            }
        diversity_success = (
            distillation_success
            and diversity_boot["estimate"] >= 0.02
            and diversity_boot["ci95_low"] > 0
        )
        small_result = {
            "per_seed": small_per_seed,
            "mean_delta_fallback": sum(small_deltas) / len(small_deltas),
            "clustered_bootstrap": small_boot,
        }
        diversity_result = {
            "per_seed": large_vs_small_per_seed,
            "clustered_bootstrap": diversity_boot,
        }
    result = {
        "stage": "complete",
        "kind": "deepmath_formal_confirmation",
        "formal": not args.fixture,
        "analysis_stage": analysis_stage,
        "primary_metric": "MATH L4-L5 fallback",
        "n_questions": len(indices),
        "seeds": seeds,
        "baseline_sha256": D.sha256_file(args.baseline),
        "baseline_provenance": baseline_provenance,
        "baseline_metrics": record_metrics([baseline[idx] for idx in indices]),
        "small": small_result,
        "large": {
            "per_seed": large_per_seed,
            "mean_delta_fallback": sum(large_deltas) / len(large_deltas),
            "clustered_bootstrap": large_boot,
        },
        "large_minus_small": diversity_result,
        "claims": {
            "deepmath_distillation_improves_capability": distillation_success,
            "unique_question_scaling_improves_by_at_least_2pp": diversity_success,
            "rule": {
                "distillation": "all three seed deltas >0, mean delta >=2pp, clustered CI low >0",
                "diversity": (
                    "not evaluated until a passing hash-bound large-only confirmation unlocks small"
                    if diversity_success is None
                    else "distillation passes and large-small mean >=2pp with clustered CI low >0"
                ),
            },
        },
        "input_sha256": {
            "small": {str(seed): D.sha256_file(path) for seed, path in sorted(small_paths.items())},
            "large": {str(seed): D.sha256_file(path) for seed, path in sorted(large_paths.items())},
            "large_confirmation": large_confirm_sha,
        },
    }
    D.write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
