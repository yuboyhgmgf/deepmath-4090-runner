#!/usr/bin/env python3
"""DeepMath-103K distillation pipeline shared helpers.

This module deliberately contains no vLLM import.  Generation and Math-Verify
scoring are separate passes so signal-based Math-Verify timeouts always run in
the scorer's main thread.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DEEPMATH_DATASET = "zwhe99/DeepMath-103K"
DEEPMATH_REVISION = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
DEEPMATH_EXPECTED_ROWS = 103_022
STUDENT_MODEL = "google/gemma-3-4b-it"
STUDENT_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
# vLLM 0.24.0's current CUDA-13 wheel requires Transformers 5.x.  Training
# remains in a separate Transformers 4.50.0 environment; never mix these two
# runtimes in one formal evidence directory.
VLLM_TRANSFORMERS_VERSION = "5.12.1"
VLLM_TORCH_VERSION = "2.11.0+cu130"
VLLM_TORCH_CUDA = "13.0"
# vLLM 0.24's FlashInfer sampler cannot initialize reliably on the Colab G4
# Blackwell image (CUDA 12.8 torch runtime versus SM12.x requiring CUDA >=12.9).
# Disable it explicitly and record the setting in every formal manifest.
VLLM_USE_FLASHINFER_SAMPLER = "0"
MATH_DATASET_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
TRAIN_MAXLEN = 2048
MATH_L45_TEST_ROWS = 2538
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

INSTR = (
    "Solve the math problem. Please reason step by step, and put your final answer "
    "within \\boxed{{}}.\n\n{q}"
)
INSTR_SHA = hashlib.sha1(INSTR.encode("utf-8")).hexdigest()[:12]
assert INSTR_SHA == "28557dc5b760", "DeepMath prompt drifted from the final MATH ruler"

MAXNEW = 4096
MAXLEN = 1024
MAX_MODEL_LEN = 5136
SAMPLED_K = 8
SAMPLED_TEMPERATURE = 0.8
SAMPLED_TOP_P = 0.95
SAMPLED_SEED = 42
R1_FINAL_SECTION_EXTRACTOR_VERSION = "deepmath-r1-final-section-v1"
_R1_FINAL_ANSWER_MARKER = re.compile(
    r"(?im)^\s*(?:\*\*)?final answer(?:\*\*)?\s*:?[ \t]*$"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_r1_final_section(text: str) -> dict[str, Any] | None:
    """Extract DeepMath's polished answer appended after the R1 final marker.

    DeepMath's bundled trajectories contain a long reasoning draft, a line
    headed ``Final Answer``, a short answer-only paragraph, and then a second
    self-contained polished solution.  Iterate markers from the end and keep
    only that polished section.  The 200-character bound prevents discarding a
    substantive paragraph when the expected short answer block is absent.
    """
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_R1_FINAL_ANSWER_MARKER.finditer(normalized))
    for match in reversed(matches):
        tail = normalized[match.end():].strip()
        parts = [part.strip() for part in re.split(r"\n[ \t]*\n+", tail) if part.strip()]
        if len(parts) < 2 or len(parts[0]) > 200:
            continue
        content = "\n\n".join(parts[1:]).strip()
        if content:
            return {
                "content": content,
                "marker_count": len(matches),
                "discarded_answer_block": parts[0],
                "marker_start": match.start(),
            }
    return None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | os.PathLike[str], value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(canonical_json(row) + "\n")
            n += 1
    os.replace(tmp, p)
    return n


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc


def deepmath_row_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Only immutable upstream fields participate in the source-row digest."""
    return {
        "question": row.get("question"),
        "final_answer": row.get("final_answer"),
        "difficulty": row.get("difficulty"),
        "topic": row.get("topic"),
        "r1_solution_1": row.get("r1_solution_1"),
        "r1_solution_2": row.get("r1_solution_2"),
        "r1_solution_3": row.get("r1_solution_3"),
    }


def deepmath_row_hash(row: dict[str, Any]) -> str:
    return sha256_obj(deepmath_row_payload(row))


_LATEX_SPACE = re.compile(r"(?:\\,|\\!|\\;|\\:|\\quad|\\qquad)")


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = _LATEX_SPACE.sub(" ", text)
    text = text.replace("$", " ")
    text = re.sub(r"\\(?:left|right)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalized_question_hash(text: str) -> str:
    return sha256_text(normalize_question(text))


def normalize_template(text: str) -> str:
    text = normalize_question(text)
    text = re.sub(r"(?<![a-z])[-+]?\d+(?:[.,]\d+)*(?:e[-+]?\d+)?", " <num> ", text)
    # Single Latin symbols in mathematical prose are treated as renameable variables.
    text = re.sub(r"(?<![a-z])[a-z](?![a-z])", " <var> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def template_hash(text: str) -> str:
    return sha256_text(normalize_template(text))


def char_ngrams(text: str, n: int = 5) -> frozenset[str]:
    compact = re.sub(r"\s+", " ", normalize_question(text))
    if not compact:
        return frozenset()
    if len(compact) < n:
        return frozenset({compact})
    return frozenset(compact[i:i + n] for i in range(len(compact) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def build_ngram_index(texts: Sequence[str], n: int = 5):
    grams = [char_ngrams(t, n=n) for t in texts]
    index: dict[str, list[int]] = defaultdict(list)
    for i, gs in enumerate(grams):
        for gram in gs:
            index[gram].append(i)
    return grams, index


def nearest_ngram_match(
    text: str,
    reference_grams: Sequence[frozenset[str]],
    inverted_index: dict[str, list[int]],
    n: int = 5,
) -> tuple[float, int | None]:
    grams = char_ngrams(text, n=n)
    candidates: set[int] = set()
    for gram in grams:
        candidates.update(inverted_index.get(gram, ()))
    best_score, best_idx = 0.0, None
    for idx in candidates:
        score = jaccard(grams, reference_grams[idx])
        if score > best_score:
            best_score, best_idx = score, idx
    return best_score, best_idx


_THINK_TAG = re.compile(r"</?think\s*>", flags=re.I)


def strip_think_tags(text: str) -> str:
    return _THINK_TAG.sub("", text or "").strip()


def boxed_spans(text: str) -> list[tuple[int, int]]:
    """Return spans for every complete ``\boxed``/``\fbox`` expression."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*", text or ""):
        start = match.start()
        brace = (text or "").find("{", match.end())
        if brace < 0:
            continue
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, i + 1))
                    break
    return spans


def unbox_nonterminal_boxes(text: str) -> tuple[str, int]:
    """Keep the terminal box and preserve earlier boxed math without the boxes.

    Overlapping/nested box spans are left unchanged so the normal hygiene check
    rejects them as ambiguous instead of attempting a lossy rewrite.
    """
    spans = boxed_spans(text)
    if len(spans) <= 1:
        return text, 0
    if any(start < previous_end for previous_end, (start, _) in zip(
        (end for _, end in spans), spans[1:]
    )):
        return text, 0
    parts: list[str] = []
    cursor = 0
    for start, end in spans[:-1]:
        chunk = text[start:end]
        brace = chunk.find("{")
        if brace < 0 or not chunk.endswith("}"):
            return text, 0
        parts.append(text[cursor:start])
        parts.append(chunk[brace + 1:-1])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), len(spans) - 1


def has_unfinished_box(text: str) -> bool:
    starts = len(re.findall(r"\\(?:boxed|fbox)\b", text or ""))
    return starts != len(boxed_spans(text or ""))


def has_substantive_after_box(text: str) -> bool:
    spans = boxed_spans(text)
    if not spans:
        return False
    tail = text[spans[-1][1]:]
    # Only sentence punctuation, TeX display closers, and whitespace are harmless.
    tail = re.sub(r"(?:\\\]|\\\)|\$)+", "", tail)
    return bool(re.sub(r"[\s.,;:!?]+", "", tail))


_TOOL_RESIDUE = re.compile(
    r"(?:```|<tool|</tool>|tool[_ -]?(?:call|output)|python\s*(?:code|output)|"
    r"sympy\.|wolfram|calculator\s*(?:output|result)|execution\s+result)",
    flags=re.I,
)


def has_tool_residue(text: str) -> bool:
    return bool(_TOOL_RESIDUE.search(text or ""))


def has_repetition_loop(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", x).strip().casefold() for x in (text or "").splitlines()]
    counts = Counter(x for x in lines if len(x) >= 24)
    if any(count >= 3 for count in counts.values()):
        return True
    words = re.findall(r"\S+", (text or "").casefold())
    if len(words) < 60:
        return False
    counts20 = Counter(tuple(words[i:i + 20]) for i in range(len(words) - 19))
    return any(count >= 3 for count in counts20.values())


_PROOF = re.compile(r"\b(?:prove|show|demonstrate|establish)\s+(?:that|why)\b", re.I)
_MULTIPLE_CHOICE = re.compile(
    r"(?:\\text\s*\{?\s*\(?[A-E]\)?\s*\}?\s*[,;]|"
    r"(?:^|\n)\s*\(?[A-E]\)\s+.*(?:\n\s*\(?[B-E]\)\s+)|"
    r"\b(?:multiple choice|choose (?:the|an) (?:correct|best)|answer choices?)\b)",
    re.I | re.S,
)
_DIAGRAM = re.compile(
    r"\b(?:diagram|figure|graph shown|pictured|illustration|shown (?:above|below)|not drawn to scale)\b",
    re.I,
)
_NON_UNIQUE = re.compile(
    r"\b(?:give|provide|construct|find)\s+(?:an|one|some)\s+(?:example|possible|such)\b|"
    r"\banswers?\s+(?:may|can)\s+vary\b",
    re.I,
)
_ANSWER_CUE = re.compile(r"\b(?:answer|solution)\s*(?:is|:|=)|\\boxed\s*\{", re.I)


def question_reject_reason(question: str, _final_answer: str) -> str | None:
    question = question or ""
    if not question.strip():
        return "empty_question"
    if _PROOF.search(question):
        return "proof_problem"
    inline_options = re.findall(r"(?:^|\s)(?:\\text\s*\{\s*)?\(?[A-E]\)(?:\s*\})?", question, re.I)
    if _MULTIPLE_CHOICE.search(question) or len(inline_options) >= 3:
        return "multiple_choice"
    if _DIAGRAM.search(question):
        return "diagram_dependent"
    if _NON_UNIQUE.search(question):
        return "non_unique_answer"
    if _ANSWER_CUE.search(question):
        return "answer_leakage_cue"
    return None


def gold_boxed_text(final_answer: str) -> str:
    answer = (final_answer or "").strip()
    if re.search(r"\\(?:boxed|fbox)\s*\{", answer):
        return answer
    return f"\\boxed{{{answer}}}"


def solution_static_reject_reason(solution: str) -> str | None:
    if not solution.strip():
        return "empty_solution"
    if has_unfinished_box(solution):
        return "truncated_or_unbalanced_box"
    if len(boxed_spans(solution)) != 1:
        return "boxed_count_not_one"
    if has_substantive_after_box(solution):
        return "text_after_boxed_answer"
    if has_tool_residue(solution):
        return "tool_residue"
    if has_repetition_loop(solution):
        return "repetition_loop"
    return None


def stable_order(rows: Sequence[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: sha256_text(
            f"{salt}\0{r.get('row_hash') or r.get('question_hash') or r.get('question') or r.get('problem')}"
        ),
    )


def token_deciles(rows: Sequence[dict[str, Any]], field: str) -> dict[str, int]:
    ordered = sorted((int(r[field]), str(r["row_hash"])) for r in rows)
    result: dict[str, int] = {}
    n = len(ordered)
    for rank, (_, row_hash) in enumerate(ordered):
        result[row_hash] = min(9, (10 * rank) // max(1, n))
    return result


def assistant_token_deciles(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return token_deciles(rows, "assistant_tokens")


def difficulty_bucket(value: float) -> str:
    return f"{math.floor(float(value) * 2) / 2:.1f}"


def stratified_take(
    rows: Sequence[dict[str, Any]],
    n: int,
    fields: Sequence[str],
    salt: str,
) -> list[dict[str, Any]]:
    """Deterministic round-robin over compound strata."""
    if n > len(rows):
        raise ValueError(f"cannot take {n} rows from {len(rows)}")
    buckets: dict[tuple[Any, ...], deque[dict[str, Any]]] = defaultdict(deque)
    for row in stable_order(rows, salt):
        buckets[tuple(row.get(field) for field in fields)].append(row)
    keys = sorted(buckets, key=lambda x: canonical_json(x))
    out: list[dict[str, Any]] = []
    while len(out) < n:
        progressed = False
        for key in keys:
            if buckets[key] and len(out) < n:
                out.append(buckets[key].popleft())
                progressed = True
        if not progressed:
            raise RuntimeError("stratified_take exhausted rows unexpectedly")
    return out


def distribution(rows: Sequence[dict[str, Any]], field: str) -> dict[str, float]:
    counts = Counter(str(r.get(field)) for r in rows)
    n = max(1, len(rows))
    return {key: value / n for key, value in sorted(counts.items())}


def max_distribution_gap(
    a: Sequence[dict[str, Any]], b: Sequence[dict[str, Any]], field: str
) -> float:
    da, db = distribution(a, field), distribution(b, field)
    return max((abs(da.get(k, 0.0) - db.get(k, 0.0)) for k in set(da) | set(db)), default=0.0)


def config_hash(config: dict[str, Any]) -> str:
    return sha256_obj(config)
