#!/usr/bin/env python3
"""
MATH (Hendrycks, Level 1-3) ruler + loader + resumable eval.

SEPARATE from the GSM8K ruler (kd_sft_common.py) ON PURPOSE: MATH answers are non-numeric
(fractions / surds / intervals / sets), so scoring uses Math-Verify SYMBOLIC EQUIVALENCE,
NOT the GSM8K numeric regex. Importing this does NOT touch the GSM8K pipeline.

Source: EleutherAI/hendrycks_math (original hendrycks/competition_math was DMCA-disabled).
Deps (install BEFORE import): math-verify[antlr4_13_2]==0.9.0  (pip name has a hyphen,
import name has an underscore: `import math_verify`).
"""
import os, re, json, time, hashlib, pathlib

import torch
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import HfApi, hf_hub_download, create_repo
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig

MODEL_SHORT = {
    "google/gemma-3-1b-it": "g1b",
    "google/gemma-3-4b-it": "g4b",
    "google/gemma-3-12b-it": "g12b",
    "google/gemma-3-27b-it": "g27b",
}
def model_short(model_id):
    return MODEL_SHORT.get(model_id, re.sub(r"[^a-z0-9]+", "-", model_id.lower()).strip("-"))

# Force a \boxed{} final answer — this exact phrasing is the MATH-500/Minerva standard and is
# what Math-Verify is built to parse. Math-Verify extracts NOTHING from undelimited LaTeX.
INSTR = (
    "Solve the math problem. Please reason step by step, and put your final answer "
    # {{}} escapes to a literal \boxed{} after .format(q=...); a bare {} would be parsed
    # as positional field 0 and raise "Replacement index 0 out of range" (no positional args).
    "within \\boxed{{}}.\n\n{q}"
)
RULER_VERSION = "math-mathverify-terminal-v2"
FALLBACK_CONTRACT = "last-complete-box-else-explicit-terminal-answer-v2"
INSTR_SHA = hashlib.sha1(INSTR.encode("utf-8")).hexdigest()[:12]

MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
def _parse_level_nums(raw):
    raw = (raw or "1,2,3").replace("Level", "").replace("level", "")
    nums = []
    for piece in re.split(r"[,;]+", raw):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            a, b = [int(x.strip()) for x in piece.split("-", 1)]
            lo, hi = min(a, b), max(a, b)
            nums.extend(range(lo, hi + 1))
        else:
            nums.append(int(piece))
    nums = sorted(dict.fromkeys(nums))
    if not nums or any(n < 1 or n > 5 for n in nums):
        raise ValueError(f"MATH_LEVELS must be levels 1-5, got {raw!r}")
    return tuple(nums)


def _level_tag(nums):
    if len(nums) > 1 and list(nums) == list(range(nums[0], nums[-1] + 1)):
        return f"l{nums[0]}{nums[-1]}"
    return "l" + "".join(str(n) for n in nums)


MATH_LEVEL_NUMS = _parse_level_nums(os.environ.get("MATH_LEVELS", "1,2,3"))
MATH_LEVELS = tuple(f"Level {n}" for n in MATH_LEVEL_NUMS)  # 'level' is a STRING, not int
MATH_LEVEL_TAG = _level_tag(MATH_LEVEL_NUMS)
MATH_LEVEL_LABEL = ",".join(MATH_LEVELS)
# Immutable dataset commit used by the final research line.  An explicit
# environment override remains available for historical replay, and is recorded
# by callers rather than silently following the dataset's moving default branch.
MATH_REVISION = os.environ.get("MATH_REVISION") or "21a5633873b6a120296cce3e2df9d5550074f4a3"
# Shuffle the selected levels with a FIXED seed so a small N_TEST is cross-subject representative, not
# all-algebra (the 7 configs concatenate in order). Full-set accuracy is order-invariant; the fixed
# seed keeps baseline/finetuned idx pairing reproducible. Default ON; MATH_SHUFFLE=0 = dataset order.
MATH_SHUFFLE = os.environ.get("MATH_SHUFFLE", "1").strip().lower() not in {"0", "false", "no"}
MATH_SHUFFLE_SEED = int(os.environ.get("MATH_SHUFFLE_SEED", "42"))


def read_token():
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    for p in [pathlib.Path.home() / ".hf_token", pathlib.Path("/root/.hf_token")]:
        try:
            if p.exists():
                t = p.read_text().strip()
                if t:
                    return t
        except Exception:
            pass
    return None


# ---------------- \boxed{} extraction (balanced-brace; for LOGGING the raw substring only) ----------------
def last_boxed_only_string(s):
    # Iterate \boxed/\fbox occurrences from last to first; skip any that are incomplete (no '{' or
    # unbalanced/truncated) so a truncated trailing "\boxed" doesn't hide an earlier COMPLETE "\boxed{3}".
    for m in reversed(list(re.finditer(r"\\(?:boxed|fbox)", s))):
        idx = m.start()
        i = s.find("{", idx)
        if i < 0:
            continue
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[idx:j + 1]
        # unbalanced (truncated mid-brace) -> try the previous occurrence
    return None


def boxed_inner(s):
    b = last_boxed_only_string(s)
    if b is None:
        return ""
    m = re.match(r"\\(?:boxed|fbox)\{(.*)\}$", b, flags=re.S)
    return (m.group(1).strip() if m else b)


# ---------------- Math-Verify scoring (THE RULER) ----------------
def _parse_math(text, extraction_config):
    try:
        parsing_timeout = None if os.environ.get("MATH_VERIFY_DISABLE_TIMEOUT") == "1" else 5
        return parse(text, extraction_config=extraction_config, parsing_timeout=parsing_timeout)
    except ValueError as e:
        # Math-Verify uses signal-based timeouts. Running it in a worker thread is an
        # instrument failure, not a wrong answer; never silently turn that into zero.
        if "signal" in str(e).lower() or "thread" in str(e).lower():
            raise RuntimeError("Math-Verify must run in the main thread") from e
        return None
    except Exception:
        # Arbitrary model text can be unparsable. That is a normal parse failure.
        return None


def parse_gold(solution):
    return parse_strict(solution)

def parse_strict(text):
    """Parse only the last complete boxed/fbox answer.

    Passing the whole chain-of-thought to Math-Verify lets an unboxed intermediate
    expression win extraction. Strict is defined by the marker, so isolate that
    marker before semantic parsing.
    """
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    # Math-Verify intentionally extracts nothing from undelimited LaTeX.
    return _parse_math(f"${boxed}$", [LatexExtractionConfig(boxed_match_priority=0)])


_FINAL_CUE = re.compile(
    r"(?:final\s+answer(?:\s+is)?|answer(?:\s+is)?|therefore|thus|hence)\s*[:=,-]?\s*",
    flags=re.I,
)


def terminal_answer_region(text):
    """Return (small terminal region, source), never the whole reasoning trace.

    Without an explicit cue, accept only a standalone final math line. This is
    deliberately conservative: an ambiguous output is wrong under fallback rather
    than being credited for repeating a number from the problem or an intermediate
    step.
    """
    s = (text or "").strip()
    if not s:
        return "", "none"
    cues = []
    for m in _FINAL_CUE.finditer(s):
        before = s[max(0, m.start() - 16):m.start()].lower()
        if re.search(r"(?:no|without)\s*$", before):
            continue
        cues.append(m)
    if cues:
        tail = s[cues[-1].end():].strip()
        line = next((x.strip() for x in tail.splitlines() if x.strip()), "")
        looks_math = bool(re.search(r"\d|\\|\$|[=+*/^_{}\[\]()]", line)) or \
            (len(line) <= 5 and " " not in line)
        if line and looks_math:
            return line, "explicit_cue"
        return "", "none"

    lines = [x.strip() for x in s.splitlines() if x.strip()]
    if not lines:
        return "", "none"
    last = lines[-1]
    delimited = (
        (last.startswith("$") and last.rstrip(". ").endswith("$"))
        or (last.startswith(r"\(") and last.rstrip(". ").endswith(r"\)"))
        or (last.startswith(r"\[") and last.rstrip(". ").endswith(r"\]"))
    )
    # Permit a short bare expression such as "x=2" or "1/2", but reject prose.
    no_commands = re.sub(r"\\[A-Za-z]+", "", last)
    bare_math = len(last) <= 160 and not re.search(r"[A-Za-z]{2,}", no_commands)
    if delimited or bare_math:
        return last, "standalone_last_line"
    return "", "none"

def parse_fallback(text):
    # A strict answer must also be a fallback answer; using the identical isolated
    # candidate guarantees strict_correct <= fallback_correct by construction.
    if last_boxed_only_string(text) is not None:
        return parse_strict(text)
    region, _ = terminal_answer_region(text)
    if not region:
        return None
    return _parse_math(region, [LatexExtractionConfig(), ExprExtractionConfig()])


def fallback_source(text):
    if last_boxed_only_string(text) is not None:
        return "boxed"
    return terminal_answer_region(text)[1]

def is_correct(gold_parsed, pred_parsed):
    """Math-Verify symbolic equivalence. GOLD MUST be first (verify is asymmetric for
    intervals/inequalities/sets). Returns 0 on any empty/failed parse."""
    if not gold_parsed or not pred_parsed:
        return 0
    try:
        timeout_seconds = None if os.environ.get("MATH_VERIFY_DISABLE_TIMEOUT") == "1" else 5
        return int(bool(verify(gold_parsed, pred_parsed, timeout_seconds=timeout_seconds)))
    except ValueError as e:
        if "signal" in str(e).lower() or "thread" in str(e).lower():
            raise RuntimeError("Math-Verify must run in the main thread") from e
        raise RuntimeError(f"Math-Verify verification failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Math-Verify verification failed: {e}") from e


def metrics(records):
    n = len(records)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "strict_acc": sum(r["strict_correct"] for r in records) / n,
        "fallback_acc": sum(r["fallback_correct"] for r in records) / n,
        "parse_fail_rate": sum(r["parse_fail"] for r in records) / n,
        "hit_maxnew_rate_approx": sum(r["hit_maxnew_approx"] for r in records) / n,
        "mean_gen_tokens": sum(r["generated_tokens_approx"] for r in records) / n,
    }


# ---------------- dataset ----------------
def load_math(split, levels=MATH_LEVELS):
    """Concatenate the 7 subject configs (there is NO combined config), filter configured levels, and
    (default) shuffle with a fixed seed so a small N_TEST is cross-subject representative rather
    than all-algebra. The level filter drops the 2 anomalous 'Level ?' rows in geometry/train."""
    ds = concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", c, split=split, revision=MATH_REVISION)
        for c in MATH_CONFIGS
    ])
    ds = ds.filter(lambda x: x["level"] in levels)
    if MATH_SHUFFLE:
        ds = ds.shuffle(seed=MATH_SHUFFLE_SEED)
    return ds


# ---------------- SELFTEST: validate Math-Verify works BEFORE trusting any accuracy ----------------
def selftest():
    """Validate the scorer on known (gold, prediction, expected) triples. A broken/incompatible
    Math-Verify (or wrong API) shows up HERE, not as a fake ~0% accuracy on the real eval."""
    cases = [
        (r"The answer is $\boxed{\frac{1}{2}}$.", r"... therefore the final answer is \boxed{\frac{1}{2}}.", 1),
        (r"$\boxed{0.5}$",        r"so \boxed{\frac{1}{2}}", 1),     # 0.5 == 1/2
        (r"$\boxed{2}$",          r"the answer is \boxed{3}", 0),
        (r"$\boxed{\sqrt{2}}$",   r"\boxed{2^{1/2}}", 1),            # sqrt2 == 2^(1/2)
        (r"$\boxed{(0,1)}$",      r"the interval \boxed{(0,1)}", 1), # interval equality
        (r"$\boxed{12}$",         r"there is no boxed answer here", 0),  # missing boxed -> wrong
    ]
    ok = 0
    for g, p, exp in cases:
        got = is_correct(parse_gold(g), parse_fallback(p))
        flag = "OK" if got == exp else "MISMATCH"
        print(f"[SELFTEST] expect {exp} got {got} {flag} | gold={g[:34]!r} pred={p[:34]!r}", flush=True)
        ok += int(got == exp)
    print(f"[SELFTEST] {ok}/{len(cases)} Math-Verify checks passed", flush=True)
    contract = [
        # fallback accepts an explicit unboxed terminal answer; strict does not.
        (not parse_strict(r"Final answer: $\frac{1}{2}$") and
         is_correct(parse_gold(r"\boxed{1/2}"), parse_fallback(r"Final answer: $\frac{1}{2}$")) == 1),
        # A gold-looking intermediate number followed by prose is not an answer.
        (parse_fallback("We considered 50 degrees in an intermediate step.\nNo final answer was given.") is None),
        # The terminal answer wins over an earlier correct intermediate value.
        (is_correct(parse_gold(r"\boxed{2}"), parse_fallback("Intermediate: $2$.\nFinal answer: $3$")) == 0),
    ]
    print(f"[SELFTEST contract] {sum(contract)}/{len(contract)} terminal-answer checks passed", flush=True)
    return ok == len(cases) and all(contract)


# ---------------- evaluate (resumable; mirrors kd_sft_common.evaluate but Math-Verify scored) ----------------
@torch.no_grad()
def evaluate(model, tok, ds, eval_bs, maxlen, maxnew, tag, records_path=None, log_every=5):
    model.eval()
    old_ps, old_ts = tok.padding_side, tok.truncation_side
    tok.padding_side = "left"
    tok.truncation_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot is None or eot == tok.unk_token_id:
        eot = tok.eos_token_id

    records = []
    if records_path and os.path.exists(records_path):
        try:
            records = json.load(open(records_path, encoding="utf-8"))
            records = records[: (len(records) // eval_bs) * eval_bs]
            if records:
                print(f"[RESUME {tag}] loaded {len(records)} prior records", flush=True)
        except Exception as e:
            print(f"[RESUME {tag}] partial load failed ({e!r}); fresh", flush=True)
            records = []
    start = len(records)

    try:
        for i in range(start, len(ds), eval_bs):
            bt = ds.select(range(i, min(i + eval_bs, len(ds))))
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": INSTR.format(q=q)}],
                    add_generation_prompt=True, tokenize=False,
                )
                for q in bt["problem"]
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                      max_length=maxlen, add_special_tokens=False).to(model.device)
            input_len = enc["input_ids"].shape[1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_ids = model.generate(**enc, max_new_tokens=maxnew, do_sample=False,
                                         pad_token_id=pad, eos_token_id=[tok.eos_token_id, eot])
            for j, sol in enumerate(bt["solution"]):
                new_ids = out_ids[j][input_len:].detach().cpu().tolist()
                stop_ids = {int(tok.eos_token_id), int(eot)}
                stop_pos = next((p for p, t in enumerate(new_ids) if int(t) in stop_ids), None)
                text = tok.decode(new_ids, skip_special_tokens=True)
                gp = parse_gold(sol)
                sp = parse_strict(text)
                fp = parse_fallback(text)
                has_boxed = last_boxed_only_string(text) is not None
                records.append({
                    "idx": i + j,
                    "problem": bt["problem"][j],
                    "level": bt["level"][j],
                    "type": bt["type"][j],
                    "gold": boxed_inner(sol),                # raw boxed string, for logging only
                    "strict_pred": boxed_inner(text),        # raw boxed string from generation
                    "strict_correct": int(has_boxed and is_correct(gp, sp)),
                    "fallback_correct": int(is_correct(gp, fp)),
                    "fallback_source": fallback_source(text),
                    "parse_fail": int(not has_boxed),
                    "generated_tokens_approx": int((stop_pos + 1) if stop_pos is not None else len(new_ids)),
                    "hit_maxnew_approx": int(stop_pos is None and len(new_ids) >= maxnew),
                    "text": text,
                })
            if records_path and ((i // eval_bs) % 4 == 0):
                json.dump(records, open(records_path, "w", encoding="utf-8"), ensure_ascii=False)
            if (i // eval_bs) % log_every == 0:
                sa = sum(r["strict_correct"] for r in records) / len(records)
                fa = sum(r["fallback_correct"] for r in records) / len(records)
                print(f"[EVAL {tag}] {len(records)}/{len(ds)} strict={sa:.4f} fallback={fa:.4f}", flush=True)
    finally:
        if records_path:
            json.dump(records, open(records_path, "w", encoding="utf-8"), ensure_ascii=False)
        tok.padding_side, tok.truncation_side = old_ps, old_ts
    return records


# ---------------- HF helpers (same contract as kd_sft_common) ----------------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return path

def _set_visibility(api, repo, repo_type, public):
    for fn in ("update_repo_visibility", "update_repo_settings"):
        m = getattr(api, fn, None)
        if m is None:
            continue
        try:
            m(repo_id=repo, repo_type=repo_type, private=not public)
        except Exception as e:
            print("[WARN] visibility:", repr(e), flush=True)
        return

def upload_results(api, folder, repo, token, public=True, msg="update", marker="summary.json", retries=3):
    last = None
    for attempt in range(retries):
        try:
            create_repo(repo, repo_type="dataset", private=not public, exist_ok=True, token=token)
            api.upload_folder(folder_path=folder, repo_id=repo, repo_type="dataset",
                              commit_message=msg + " (data)", ignore_patterns=[marker])
            mp = os.path.join(folder, marker)
            if os.path.exists(mp):
                api.upload_file(path_or_fileobj=mp, path_in_repo=marker, repo_id=repo,
                                repo_type="dataset", commit_message=msg + " (marker)")
            _set_visibility(api, repo, "dataset", public)
            return
        except Exception as e:
            last = e
            print(f"[WARN] results upload attempt {attempt + 1}/{retries}: {e!r}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise last

def cell_done(repo, token, repo_type="dataset", marker="summary.json"):
    try:
        p = hf_hub_download(repo_id=repo, filename=marker, repo_type=repo_type, token=token)
        d = json.load(open(p, encoding="utf-8"))
        return (d.get("stage") == "complete" and
                d.get("ruler_version") == RULER_VERSION and
                d.get("instr_sha") == INSTR_SHA)
    except Exception:
        return False

def selfcheck_bos(tok):
    ids = tok(tok.apply_chat_template([{"role": "user", "content": INSTR.format(q="1+1=?")}],
              add_generation_prompt=True, tokenize=False),
              return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    n = ids.count(tok.bos_token_id)
    print(f"[SELF-CHECK] bos_count={n} (expect 1)", flush=True)
    return n
