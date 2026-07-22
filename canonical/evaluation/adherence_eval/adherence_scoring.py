"""
adherence_scoring.py

Final, consolidated adherence-scoring script for the cross-lingual
rule-following project (PRISM Stream 2, "To Rule or Not to Rule").

Brings together three things that were previously spread across three
different notebooks:
  1. The deterministic checkers originally written in Design1Experiments.ipynb
     (check_uppercase, check_lowercase, check_bold, check_italic,
     check_banned_word, check_language, check_word_count, parse_target).
  2. The ack_invert checker from the 3B patching notebook's
     check_rule_following -- kept as the corrected FULL-match version, per
     that notebook's own comment that the row-stored checker string was
     prefix-only buggy.
  3. llm_judge.py's weighted-logprob, coherence-gated Yes/No judge, wired in
     as the fallback path for any category with no deterministic ground
     truth (tone_norm and other cultural/register rules).

Two checkers are genuinely NEW here, because the methodology doc names two
rule categories ("include-word", "start-with-token") that never had a
corresponding function in any of the three uploaded files:
  - check_include_word   (banned_word's mirror image: word must be present)
  - check_start_with_token

One real bug from Design1Experiments.ipynb is fixed here, not just carried
over: its evaluate() called parse_target(cat, row["full_rule"]) for
"banned_word", but parse_target had no "banned_word" branch -- it always
returned None, so check_banned_word(output, None) would have crashed the
moment a banned_word row went through the prose-fallback path instead of
an explicit "word" field. Fixed by adding that branch below.

Design choices, stated up front (same honesty convention as the rest of
this project's code):
  - Every checker returns True / False / None, never raises on bad input.
    A batch run over hundreds of rows should never die partway through
    because one row is missing a field. None means "could not evaluate"
    and is reported separately from adherence failures in the summary --
    collapsing "failed to check" into "did not adhere" would quietly
    inflate the numbers Step 1 of the methodology doc is meant to measure
    honestly.
  - Checkers take their argument from an EXPLICIT field on the row
    (row["target_count"], row["word"], row["lang_code"], row["token"])
    wherever the dataset already carries it, and only fall back to
    parse_target's regex-on-prose extraction when it doesn't. Explicit
    fields are the more reliable convention (already used in the
    tone_norm / word_count JSON files) -- prose-regex was always a
    workaround for datasets that hadn't been given that field yet.
  - evaluate_deterministic() is a fixed category -> function dispatch, not
    an eval() of a stored checker-expression string. The 3B notebook's
    approach (row["checker"] as a literal Python expression run through a
    safe_builtins sandbox) works, but is arbitrary-code-execution-over-data
    even when sandboxed, and fails silently on a typo in the stored string.
    A fixed dispatch fails loudly (unrecognized category is visible, not
    swallowed) and is auditable by reading this one file. If a dataset
    still carries legacy row["checker"] expression strings,
    evaluate_legacy_checker_expr() below reproduces the 3B sandbox
    unchanged, so those rows don't need a rewrite.

KNOWN LIMITATION worth flagging explicitly: check_uppercase / check_lowercase
rely on Python's str.isupper()/.islower(), which is only meaningful for
cased scripts (Latin, Cyrillic, Greek...). Devanagari (Hindi) has no case
distinction, so every Devanagari letter is neither isupper() nor islower()
-- meaning check_uppercase on real Hindi text will always return False,
regardless of what the model actually produced. This isn't a bug so much
as a real fact about the category: "uppercase" as a rule concept may not
transfer to Hindi at all, and that itself might be worth reporting rather
than silently working around. Swahili is Latin-script, so it's unaffected.

HONEST LIMITATION carried over from llm_judge.py unchanged: the LLM-judge
path uses the SAME model as generator and judge unless you pass a separate
`judge_model` explicitly -- self-evaluation, not independent evaluation.
Treat any adherence numbers that went through that path as first-pass.
"""

import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # deterministic langdetect
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

from llm_judge import llm_judge_compliance  # reused, not reimplemented


# ---------------------------------------------------------------------------
# Deterministic checkers -- carried over from Design1Experiments.ipynb
# ---------------------------------------------------------------------------

def check_uppercase(output: str) -> bool:
    """See KNOWN LIMITATION above re: cased vs. caseless scripts."""
    letters = [c for c in output if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def check_lowercase(output: str) -> bool:
    letters = [c for c in output if c.isalpha()]
    if not letters:
        return False
    return all(c.islower() for c in letters)


def check_banned_word(output: str, word: Optional[str]) -> Optional[bool]:
    if not word:
        return None  # guards the Design1 crash path -- see module docstring
    return re.search(rf"\b{re.escape(word)}\b", output, flags=re.IGNORECASE) is None


def check_language(output: str, lang_code: Optional[str], min_words: int = 6) -> Optional[bool]:
    # Too short to trust langdetect -> cannot confirm -> False (per Design1)
    if len(output.split()) < min_words:
        return False
    if not lang_code:
        return None
    if not _HAS_LANGDETECT:
        return None  # softened from Design1's `raise RuntimeError` -- a missing
                     # optional dependency shouldn't kill a whole batch run
    try:
        return detect(output) == lang_code
    except Exception:
        return False


def check_word_count(output: str, n: Optional[int]) -> Optional[bool]:
    if n is None:
        return None
    return len(output.strip().split()) == n


def check_bold(output: str) -> bool:
    return re.search(r"\*\*(?!\s*\*)(.+?)\*\*", output, flags=re.DOTALL) is not None


def check_italic(output: str) -> bool:
    if re.search(r"\*\*\*(?!\s*\*)(.+?)\*\*\*", output, flags=re.DOTALL):
        return True
    without_bold = re.sub(r"\*\*.+?\*\*", "", output, flags=re.DOTALL)
    return re.search(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", without_bold, flags=re.DOTALL) is not None


# ---------------------------------------------------------------------------
# New checkers -- categories the methodology doc names (banned-word,
# uppercase, language-adherence, word-count, bold/formatting, include-word,
# start-with-token) that had no function anywhere in the uploaded files.
# ---------------------------------------------------------------------------

def check_include_word(output: str, word: Optional[str]) -> Optional[bool]:
    """Mirror image of check_banned_word: satisfied when the word IS present."""
    if not word:
        return None
    return re.search(rf"\b{re.escape(word)}\b", output, flags=re.IGNORECASE) is not None


def check_start_with_token(output: str, token: Optional[str]) -> Optional[bool]:
    """For the 'start-with-token' design (methodology doc's Design 3,
    recognition-in-output: 'Begin with X'). Case-insensitive, tolerant of
    leading whitespace and one immediately-following comma/colon, so a
    required token of 'Understood' matches an output starting 'Understood,'."""
    if not token:
        return None
    stripped = output.strip()
    return re.match(rf"^{re.escape(token)}\s*[,:]?\b", stripped, flags=re.IGNORECASE) is not None


def check_ack_invert(output: str) -> bool:
    """From the 3B patching notebook's check_rule_following -- kept as the
    corrected FULL-match version. The prefix-only original would count
    'Understood, no -- let me explain more' as compliant; this requires the
    whole response to be just the acknowledgment + yes/no, nothing trailing."""
    return bool(re.fullmatch(r"\s*understood[\s\W]+(yes|no)[\s\W]*", output, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Target-argument resolution: prefer an explicit row field, fall back to
# Design1's prose-regex parsing only if the field is genuinely absent.
# ---------------------------------------------------------------------------

def parse_target(category: str, full_rule: str) -> Optional[Any]:
    """Design1Experiments.ipynb's parse_target, plus a banned_word branch
    that was missing there (see module docstring for why that was a bug)
    and a start_with_token branch for the new category."""
    if not full_rule:
        return None
    if category == "include_word":
        m = re.search(r"with the word (\w+)", full_rule, flags=re.IGNORECASE)
        return m.group(1) if m else None
    if category == "language":
        m = re.search(r"in (\w+)\.", full_rule)
        langmap = {"French": "fr", "Spanish": "es", "German": "de", "Italian": "it",
                   "Russian": "ru", "Swahili": "sw", "Hindi": "hi"}
        return langmap.get(m.group(1)) if m else None
    if category == "word_count":
        words = {"three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
        m = re.search(r"in (\w+) words", full_rule, flags=re.IGNORECASE)
        return words.get(m.group(1).lower()) if m else None
    if category == "banned_word":
        m = re.search(r"never use the word ['\"]?(\w+)['\"]?", full_rule, flags=re.IGNORECASE)
        return m.group(1) if m else None
    if category == "start_with_token":
        m = re.search(r"begin (?:your response )?with ['\"]?(\w+)['\"]?", full_rule, flags=re.IGNORECASE)
        return m.group(1) if m else None
    return None


def _resolve_target(row: Dict[str, Any], explicit_key: str, category: str) -> Optional[Any]:
    if row.get(explicit_key) is not None:
        return row[explicit_key]
    return parse_target(category, row.get("full_rule", ""))


# ---------------------------------------------------------------------------
# Category -> checker dispatch
# ---------------------------------------------------------------------------

_DETERMINISTIC_CATEGORIES = {
    "uppercase", "lowercase", "bold", "italic", "banned_word",
    "include_word", "language", "word_count", "start_with_token", "ack_invert",
}


def evaluate_deterministic(row: Dict[str, Any], output: str) -> Optional[bool]:
    """Dispatches by row['category']. Returns None for an unrecognized
    category rather than raising, since callers (score_adherence,
    run_adherence_scoring) need to keep going over a whole batch."""
    cat = row.get("category")
    if cat == "uppercase":
        return check_uppercase(output)
    if cat == "lowercase":
        return check_lowercase(output)
    if cat == "bold":
        return check_bold(output)
    if cat == "italic":
        return check_italic(output)
    if cat == "banned_word":
        return check_banned_word(output, _resolve_target(row, "word", cat))
    if cat == "include_word":
        return check_include_word(output, _resolve_target(row, "word", cat))
    if cat == "language":
        return check_language(output, _resolve_target(row, "lang_code", cat))
    if cat == "word_count":
        return check_word_count(output, _resolve_target(row, "target_count", cat))
    if cat == "start_with_token":
        return check_start_with_token(output, _resolve_target(row, "token", cat))
    if cat == "ack_invert":
        return check_ack_invert(output)
    return None


# ---------------------------------------------------------------------------
# Unified entry point: deterministic where possible, LLM-judge otherwise
# ---------------------------------------------------------------------------

def score_adherence(row: Dict[str, Any], response: str, judge_model=None) -> Dict[str, Any]:
    """
    Returns {"compliant": bool|None, "method": "checker"|"llm_judge"|"unscored", ...}.
    "llm_judge" results also carry p_comply / p_coherent / *_entropy /
    low_confidence, straight from llm_judge_compliance.

    Routing:
      1. row["category"] is one of the deterministic categories AND the row
         isn't explicitly flagged for manual/judge scoring (row["checker"]
         containing "manual" or "llm-judge", the convention already used in
         the 3B notebook's data) -> deterministic checker.
      2. Otherwise, if judge_model is supplied -> llm_judge_compliance
         (tone_norm and any other category with no ground-truth checker
         lands here).
      3. Otherwise -> method="unscored", explicit rather than a silent None,
         so a batch summary reports "N rows need a judge you didn't supply"
         instead of quietly counting them as failures.
    """
    checker_field = str(row.get("checker", "")).lower()
    forced_manual = "manual" in checker_field or "llm-judge" in checker_field
    cat = row.get("category")

    if cat in _DETERMINISTIC_CATEGORIES and not forced_manual:
        return {"compliant": evaluate_deterministic(row, response), "method": "checker"}

    if judge_model is not None:
        rule_clause = row.get("rule_clause") or row.get("full_rule") or ""
        verdict = llm_judge_compliance(judge_model, rule_clause, response)
        verdict["method"] = "llm_judge"
        return verdict

    return {"compliant": None, "method": "unscored",
            "note": f"category '{cat}' has no deterministic checker and no judge_model was passed"}


# ---------------------------------------------------------------------------
# Batch runner -- Step 1 of the methodology doc: "Score all prompts x
# languages x conditions with checker functions. Output: adherence rate per
# (model, language, rule category)."
# ---------------------------------------------------------------------------

def run_adherence_scoring(pairs: List[Dict[str, Any]], responses: List[str],
                           language: str, judge_model=None) -> Dict[str, Any]:
    if len(pairs) != len(responses):
        raise ValueError(f"pairs ({len(pairs)}) and responses ({len(responses)}) must be the same length")

    per_row = []
    for row, response in zip(pairs, responses):
        result = score_adherence(row, response, judge_model=judge_model)
        per_row.append({"id": row.get("id"), "category": row.get("category"),
                         "language": language, "response": response, **result})

    by_category = defaultdict(list)
    for r in per_row:
        by_category[r["category"]].append(r)

    summary = {}
    for cat, rows in by_category.items():
        scored = [r for r in rows if r["compliant"] is not None]
        low_conf = sum(1 for r in rows if r.get("low_confidence"))
        summary[cat] = {
            "n": len(rows),
            "n_scored": len(scored),
            "n_unscored": len(rows) - len(scored),
            "adherence_rate": (sum(r["compliant"] for r in scored) / len(scored)) if scored else None,
            "n_low_confidence": low_conf,
        }

    return {"language": language, "per_row": per_row, "summary": summary}


def print_adherence_summary(result: Dict[str, Any]) -> None:
    print(f"\nAdherence summary -- {result['language']}")
    print(f"{'category':<18} {'n':>4} {'scored':>7} {'unscored':>9} {'adherence':>10} {'low-conf':>9}")
    for cat, s in sorted(result["summary"].items()):
        rate_str = f"{s['adherence_rate']:.0%}" if s["adherence_rate"] is not None else "n/a"
        print(f"{cat:<18} {s['n']:>4} {s['n_scored']:>7} {s['n_unscored']:>9} {rate_str:>10} {s['n_low_confidence']:>9}")


def compare_languages(results_by_language: Dict[str, Dict[str, Any]]) -> None:
    """Side-by-side adherence-rate table across languages, per category --
    the cross-lingual comparison Step 5 of the methodology doc asks for at
    the behavioral level (before any circuit-level analysis)."""
    languages = list(results_by_language.keys())
    all_categories = sorted({cat for r in results_by_language.values() for cat in r["summary"]})
    header = f"{'category':<18}" + "".join(f"{lang:>12}" for lang in languages)
    print(header)
    for cat in all_categories:
        row_str = f"{cat:<18}"
        for lang in languages:
            s = results_by_language[lang]["summary"].get(cat)
            rate = s["adherence_rate"] if s else None
            row_str += f"{(f'{rate:.0%}' if rate is not None else 'n/a'):>12}"
        print(row_str)


# ---------------------------------------------------------------------------
# Legacy support: the 3B patching notebook's convention of a literal Python
# boolean expression in row["checker"], evaluated in a restricted sandbox.
# Reproduced unchanged so datasets still generated in that format keep
# working -- new datasets should prefer the category+field dispatch above.
# ---------------------------------------------------------------------------

def evaluate_legacy_checker_expr(row: Dict[str, Any], response: str) -> Optional[bool]:
    checker = str(row.get("checker", ""))
    if "manual" in checker.lower() or "llm-judge" in checker.lower():
        return None
    if row.get("category") == "ack_invert":
        return check_ack_invert(response)
    safe_builtins = {"len": len, "str": str, "int": int, "bool": bool, "min": min,
                      "max": max, "sum": sum, "any": any, "all": all}
    local_ns = {"out": response, "target_count": row.get("target_count")}
    try:
        return bool(eval(checker, {"__builtins__": safe_builtins}, local_ns))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sanity checks -- every checker exercised on a clear pass/fail case, plus
# routing logic, before this touches any real generation.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    assert check_uppercase("HELLO WORLD") is True
    assert check_uppercase("Hello World") is False
    assert check_lowercase("hello world") is True
    assert check_lowercase("Hello") is False
    assert check_banned_word("this is fine", "guarantee") is True
    assert check_banned_word("I guarantee it", "guarantee") is False
    assert check_banned_word("I guarantee it", None) is None
    assert check_include_word("I guarantee it", "guarantee") is True
    assert check_include_word("this is fine", "guarantee") is False
    assert check_word_count("one two three", 3) is True
    assert check_word_count("one two three four", 3) is False
    assert check_bold("this is **bold** text") is True
    assert check_bold("no bold here") is False
    assert check_italic("this is *italic* text") is True
    assert check_italic("this is **bold** text") is False
    assert check_start_with_token("Understood, let's continue", "Understood") is True
    assert check_start_with_token("Sure, Understood later", "Understood") is False
    assert check_ack_invert("Understood, no.") is True
    assert check_ack_invert("Understood, no -- let me explain more") is False
    print("All deterministic checker sanity checks passed.")

    # explicit-field dispatch path
    assert evaluate_deterministic({"category": "word_count", "target_count": 3}, "one two three") is True
    assert evaluate_deterministic({"category": "include_word", "word": "cat"}, "I have a cat") is True

    # prose-fallback dispatch path -- also confirms the banned_word bug fix
    row = {"category": "banned_word", "full_rule": "Never use the word 'guarantee'."}
    assert evaluate_deterministic(row, "I guarantee it") is False
    assert evaluate_deterministic(row, "this is fine") is True
    print("Dispatch (explicit-field and prose-fallback paths, including the banned_word fix) all correct.")

    # score_adherence routing
    r1 = score_adherence({"category": "word_count", "target_count": 3}, "one two three")
    assert r1["method"] == "checker" and r1["compliant"] is True

    r2 = score_adherence({"category": "tone_norm"}, "some response")
    assert r2["method"] == "unscored"

    r3 = score_adherence({"category": "word_count", "target_count": 3, "checker": "manual"}, "one two three")
    assert r3["method"] == "unscored"  # forced manual, no judge_model -> stays unscored, never silently checked
    print("score_adherence routing all correct.")

    # run_adherence_scoring + summary, mixed categories, no judge_model
    pairs = [
        {"id": "a", "category": "word_count", "target_count": 3},
        {"id": "b", "category": "word_count", "target_count": 3},
        {"id": "c", "category": "banned_word", "word": "guarantee"},
        {"id": "d", "category": "tone_norm"},  # no checker, no judge_model -> unscored
    ]
    responses = ["one two three", "one two three four", "I guarantee it", "some response"]
    result = run_adherence_scoring(pairs, responses, language="EN")
    assert result["summary"]["word_count"]["n"] == 2
    assert result["summary"]["word_count"]["adherence_rate"] == 0.5
    assert result["summary"]["banned_word"]["adherence_rate"] == 0.0
    assert result["summary"]["tone_norm"]["n_unscored"] == 1
    print_adherence_summary(result)

    print("\nAll sanity checks passed. Still needs a real judge_model + real generations to")
    print("confirm the llm_judge branch end-to-end -- that path is exercised in llm_judge.py's")
    print("own __main__ block, not re-tested here, since it needs a loaded HookedTransformer.")
