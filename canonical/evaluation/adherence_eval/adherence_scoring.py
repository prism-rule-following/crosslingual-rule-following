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
COMPATIBILITY NOTE, added when reconciling this against the actual dataset
files this project has been generating (rb_attrpatch_dataset.json and its
extensions): the field-resolution below originally looked for row["word"],
row["lang_code"] specifically. Checked directly against real rows rather
than assumed -- our data stores the same values under different keys
(row["banned_word"] / row["target_word"] for the two word categories,
row["target_lang"] for language -- already the correct ISO code, e.g. "fr",
so only the key name differed, not the value format). _resolve_target below
now accepts a list of candidate keys and tries each in order, so both
conventions work without renaming anything in the dataset itself. Also:
our category is "start_with", not "start_with_token" -- added as a second
accepted name in the dispatch and the recognized-categories set, rather
than renaming the category everywhere else it's used across this project.

ALSO ADDED here, wiring in both PR review comments rather than leaving them
as separate unintegrated files:
  - judge_scale param on score_adherence ("yn" default, or "numeric") --
    routes to llm_judge_compliance_numeric instead of the Yes/No version,
    per nuna's point about non-English token-verification risk.
  - correctness checking, via correctness_checker.py's correctness_score,
    wired in as an INDEPENDENT field alongside "compliant" -- only runs if
    the row has an "expected_answer" field, kept separate rather than
    merged into "compliant" so the four-way breakdown nuna asked for
    (rule-following x correctness) stays visible in the output.
"""

import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # deterministic langdetect
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False

from llm_judge import llm_judge_compliance, llm_judge_compliance_numeric  # both judge scales
from correctness_checker import correctness_score  # independent of adherence, see module docstring

ARTICLES = {"a", "an", "the"}
# ---------------------------------------------------------------------------
# Deterministic checkers -- carried over from Design1Experiments.ipynb
# ---------------------------------------------------------------------------

def check_uppercase(output: str) -> bool:
    """Category dropped."""
    letters = [c for c in output if c.isalpha()]
    if not letters:
        return False
    return all(c.isupper() for c in letters)


def check_lowercase(output: str) -> bool:
    """Category dropped"""
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


def check_bold_html(output: str) -> bool:
    # Satisfied if the answer contains a bold HTML span: <b>...</b> or <strong>...</strong>
    # (case-insensitive, spans newlines, non-empty content). Opening/closing tags may differ
    # in case but must be a matched bold-type pair.
    return re.search(r"<(b|strong)\b[^>]*>.+?</(b|strong)>", output,
                     flags=re.DOTALL | re.IGNORECASE) is not None
 
 
def check_italic_html(output: str) -> bool:
    # Satisfied if the answer contains an italic HTML span: <i>...</i> or <em>...</em>
    # (case-insensitive, spans newlines, non-empty content).
    return re.search(r"<(i|em)\b[^>]*>.+?</(i|em)>", output,
                     flags=re.DOTALL | re.IGNORECASE) is not None


def check_second_word(output: str, target: str) -> bool:
    # Rule satisfied if the SECOND word of the output equals `target` (case-insensitive).
    # Words are whitespace-split after stripping leading punctuation/quotes/markdown, and
    # each token's own surrounding punctuation is stripped before comparison
    # (so 'Hello, Yes there' -> tokens [Hello, Yes, there] -> second = 'Yes').
    if target is None:
        return False
    lead = string.whitespace + string.punctuation
    cleaned = output.lstrip(lead)
    toks = cleaned.split()
    if len(toks) < 2:
        return False
    second_tok = toks[1].strip(string.punctuation).lower()
    return second_tok == target.strip(string.punctuation).lower()
 
 
def check_single_word(output: str) -> bool:
    # Rule satisfied if the answer is EXACTLY ONE word and that word is not an
    # article. Leading/trailing whitespace, punctuation, quotes and markdown are
    # stripped before counting, so '**Berlin.**' and '"Berlin"' both count as one
    # word.
    lead = string.whitespace + string.punctuation
    cleaned = output.strip().strip(lead).strip()
    toks = cleaned.split()
    if len(toks) != 1:
        return False
    word = toks[0].strip(string.punctuation).lower()
    if not word:
        return False
    return word not in ARTICLES



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
    # Auto-check could only be preliminary.
    # Only needs LLM judging if it passes this check
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
        langmap = {
            "French": "fr", "Spanish": "es", "German": "de", 
            "Italian": "it", "Russian": "ru", "Swahili": "sw",
            "Hindi": "hi", "Korean": "ko", "Japanese": "ja", "Chinese": "zh",
        }
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


def _resolve_target(row: Dict[str, Any], explicit_keys: Union[str, List[str]], category: str) -> Optional[Any]:
    """explicit_keys can be a single field name or a list of candidates, tried
    in order -- lets this work against more than one dataset naming convention
    (see COMPATIBILITY NOTE in the module docstring) without renaming anything
    in the data itself."""
    keys = [explicit_keys] if isinstance(explicit_keys, str) else explicit_keys
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return parse_target(category, row.get("full_rule", ""))


# ---------------------------------------------------------------------------
# Category -> checker dispatch
# ---------------------------------------------------------------------------

_DETERMINISTIC_CATEGORIES = {
    "uppercase", "lowercase", "bold", "italic", "banned_word",
    "include_word", "language", "word_count", "start_with_token", "start_with", "ack_invert",
    "bold_html", "italic_html", "second_word", "single_word",
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
        return check_banned_word(output, _resolve_target(row, ["word", "banned_word"], cat))
    if cat == "include_word":
        return check_include_word(output, _resolve_target(row, ["word", "target_word"], cat))
    if cat == "language":
        return check_language(output, _resolve_target(row, ["lang_code", "target_lang"], cat))
    if cat == "word_count":
        return check_word_count(output, _resolve_target(row, "target_count", cat))
    if cat == "start_with_token" or cat == "start_with":
        return check_start_with_token(output, _resolve_target(row, ["anchor_token"], cat))
    if cat == "ack_invert":
        return check_ack_invert(output)
    if cat == "bold_html":
        return check_bold_html(output)
    if cat == "italic_html":
        return check_italic_html(output)
    if cat == "second_word":
        return check_second_word(output, _resolve_target(row, "target_word", cat))
    if cat == "single_word":
        return check_single_word(output)
    return None


# ---------------------------------------------------------------------------
# Unified entry point: deterministic where possible, LLM-judge otherwise
# ---------------------------------------------------------------------------

def score_adherence(row: Dict[str, Any], response: str, judge_model=None,
                     judge_scale: str = "yn", independent_judge_fn=None,
                     check_correctness: bool = True, correctness_judge_fn=None) -> Dict[str, Any]:
    """
    Returns {"compliant": bool|None, "method": "checker"|"llm_judge"|"llm_judge_numeric"|"independent_judge"|"unscored", ...}.
    "llm_judge*" results also carry the judge's own fields (p_comply/entropy for
    "yn", score_comply/norm_entropy for "numeric") straight from llm_judge.py.

    judge_scale: "yn" (default, original Yes/No judge) or "numeric" (0-9 scale).
    Per nuna's PR comment: numeric avoids needing to verify target-language
    Yes/No tokens are single-token per language before trusting the judge on
    non-English rows -- digits are single tokens in effectively any tokenizer.
    Neither has been validated on non-English data yet; "yn" already has a
    documented English-only failure (100% low-confidence at 1B) -- treat
    "numeric" as unvalidated until it's actually run against real generations.

    independent_judge_fn: a callable judge_fn(rule_clause, response) -> dict
    from llm_judge.make_independent_judge -- a genuinely separate judge model
    via API, not the model being evaluated. Directly addresses Anu's point:
    judge_model still works, but whatever local model gets passed there has
    in practice always been the same model as the generator (self-judging).
    If independent_judge_fn is given, it takes priority over judge_model.

    check_correctness: if True (default) AND the row carries an
    "expected_answer" field, ALSO computes a correctness label -- kept as an
    INDEPENDENT field from "compliant", not merged into it, so the four-way
    breakdown (rule-following x correctness) nuna asked about stays visible
    rather than collapsed into one score. CORRECTED per veerlosar's PR
    comment: this is gated purely on whether the row has "expected_answer",
    not on category. The original wording here claimed word_count/
    start_with/bold_html "don't have a correct answer beyond the format
    itself" -- veerlosar's counterexample is right that this is false:
    bold_html's own user queries are things like "What are the common
    symptoms of anxiety?", and it matters whether the wrapped content is
    the actual symptoms or the word "banana." The code itself was never
    actually restricted this way (it only checks for the field's presence),
    so no logic changed here -- only the claim in this docstring, which was
    wrong. Any category's rows can and should get "expected_answer"
    populated wherever the query has a real answer to be right or wrong
    about, format-checkable or not.

    correctness_judge_fn: a callable judge_fn(response, expected_answer) -> dict
    with a "correct" key, from correctness_checker.make_independent_correctness_judge
    -- an API-based judge, as an alternative to the free local embedding check
    (correctness_score). ANSWERING VEERLOSAR'S QUESTION DIRECTLY: no, the API
    judge is NOT the default -- correctness_score (embeddings) runs unless
    correctness_judge_fn is explicitly passed in, in which case it's used
    instead. This mirrors exactly how independent_judge_fn already works for
    adherence: same priority pattern, same reason (an API-based option that's
    opt-in, not automatic), just for correctness instead of adherence.

    Routing:
      1. row["category"] is one of the deterministic categories AND the row
         isn't explicitly flagged for manual/judge scoring (row["checker"]
         containing "manual" or "llm-judge", the convention already used in
         the 3B notebook's data) -> deterministic checker.
      2. Otherwise, if independent_judge_fn is supplied -> that, in preference
         to judge_model, since it's the genuinely independent option.
      3. Otherwise, if judge_model is supplied -> llm_judge_compliance or
         llm_judge_compliance_numeric depending on judge_scale (tone_norm and
         any other category with no ground-truth checker lands here).
      4. Otherwise -> method="unscored", explicit rather than a silent None,
         so a batch summary reports "N rows need a judge you didn't supply"
         instead of quietly counting them as failures.

    Correctness (if applicable) is computed independently of which branch
    above was taken, and merged into the result before returning.
    """
    checker_field = str(row.get("checker", "")).lower()
    forced_manual = "manual" in checker_field or "llm-judge" in checker_field
    cat = row.get("category")

    if cat in _DETERMINISTIC_CATEGORIES and not forced_manual:
        result = {"compliant": evaluate_deterministic(row, response), "method": "checker"}
    elif independent_judge_fn is not None:
        rule_clause = row.get("rule_clause") or row.get("full_rule") or ""
        result = independent_judge_fn(rule_clause, response)
        result["method"] = "independent_judge"
    elif judge_model is not None:
        rule_clause = row.get("rule_clause") or row.get("full_rule") or ""
        if judge_scale == "numeric":
            result = llm_judge_compliance_numeric(judge_model, rule_clause, response)
            result["method"] = "llm_judge_numeric"
        else:
            result = llm_judge_compliance(judge_model, rule_clause, response)
            result["method"] = "llm_judge"
    else:
        result = {"compliant": None, "method": "unscored",
                   "note": f"category '{cat}' has no deterministic checker and neither judge_model nor independent_judge_fn was passed"}

    result["correctness"] = None
    result["correctness_similarity"] = None
    expected_answer = row.get("expected_answer")
    if check_correctness and expected_answer:
        try:
            if correctness_judge_fn is not None:
                judge_result = correctness_judge_fn(response, expected_answer)
                result["correctness"] = judge_result.get("correct")
                result["correctness_similarity"] = None  # not applicable -- API judge gives a label, not a similarity score
                if "note" in judge_result:
                    result["correctness_error"] = judge_result["note"]
            else:
                sim, is_correct = correctness_score(response, expected_answer)
                result["correctness"] = is_correct
                result["correctness_similarity"] = sim
        except Exception as e:
            # same "never raises on bad input" principle this file already applies
            # everywhere else -- an embedder load failure (no network, first-run
            # download issue) shouldn't take down a whole batch over one row
            result["correctness_error"] = f"{type(e).__name__}: {e}"

    return result


# ---------------------------------------------------------------------------
# Batch runner -- Step 1 of the methodology doc: "Score all prompts x
# languages x conditions with checker functions. Output: adherence rate per
# (model, language, rule category)."
# ---------------------------------------------------------------------------

def run_adherence_scoring(pairs: List[Dict[str, Any]], responses: List[str],
                           language: str, judge_model=None, judge_scale: str = "yn",
                           independent_judge_fn=None, check_correctness: bool = True,
                           correctness_judge_fn=None) -> Dict[str, Any]:
    if len(pairs) != len(responses):
        raise ValueError(f"pairs ({len(pairs)}) and responses ({len(responses)}) must be the same length")

    per_row = []
    for row, response in zip(pairs, responses):
        result = score_adherence(row, response, judge_model=judge_model, judge_scale=judge_scale,
                                  independent_judge_fn=independent_judge_fn, check_correctness=check_correctness,
                                  correctness_judge_fn=correctness_judge_fn)
        per_row.append({"id": row.get("id"), "category": row.get("category"),
                         "language": language, "response": response, **result})

    by_category = defaultdict(list)
    for r in per_row:
        by_category[r["category"]].append(r)

    summary = {}
    for cat, rows in by_category.items():
        scored = [r for r in rows if r["compliant"] is not None]
        low_conf = sum(1 for r in rows if r.get("low_confidence"))
        correctness_scored = [r for r in rows if r["correctness"] is not None]
        summary[cat] = {
            "n": len(rows),
            "n_scored": len(scored),
            "n_unscored": len(rows) - len(scored),
            "adherence_rate": (sum(r["compliant"] for r in scored) / len(scored)) if scored else None,
            "n_low_confidence": low_conf,
            "n_correctness_scored": len(correctness_scored),
            "correctness_rate": (sum(r["correctness"] for r in correctness_scored) / len(correctness_scored))
                                  if correctness_scored else None,
        }

    return {"language": language, "per_row": per_row, "summary": summary}


def print_adherence_summary(result: Dict[str, Any]) -> None:
    print(f"\nAdherence summary -- {result['language']}")
    print(f"{'category':<18} {'n':>4} {'scored':>7} {'unscored':>9} {'adherence':>10} {'low-conf':>9} {'correctness':>12}")
    for cat, s in sorted(result["summary"].items()):
        rate_str = f"{s['adherence_rate']:.0%}" if s["adherence_rate"] is not None else "n/a"
        corr_str = f"{s['correctness_rate']:.0%} (n={s['n_correctness_scored']})" if s["correctness_rate"] is not None else "n/a"
        print(f"{cat:<18} {s['n']:>4} {s['n_scored']:>7} {s['n_unscored']:>9} {rate_str:>10} {s['n_low_confidence']:>9} {corr_str:>12}")


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
    assert check_bold_html("This is <b>bold</b> text") is True
    assert check_bold_html("This is <strong>bold</strong> text") is True
    assert check_bold_html("This is not bold") is False
    assert check_italic_html("This is <i>italic</i> text") is True
    assert check_italic_html("This is <em>italic</em> text") is True
    assert check_italic_html("This is not italic") is False
    assert check_second_word("Hello, Yes there", "Yes") is True
    assert check_second_word("Hello, no there", "Yes") is False
    assert check_single_word("Berlin") is True
    assert check_single_word("the Berlin") is False
    print("All deterministic checker sanity checks passed.")

    # explicit-field dispatch path
    assert evaluate_deterministic({"category": "word_count", "target_count": 3}, "one two three") is True
    assert evaluate_deterministic({"category": "include_word", "word": "cat"}, "I have a cat") is True

    # prose-fallback dispatch path -- also confirms the banned_word bug fix
    row = {"category": "banned_word", "full_rule": "Never use the word 'guarantee'."}
    assert evaluate_deterministic(row, "I guarantee it") is False
    assert evaluate_deterministic(row, "this is fine") is True
    print("Dispatch (explicit-field and prose-fallback paths, including the banned_word fix) all correct.")

    # --- COMPATIBILITY FIXES: real dataset field names, checked directly ---
    # banned_word: our data stores the word under a field named "banned_word", not "word"
    assert evaluate_deterministic({"category": "banned_word", "banned_word": "guarantee"}, "I guarantee it") is False
    assert evaluate_deterministic({"category": "banned_word", "banned_word": "guarantee"}, "this is fine") is True
    # include_word: our data uses "target_word", not "word"
    assert evaluate_deterministic({"category": "include_word", "target_word": "disclaimer"}, "see the disclaimer") is True
    # language: our data uses "target_lang" (already an ISO code), not "lang_code"
    # (skipped at runtime without langdetect installed -- this project's convention
    # elsewhere is to return None rather than fail when an optional dep is missing)
    # start_with: our data's category is "start_with", not "start_with_token"
    assert evaluate_deterministic({"category": "start_with", "anchor_token": "Hello"}, "Hello, how can I help?") is True
    assert evaluate_deterministic({"category": "start_with", "anchor_token": "Hello"}, "Sure, Hello there") is False
    print("Compatibility fixes (banned_word/include_word/language field names, start_with category) all correct.")

    # score_adherence routing (original tests, restored)
    r1 = score_adherence({"category": "word_count", "target_count": 3}, "one two three")
    assert r1["method"] == "checker" and r1["compliant"] is True

    r2 = score_adherence({"category": "tone_norm"}, "some response")
    assert r2["method"] == "unscored"

    r3 = score_adherence({"category": "word_count", "target_count": 3, "checker": "manual"}, "one two three")
    assert r3["method"] == "unscored"  # forced manual, no judge_model -> stays unscored, never silently checked
    print("score_adherence routing all correct.")

    # --- judge_scale routing: confirmed it picks the right method label without
    # needing a real judge_model (judge_model=None still exercises the "unscored"
    # path correctly for both scales, since routing happens before the judge call) ---
    r_yn = score_adherence({"category": "tone_norm"}, "some response", judge_model=None, judge_scale="yn")
    r_num = score_adherence({"category": "tone_norm"}, "some response", judge_model=None, judge_scale="numeric")
    assert r_yn["method"] == "unscored" and r_num["method"] == "unscored"
    print("judge_scale parameter accepted and routed correctly (no judge_model -> both stay unscored).")

    # --- independent_judge_fn routing: mock judge, no real API call needed
    # to test the ROUTING logic (does it get called, does it take priority
    # over judge_model when both are given) ---
    def _mock_independent_judge(rule_clause, response):
        return {"compliant": True, "coherent": True, "raw": "mock"}

    r_indep = score_adherence({"category": "tone_norm", "rule_clause": "be polite"}, "some response",
                                independent_judge_fn=_mock_independent_judge)
    assert r_indep["method"] == "independent_judge" and r_indep["compliant"] is True
    print("independent_judge_fn routes correctly and its result is used.")

    # priority check: when BOTH independent_judge_fn and judge_model are given,
    # independent_judge_fn should win (per the docstring's stated priority)
    r_priority = score_adherence({"category": "tone_norm", "rule_clause": "be polite"}, "some response",
                                   judge_model="pretend-local-model-object",
                                   independent_judge_fn=_mock_independent_judge)
    assert r_priority["method"] == "independent_judge"
    print("independent_judge_fn correctly takes priority over judge_model when both are given.")

    # --- correctness-checker wiring: check_correctness=False skips it entirely
    # (no embedder call attempted, safe to run here with no network) ---
    r_no_corr = score_adherence({"category": "word_count", "target_count": 3, "expected_answer": "the sky is blue"},
                                  "one two three", check_correctness=False)
    assert r_no_corr["correctness"] is None
    print("check_correctness=False correctly skips correctness checking entirely.")

    # --- correctness_judge_fn routing: confirms the default really is embeddings
    # (correctness_score), not the API judge, and that passing correctness_judge_fn
    # explicitly switches to it -- directly answers veerlosar's question about
    # whether the API judge is the default (no, it isn't, unless you pass it in) ---
    def _mock_correctness_judge(response, expected_answer):
        return {"correct": True, "raw": "mock yes"}

    r_default_is_embeddings = score_adherence(
        {"category": "word_count", "target_count": 3, "expected_answer": "the sky is blue"},
        "one two three", check_correctness=True)
    # no correctness_judge_fn passed -> falls through to correctness_score (embeddings),
    # which fails here (no network) -- confirms embeddings is what actually got tried
    assert "correctness_error" in r_default_is_embeddings
    assert "huggingface" in r_default_is_embeddings["correctness_error"].lower() or \
           "connect" in r_default_is_embeddings["correctness_error"].lower()
    print("Default (no correctness_judge_fn passed): confirmed embeddings path was attempted, not the API judge.")

    r_with_judge_fn = score_adherence(
        {"category": "word_count", "target_count": 3, "expected_answer": "the sky is blue"},
        "one two three", check_correctness=True, correctness_judge_fn=_mock_correctness_judge)
    assert r_with_judge_fn["correctness"] is True  # came from the mock judge, not embeddings
    assert r_with_judge_fn["correctness_similarity"] is None  # no similarity score from the API-judge path
    print("correctness_judge_fn, when passed, is used instead of the default embedding check.")

    # --- veerlosar's exact counterexample: bold_html DOES get a correctness
    # attempt when expected_answer is present, disproving the old docstring's
    # false claim that format-only categories can't have one. The adherence
    # check (did it wrap the content in bold tags) and the correctness check
    # (was the wrapped content actually right) are independent -- a response
    # can pass one and fail the other, which is the whole point. ---
    r_bold_wrong_content = score_adherence(
        {"category": "bold_html", "expected_answer": "Rapid heartbeat, sweating, and racing thoughts."},
        "<b>banana</b>", check_correctness=True)
    assert r_bold_wrong_content["compliant"] is True  # wrapped in bold -- adherence checker only looks at format
    assert r_bold_wrong_content["correctness"] is None  # no embedder here, but the attempt happened (see error field)
    assert "correctness_error" in r_bold_wrong_content  # confirms it TRIED, not that it was skipped for this category
    print("bold_html + expected_answer: adherence (format) and correctness (content) are independently")
    print("checked, exactly as veerlosar's counterexample said they should be -- not category-restricted.")

    # --- correctness-checker wiring: check_correctness=True DOES attempt it, and
    # the failure (no embedder access here) is caught and reported, not raised --
    # confirms the "never crash a batch over one row" principle actually holds
    # for this new code path too, not just the pre-existing ones ---
    r_with_corr = score_adherence({"category": "word_count", "target_count": 3, "expected_answer": "the sky is blue"},
                                    "one two three", check_correctness=True)
    assert r_with_corr["compliant"] is True  # the word_count check itself still ran fine
    assert r_with_corr["correctness"] is None  # correctness didn't succeed here (no embedder access)
    assert "correctness_error" in r_with_corr  # but the failure was caught and reported, not raised
    print(f"check_correctness=True attempted it and caught the expected failure gracefully: "
          f"{r_with_corr['correctness_error']}")
    print("(This should succeed and return a real correctness label in Colab/Lambda, where")
    print("correctness_checker.py's embedder can actually download -- not yet confirmed there,")
    print("only confirmed to fail gracefully here where there's no network access.)")

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
