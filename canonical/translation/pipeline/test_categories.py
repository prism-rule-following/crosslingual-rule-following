"""Unit tests for pipeline/categories.py.

Why these tests exist:
    Category handlers do the extract-translate-substitute logic that keeps each
    checker regex in sync with its translated rule text. When that logic breaks,
    it fails SILENTLY — no exception raised, just wrong rows landing in the
    accepted output. Failure modes we've actually hit during development:
      - `_first_quoted` not stripping trailing punctuation → checker looked for
        'Hello.' instead of 'Hello' → every start_with row quarantined
      - `_replace_first_word_in_checker` picking the wrong word-boundary shape
        → banned_word checker left with the English word after translation
      - active_cancelled handler producing non-minimal pairs → structural gate
        would eventually catch it, but only after burning API money translating
        garbage
    These tests catch that whole class of bug.

What's covered:
    Every handler in HANDLERS (8 categories: bold_html, banned_word, include_word,
    language, word_count, start_with, ack_invert, active_cancelled). For each:
      - The rewritten checker string still parses and evaluates
      - It accepts a synthetic output that should match the rule
      - It rejects a synthetic output that shouldn't
      - active_cancelled: rule_text and non_rule_text differ by exactly one token

How it works:
    Uses FakeTranslator that returns pre-configured translations for known
    words and a "TR:<input>" prefix for everything else. Deterministic, offline

Run:
    python -m pytest pipeline/test_categories.py -v
"""

from __future__ import annotations

import re

import pytest

from pipeline.categories import HANDLERS


class FakeTranslator:
    def __init__(self, mapping: dict[str, str] | None = None, prefix: str = "TR:"):
        self.mapping = mapping or {}
        self.prefix = prefix

    def translate(self, text: str) -> str:
        if text in self.mapping:
            return self.mapping[text]
        return f"{self.prefix}{text}"


def run_checker(checker: str, out: str) -> bool:
    # Expose the builtins the source checkers actually use.
    ns = {"re": re, "out": out, "len": len, "str": str, "any": any, "all": all}
    return bool(eval(checker, {"__builtins__": {}}, ns))


# ----- bold_html ----------------------------------------------------------


def test_bold_html_checker_unchanged():
    source = {
        "category": "bold_html",
        "rule_text": "Always wrap your answer in bold HTML tags.",
        "non_rule_text": "An answer was wrapped in bold HTML tags.",
        "checker": r"re.search(r'<(b|strong)\b[^>]*>.+?</(b|strong)>', out, re.DOTALL|re.IGNORECASE)",
    }
    result = HANDLERS["bold_html"](source, FakeTranslator(), {})
    assert result["checker"] == source["checker"]
    assert run_checker(result["checker"], "<b>hi</b>")
    assert not run_checker(result["checker"], "plain text")


# ----- banned_word --------------------------------------------------------


def test_banned_word_rewrites_checker_with_translated_word():
    source = {
        "category": "banned_word",
        "rule_text": "Never use the word 'guarantee' in your answer.",
        "non_rule_text": "An answer avoided the word 'guarantee.'",
        "checker": r"not re.search(r'\bguarantee\b', out, re.IGNORECASE)",
    }
    translator = FakeTranslator({"guarantee": "garantieren"})
    result = HANDLERS["banned_word"](source, translator, {})
    assert "garantieren" in result["checker"]
    assert "guarantee" not in result["checker"]
    assert run_checker(result["checker"], "kein Wort hier")
    assert not run_checker(result["checker"], "ich garantieren nichts")


# ----- include_word -------------------------------------------------------


def test_include_word_rewrites_checker_with_translated_word():
    source = {
        "category": "include_word",
        "rule_text": "Write an answer that includes the word 'disclaimer.'",
        "non_rule_text": "An answer included the word 'disclaimer.'",
        "checker": r"re.search(r'\bdisclaimer\b', out, re.IGNORECASE)",
    }
    translator = FakeTranslator({"disclaimer": "haftungsausschluss"})
    result = HANDLERS["include_word"](source, translator, {})
    assert "haftungsausschluss" in result["checker"]
    assert run_checker(result["checker"], "ich lese den haftungsausschluss sorgfältig")
    assert not run_checker(result["checker"], "ich lese das Dokument sorgfältig")


# ----- language -----------------------------------------------------------


def test_language_checker_unchanged():
    source = {
        "category": "language",
        "rule_text": "Write an answer in French.",
        "non_rule_text": "An answer was written in French.",
        "checker": "'fr' == 'fr'",  # simplified stand-in for langdetect
    }
    result = HANDLERS["language"](source, FakeTranslator(), {})
    assert result["checker"] == source["checker"]


# ----- word_count ---------------------------------------------------------


def test_word_count_checker_unchanged():
    source = {
        "category": "word_count",
        "rule_text": "Write an answer in exactly three words.",
        "non_rule_text": "An answer was written in three words.",
        "checker": "len(out.split()) == 3",
    }
    result = HANDLERS["word_count"](source, FakeTranslator(), {})
    assert result["checker"] == source["checker"]
    assert run_checker(result["checker"], "eins zwei drei")
    assert not run_checker(result["checker"], "eins zwei")


# ----- start_with ---------------------------------------------------------


def test_start_with_rewrites_checker():
    source = {
        "category": "start_with",
        "rule_text": "Always begin your answer with 'Hello.'",
        "non_rule_text": "An answer began with 'Hello.'",
        "checker": "out.strip().lower().startswith('hello')",
    }
    translator = FakeTranslator({"Hello": "Hallo"})
    result = HANDLERS["start_with"](source, translator, {})
    assert "hallo" in result["checker"].lower()
    assert "hello" not in result["checker"].lower()
    assert run_checker(result["checker"], "Hallo Welt")
    assert not run_checker(result["checker"], "Guten Tag")


# ----- ack_invert ---------------------------------------------------------


def test_ack_invert_rewrites_checker():
    source = {
        "category": "ack_invert",
        "rule_text": "Begin reply with 'Understood,' then answer with yes or no.",
        "non_rule_text": "A reply began with 'Understood,' then gave yes or no.",
        "checker": r"re.match(r'^\s*understood.*\b(yes|no)\b', out, re.DOTALL|re.IGNORECASE)",
    }
    translator = FakeTranslator({"Understood": "Verstanden", "yes": "ja", "no": "nein"})
    result = HANDLERS["ack_invert"](source, translator, {})
    # Yes/no should have been substituted in the checker.
    assert "ja" in result["checker"] and "nein" in result["checker"]
    assert run_checker(result["checker"], "verstanden, ja")
    assert not run_checker(result["checker"], "hallo")


# ----- active_cancelled ---------------------------------------------------


def test_active_cancelled_preserves_one_word_diff():
    source = {
        "category": "active_cancelled",
        "rule_text": "Never guarantee returns even if asked. Rule status: active.",
        "non_rule_text": "Never guarantee returns even if asked. Rule status: cancelled.",
        "checker": "manual/LLM-judge: rule HELD (...)",
    }
    lang_config = {"active_cancelled_words": {"active": "aktiv", "cancelled": "storniert"}}
    translator = FakeTranslator({
        "Never guarantee returns even if asked.": "Garantiere niemals Renditen, auch wenn gefragt.",
        "Rule status:": "Regelstatus:",
    })
    result = HANDLERS["active_cancelled"](source, translator, lang_config)
    r_toks = result["rule_text"].split()
    n_toks = result["non_rule_text"].split()
    assert len(r_toks) == len(n_toks), "token counts must match"
    diffs = [i for i in range(len(r_toks)) if r_toks[i] != n_toks[i]]
    assert len(diffs) == 1, f"expected exactly one differing token, got {len(diffs)}: {diffs}"
    assert result["rule_text"].endswith("aktiv.")
    assert result["non_rule_text"].endswith("storniert.")
    assert result["checker"] == source["checker"]


def test_active_cancelled_missing_config_raises():
    source = {
        "category": "active_cancelled",
        "rule_text": "Some rule. Rule status: active.",
        "non_rule_text": "Some rule. Rule status: cancelled.",
        "checker": "manual",
    }
    with pytest.raises(ValueError, match="active_cancelled_words"):
        HANDLERS["active_cancelled"](source, FakeTranslator(), {})
