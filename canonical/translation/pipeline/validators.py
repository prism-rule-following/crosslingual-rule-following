"""The checks a reviewer structurally cannot perform.
results are recorded as provenance and reported per language; the human decides."""

from __future__ import annotations

from collections import defaultdict
from typing import Any



def check_untranslated(entry: dict[str, Any]) -> str | None:
    """Empty output, or output identical to the English input.
    Returns a diagnostic string, or None if fine.
    """
    translated = entry.get("corrected") or entry.get("mt")
    if translated is None:
        return "not translated"
    if not translated.strip():
        return "empty translation"
    if translated.strip() == entry["en"].strip() and len(entry["en"]) > 3:
        return f"untranslated passthrough: {entry['en']!r}"
    return None


def check_status_invariant(row: dict, active_word: str, revoked_word: str) -> str | None:
    """The rule/non-rule pair must differ only where the status word was substituted.

    Checks every occurrence of the status word, not just the first. A short status word
    routinely appears inside an unrelated word earlier in the sentence — German "ein"
    sits inside "eine", and substituting there instead of at the status span would
    report a failure on a perfectly correct pair.
    """
    rule, non_rule = row["rule_text"], row["non_rule_text"]
    if rule == non_rule:
        return f"status words {active_word!r} and {revoked_word!r} produce identical text"

    span = len(active_word)
    for i in range(len(rule) - span + 1):
        if rule.startswith(active_word, i):
            if rule[:i] + revoked_word + rule[i + span:] == non_rule:
                return None

    return (
        f"status pair differs outside the status span "
        f"(no substitution of {active_word!r} -> {revoked_word!r} yields the non-rule text)"
    )


def grammar_triples(pairs: list[dict]) -> dict[tuple[str, str], dict[str, str]]:
    """Group the source clauses into their matched grammar triples.
    Keyed off (category, topic) — the source builds exactly one rule per cell in three
    grammar variants, so this is the grouping by construction. 

    Returns {(category, topic): {grammar_type: english_clause}}.
    """
    groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for pair in pairs:
        groups[(pair["category"], pair["topic"])][pair["grammar_type"]] = pair["rule_clause"]
    return dict(groups)


def check_grammar_collapse(
    pairs: list[dict], table: dict[str, dict]
) -> list[tuple[str, str, list[str]]]:
    """Find matched grammar triples that rendered identically in the target language.
    
    A language that genuinely cannot mark the three-way distinction is a finding about
    that language, not a translation defect — so collapses are reported, never rejected.
    Returns one (category, topic, colliding_grammar_types) tuple per collapsed triple.
    """
    from .compose import key_for

    collapsed = []
    for (category, topic), variants in grammar_triples(pairs).items():
        rendered: dict[str, str] = {}
        for grammar_type, clause in variants.items():
            entry = table.get(key_for(clause))
            if entry is None:
                continue
            text = entry["corrected"] or entry["mt"]
            if text:
                rendered[grammar_type] = text

        by_text: dict[str, list[str]] = defaultdict(list)
        for grammar_type, text in rendered.items():
            by_text[text].append(grammar_type)
        clashing = sorted(
            g for types in by_text.values() if len(types) > 1 for g in types
        )
        if clashing:
            collapsed.append((category, topic, clashing))
    return collapsed
