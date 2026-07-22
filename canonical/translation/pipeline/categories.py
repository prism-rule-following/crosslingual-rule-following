"""Per-category translation handlers.

Each handler takes the source English pair, a Translator instance, and the
language config, and returns a dict with three keys:
    rule_text, non_rule_text, checker

Handlers are dispatched from `translate_pair` in pipeline/translate.py via
the HANDLERS registry at the bottom of this file. Adding a new category =
add a function and register it — no other edits needed.

Every handler must preserve category-specific structural invariants:
    bold_html         — no literal <b>/<strong> tags in translated prose
    banned_word       — target-language checker matches the translated banned word
    include_word      — target-language checker matches the translated required word
    language          — required-answer-language checker unchanged; rule instructs in
                        row-language but asks for output in a different language
    word_count        — the number word appears in the translated rule; checker unchanged
    start_with        — translated start token appears in rule and in checker
    ack_invert        — translated acknowledgment + yes/no both appear in checker
    active_cancelled  — rule_text and non_rule_text differ by exactly one word
                        (the translated active/cancelled pair from config/translation.yaml file)
"""
from __future__ import annotations

import re
from typing import Callable

Handler = Callable[[dict, "object", dict], dict]


# Quoted-word extractor: handles straight, curly, and doubled quotes.
_QUOTED = re.compile(r"[\"'‘’“”]([^\"'‘’“”]+)[\"'‘’“”]")


def _first_quoted(text: str) -> str | None:
    """Extract the first quoted token, stripped of trailing punctuation.

    Rules often quote as 'Hello.' or 'Understood,' — the trailing punctuation is
    part of the quoted string in the source but is not part of the token we care
    about for translation or checker substitution.
    """
    m = _QUOTED.search(text)
    if not m:
        return None
    return m.group(1).rstrip(".,;:!?")


def _replace_first_word_in_checker(checker: str, source_word: str, target_word: str) -> str:
    """Substitute the first occurrence of source_word in a checker string.

    Handles two shapes that both appear in the source dataset:
      - `\\bWORD\\b` (regex word-boundary syntax in the string) — replace with
         `\\bTARGET\\b`.
      - Bare WORD in a checker with no word boundaries (e.g. `startswith('WORD')`).

    Uses plain string operations rather than regex to avoid confusion about
    backslash escaping inside the checker payload.
    """
    bounded = f"\\b{source_word}\\b"
    if bounded in checker:
        return checker.replace(bounded, f"\\b{target_word}\\b", 1)
    return checker.replace(source_word, target_word, 1)


# ----- Handlers -------------------------------------------------------------


def handle_bold_html(source: dict, translator, lang_config: dict) -> dict:
    """HTML tags are only *mentioned* in the rule prose, never as literal tags.

    Straight translation works — a structural check later verifies no literal
    tags snuck in. Checker regex is language-agnostic (matches <b>/<strong>).
    """
    return {
        "rule_text": translator.translate(source["rule_text"]),
        "non_rule_text": translator.translate(source["non_rule_text"]),
        "checker": source["checker"],
    }


def handle_banned_word(source: dict, translator, lang_config: dict) -> dict:
    """Extract the banned word, translate it, insert into rule, rewrite checker.

    Both the rule and the checker must reference the same translated word so
    that a model writing in the target language can be scored correctly.
    """
    banned = _first_quoted(source["rule_text"])
    if banned is None:
        # Fallback: translate normally, checker unchanged (structural check will flag).
        return {
            "rule_text": translator.translate(source["rule_text"]),
            "non_rule_text": translator.translate(source["non_rule_text"]),
            "checker": source["checker"],
        }
    translated_word = translator.translate(banned)
    rule_text = translator.translate(source["rule_text"].replace(banned, translated_word))
    non_rule_text = translator.translate(source["non_rule_text"].replace(banned, translated_word))
    checker = _replace_first_word_in_checker(source["checker"], banned, translated_word)
    return {"rule_text": rule_text, "non_rule_text": non_rule_text, "checker": checker}


def handle_include_word(source: dict, translator, lang_config: dict) -> dict:
    """Same pattern as banned_word — extract, translate, insert, rewrite checker."""
    required = _first_quoted(source["rule_text"])
    if required is None:
        return {
            "rule_text": translator.translate(source["rule_text"]),
            "non_rule_text": translator.translate(source["non_rule_text"]),
            "checker": source["checker"],
        }
    translated_word = translator.translate(required)
    rule_text = translator.translate(source["rule_text"].replace(required, translated_word))
    non_rule_text = translator.translate(source["non_rule_text"].replace(required, translated_word))
    checker = _replace_first_word_in_checker(source["checker"], required, translated_word)
    return {"rule_text": rule_text, "non_rule_text": non_rule_text, "checker": checker}


def handle_language(source: dict, translator, lang_config: dict) -> dict:
    """The answer-language named in the rule is orthogonal to the row's language.

    Example: a row translated into German still says "Write an answer in French"
    (in German prose). The langdetect checker keeps its original target code.
    """
    return {
        "rule_text": translator.translate(source["rule_text"]),
        "non_rule_text": translator.translate(source["non_rule_text"]),
        "checker": source["checker"],
    }


def handle_word_count(source: dict, translator, lang_config: dict) -> dict:
    """Straight translation — checker (`len(out.split()) == N`) is language-agnostic.

    Structural check later verifies the target-language number word is present.
    Note: languages with compounding/agglutination (DE, SW, TR) may produce
    different word counts for the same instruction; native reviewers judge whether
    to keep the checker as-is or adjust.
    """
    return {
        "rule_text": translator.translate(source["rule_text"]),
        "non_rule_text": translator.translate(source["non_rule_text"]),
        "checker": source["checker"],
    }


def handle_start_with(source: dict, translator, lang_config: dict) -> dict:
    """Extract the start token, translate it, rewrite both rule and checker."""
    start_token = _first_quoted(source["rule_text"])
    if start_token is None:
        return {
            "rule_text": translator.translate(source["rule_text"]),
            "non_rule_text": translator.translate(source["non_rule_text"]),
            "checker": source["checker"],
        }
    translated = translator.translate(start_token)
    rule_text = translator.translate(source["rule_text"].replace(start_token, translated))
    non_rule_text = translator.translate(source["non_rule_text"].replace(start_token, translated))
    # Checker uses a quoted lowercased literal — replace both the quoted form and any lowercased form.
    checker = source["checker"].replace(start_token, translated).replace(start_token.lower(), translated.lower())
    return {"rule_text": rule_text, "non_rule_text": non_rule_text, "checker": checker}


def handle_ack_invert(source: dict, translator, lang_config: dict) -> dict:
    """Two tokens matter: the acknowledgment ('Understood') and yes/no.

    All three are translated at runtime via the translator and
    substituted into both the rule and the checker regex. Not read from config
    because yes/no aren't the same semantic axis as active_cancelled_words.
    """
    ack_source = _first_quoted(source["rule_text"])
    ack_translated = translator.translate(ack_source) if ack_source else None

    yes_translated = translator.translate("yes")
    no_translated = translator.translate("no")

    rule_text = source["rule_text"]
    non_rule_text = source["non_rule_text"]
    checker = source["checker"]

    if ack_source and ack_translated:
        rule_text = rule_text.replace(ack_source, ack_translated)
        non_rule_text = non_rule_text.replace(ack_source, ack_translated)
        checker = checker.replace(ack_source, ack_translated).replace(ack_source.lower(), ack_translated.lower())

    rule_text = translator.translate(rule_text)
    non_rule_text = translator.translate(non_rule_text)

    # Swap the yes/no literals in the checker regex. Be conservative — only replace
    # whole-word matches to avoid mangling regex metacharacters.
    checker = re.sub(r"\byes\b", yes_translated, checker)
    checker = re.sub(r"\bno\b", no_translated, checker)

    return {"rule_text": rule_text, "non_rule_text": non_rule_text, "checker": checker}


_STATUS_RE = re.compile(r"^(.*?)\bRule status:\s*(active|cancelled)\.\s*$", re.IGNORECASE | re.DOTALL)


def handle_active_cancelled(source: dict, translator, lang_config: dict) -> dict:
    """The pair must remain identical except for one word (the status).

    Strategy: translate the shared prefix + 'Rule status:' label once, then
    append the target-language active/cancelled words from config. This
    guarantees a clean one-word diff regardless of how the translator would
    have rendered the full sentence.
    """
    words = lang_config.get("active_cancelled_words")
    if not words or "active" not in words or "cancelled" not in words:
        raise ValueError(
            f"language config missing active_cancelled_words for active_cancelled category"
        )
    active_word = words["active"]
    cancelled_word = words["cancelled"]

    match = _STATUS_RE.match(source["rule_text"])
    if not match:
        # Rule doesn't follow the expected form. Fall back to whole-text translation.
        return {
            "rule_text": translator.translate(source["rule_text"]),
            "non_rule_text": translator.translate(source["non_rule_text"]),
            "checker": source["checker"],
        }
    main_rule = match.group(1).strip()

    # Translate the rule body and the "Rule status:" label separately, then reassemble.
    translated_main = translator.translate(main_rule)
    translated_label = translator.translate("Rule status:")

    rule_text = f"{translated_main} {translated_label} {active_word}."
    non_rule_text = f"{translated_main} {translated_label} {cancelled_word}."

    # Checker for active_cancelled is typically a manual/LLM-judge string with no
    # regex reference to the specific words — leave it unchanged. Structural check
    # verifies the one-word-diff invariant.
    return {"rule_text": rule_text, "non_rule_text": non_rule_text, "checker": source["checker"]}


# Registry ------------------------------------------------------------------

HANDLERS: dict[str, Handler] = {
    "bold_html": handle_bold_html,
    "banned_word": handle_banned_word,
    "include_word": handle_include_word,
    "language": handle_language,
    "word_count": handle_word_count,
    "start_with": handle_start_with,
    "ack_invert": handle_ack_invert,
    "active_cancelled": handle_active_cancelled,
}
