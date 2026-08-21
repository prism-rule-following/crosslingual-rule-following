"""Decompose the English source into unique strings; recompose rows from translations."""


from __future__ import annotations

import hashlib
from typing import Any, Iterable

# Bare yes/no resolve from the per-language `yes_no_labels` config rather than MT.
LABEL_WORDS = frozenset({"yes", "no"})

# Structural fragments translated once per language and reused by every item.
TEMPLATE_STRINGS = {
    "rule_label": "Rule:",
    "status_label": "Rule status:",
}


DEFAULT_TEMPLATES = {
    "rule_text_tmpl": "{clause}. {status_label} {status_word}.",
    "system_tmpl": "{context} {rule_label} {rule_text}",
    "query_tmpl": "{prefix}{base}",
}

PRESSURE_WITH_PREFIX = ("L1", "L2", "L3", "L4")


def key_for(text: str) -> str:
    """Stable id for a source string. Keyed on text alone, so identical English always
    yields identical target text regardless of which field it came from."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def _base_queries(pairs: list[dict]) -> dict[tuple[str, str], str]:
    """L0 query per (category, topic) — the stem that L1-L4 prefix."""
    return {
        (p["category"], p["topic"]): p["user_query"]
        for p in pairs
        if p["pressure_level"] == "L0"
    }


def split_query(pair: dict, bases: dict[tuple[str, str], str]) -> tuple[str, str]:
    """Split a user_query into (prefix, base). Prefix is '' for L0 and L5.
    Raises ValueError if an L1-L4 query does not end with its cell's L0 query, so a failure means the
    source changed shape and the decomposition assumption needs revisiting.
    """
    query = pair["user_query"]
    if pair["pressure_level"] not in PRESSURE_WITH_PREFIX:
        return "", query
    base = bases[(pair["category"], pair["topic"])]
    if not query.endswith(base):
        raise ValueError(
            f"{pair['id']}: L1-L4 query does not end with its L0 base query; "
            "the prefix decomposition no longer holds for this source"
        )
    return query[: -len(base)], base


def decompose(pairs: list[dict]) -> dict[str, dict[str, Any]]:
    """Extract every distinct string that needs translating, keyed by sha256.

    Each entry carries the English text, a `kind` tag, and slots the translation step
    fills in (`mt`) and the review step overrides (`corrected`).
    """
    bases = _base_queries(pairs)
    table: dict[str, dict[str, Any]] = {}

    def add(text: str, kind: str) -> None:
        if not text:
            return
        entry = table.setdefault(
            key_for(text),
            {"en": text, "kind": kind, "mt": None, "corrected": None},
        )
        # A string reachable as both a label and something else stays a label, so it
        # keeps resolving from config rather than MT.
        if kind == "label":
            entry["kind"] = "label"

    for text in TEMPLATE_STRINGS.values():
        add(text, "template")

    for pair in pairs:
        add(pair["context"], "context")
        add(pair["rule_clause"], "clause")

        prefix, base = split_query(pair, bases)
        add(prefix, "prefix")
        add(base, "query")

        for turn in pair["user_turns"] or ():
            add(turn, "turn")

        answer = pair["correct_answer"]
        add(answer, "label" if answer in LABEL_WORDS else "answer")

        for word in pair["correct_keywords"]:
            add(word, "label" if word in LABEL_WORDS else "keyword")

    return table


def translatable(table: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Entries needing a translation call — everything except config-resolved labels."""
    return [(k, e) for k, e in table.items() if e["kind"] != "label"]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


class Resolver:
    """Looks up target-language text for a source string.

    Labels come from `yes_no_labels` in the language config; everything else comes from
    the string table, preferring a reviewer correction over the machine translation.
    """

    def __init__(self, table: dict[str, dict[str, Any]], lang_config: dict):
        self.table = table
        self.labels = {
            word: variants[0]
            for word, variants in (lang_config.get("yes_no_labels") or {}).items()
        }
        self.templates = {**DEFAULT_TEMPLATES, **(lang_config.get("templates") or {})}
        self.status_words = lang_config["status_words"]

    def text(self, source: str) -> str:
        if source in LABEL_WORDS:
            try:
                return self.labels[source]
            except KeyError:
                raise KeyError(f"language config has no yes_no_labels entry for {source!r}")
        entry = self.table.get(key_for(source))
        if entry is None:
            raise KeyError(f"no string-table entry for {source!r}")
        translated = entry["corrected"] or entry["mt"]
        if translated is None:
            raise KeyError(f"string not yet translated: {source!r}")
        return translated

    def status_word(self, pair_type: str, status: str) -> str:
        try:
            return self.status_words[pair_type][status]
        except KeyError:
            raise KeyError(
                f"language config missing status_words[{pair_type!r}][{status!r}]"
            )

    def source_of(self, source: str) -> str | None:
        """'human' if a reviewer corrected this string, 'mt' otherwise."""
        if source in LABEL_WORDS:
            return None
        entry = self.table.get(key_for(source))
        return "human" if entry and entry["corrected"] else "mt"


def build_row(pair: dict, resolver: Resolver, lang_code: str, bases: dict) -> dict:
    """Compose one target-language item.
    Fields are emitted by iterating the source row, so the output key set and key order
    match the source exactly and cannot drift.
    """
    t = resolver.text
    tmpl = resolver.templates

    clause = t(pair["rule_clause"])
    context = t(pair["context"])
    rule_label = t(TEMPLATE_STRINGS["rule_label"])
    status_label = t(TEMPLATE_STRINGS["status_label"])

    def rule_sentence(status: str) -> str:
        return tmpl["rule_text_tmpl"].format(
            clause=clause,
            status_label=status_label,
            status_word=resolver.status_word(pair["pair_type"], status),
        )

    rule_text = rule_sentence(pair["active_status"])
    non_rule_text = rule_sentence(pair["revoked_status"])

    def system(rule: str) -> str:
        return tmpl["system_tmpl"].format(
            context=context, rule_label=rule_label, rule_text=rule
        )

    prefix, base = split_query(pair, bases)
    user_query = (
        tmpl["query_tmpl"].format(prefix=t(prefix), base=t(base)) if prefix else t(base)
    )

    computed = {
        "language": lang_code,
        "context": context,
        "rule_clause": clause,
        "rule_text": rule_text,
        "non_rule_text": non_rule_text,
        "system_rule": system(rule_text),
        "system_non_rule": system(non_rule_text),
        "user_query": user_query,
        "user_turns": [t(x) for x in pair["user_turns"]] if pair["user_turns"] else None,
        "correct_answer": t(pair["correct_answer"]),
        "correct_keywords": [t(w) for w in pair["correct_keywords"]],
        "active_checker": _localise_checker(pair["active_checker"], resolver),
        "revoked_checker": _localise_checker(pair["revoked_checker"], resolver),
    }
    return {key: computed.get(key, pair[key]) for key in pair}


def _localise_checker(checker: dict, resolver: Resolver) -> dict:
    """Copy a checker verbatim, localising only the deterministic rubric's label values.

    Judge rubrics stay in English by design — they are culture-invariant and read by the
    judge, not matched against model output. The deterministic rubric is the opposite:
    its labels are string-matched against the model's own reply, so left in English every
    non-English answer would score as a violation.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in checker.items()}
    if checker.get("checker_type") != "deterministic_function":
        return out

    rubric = out["rubric"]
    for field in ("expected", "truth_label", "inverted_label"):
        if field in rubric:
            rubric[field] = resolver.text(rubric[field])
    if "labels" in rubric:
        rubric["labels"] = [resolver.text(label) for label in rubric["labels"]]
    return out


def build_rows(
    pairs: list[dict], table: dict, lang_config: dict, lang_code: str
) -> list[dict]:
    resolver = Resolver(table, lang_config)
    bases = _base_queries(pairs)
    return [build_row(p, resolver, lang_code, bases) for p in pairs]


# ---------------------------------------------------------------------------
# Review selection
# ---------------------------------------------------------------------------


def _row_contents(pair: dict, bases: dict) -> set[tuple[str, str]]:
    """Everything one item covers: its distinct strings plus each axis value it exhibits."""
    prefix, base = split_query(pair, bases)
    contents = {
        ("clause", pair["rule_clause"]),
        ("context", pair["context"]),
        ("answer", pair["correct_answer"]),
        ("query", base),
        ("status", pair["active_status"]),
        ("status", pair["revoked_status"]),
        ("grammar", pair["grammar_type"]),
        ("pressure", pair["pressure_level"]),
        ("pair_type", pair["pair_type"]),
        ("category", pair["category"]),
        ("topic", pair["topic"]),
    }
    if prefix:
        contents.add(("prefix", prefix))
    contents.update(("turn", x) for x in pair["user_turns"] or ())
    contents.update(("keyword", w) for w in pair["correct_keywords"])
    return contents


def covering_rows(pairs: list[dict]) -> list[str]:
    """Smallest set of item ids covering every distinct string and every axis value.

    Reviewers read whole items for context, but each item is *canonical* only for the
    strings this cover assigned to it, so no string is presented for editing twice — and
    once the set is reviewed, every one of the 2,340 items is composed entirely of
    reviewed text.
    """
    bases = _base_queries(pairs)
    contents = {p["id"]: _row_contents(p, bases) for p in pairs}
    uncovered: set[tuple[str, str]] = set().union(*contents.values())

    chosen: list[str] = []
    while uncovered:
        best_id = min(
            contents,
            key=lambda rid: (-len(contents[rid] & uncovered), rid),
        )
        gained = contents[best_id] & uncovered
        if not gained:
            break
        chosen.append(best_id)
        uncovered -= gained
    return sorted(chosen)


def canonical_strings(pairs: list[dict], chosen: Iterable[str]) -> dict[str, list[str]]:
    """Which English strings each review item owns — the fields a reviewer may edit there.
    """
    editable_kinds = {"clause", "context", "answer", "query", "prefix", "turn", "keyword"}
    bases = _base_queries(pairs)
    by_id = {p["id"]: p for p in pairs}
    seen: set[str] = set()
    owned: dict[str, list[str]] = {}
    for rid in sorted(chosen):
        fresh = []
        for kind, text in sorted(_row_contents(by_id[rid], bases)):
            if kind not in editable_kinds or text in LABEL_WORDS or text in seen:
                continue
            seen.add(text)
            fresh.append(text)
        owned[rid] = fresh
    return owned


EDITABLE_FIELDS = (
    "context", "rule_clause", "user_query", "user_turns",
    "correct_answer", "correct_keywords",
)


DERIVED_FIELDS = ("rule_text", "non_rule_text", "system_rule", "system_non_rule")


def corrections_from_row(
    source_pair: dict, composed: dict, returned: dict, bases: dict, resolver: "Resolver"
) -> tuple[dict[str, str], list[str]]:
    """Recover string-level corrections by diffing a reviewer-edited row.
    Every editable field maps to a known source string
    Returns ({english_source_string: corrected_text}, [problems]).
    """
    fixes: dict[str, str] = {}
    problems: list[str] = []

    for field, english in (("context", source_pair["context"]),
                           ("rule_clause", source_pair["rule_clause"]),
                           ("correct_answer", source_pair["correct_answer"])):
        new = returned.get(field)
        if new is not None and new != composed[field]:
            fixes[english] = new

    for field, english_list in (("user_turns", source_pair["user_turns"] or []),
                                ("correct_keywords", source_pair["correct_keywords"])):
        new_list = returned.get(field) or []
        old_list = composed[field] or []
        if len(new_list) != len(old_list):
            problems.append(
                f"{returned.get('id')}: {field} has {len(new_list)} entries, expected "
                f"{len(old_list)} — entries must not be added or removed"
            )
            continue
        for english, old, new in zip(english_list, old_list, new_list):
            if new != old:
                fixes[english] = new

    new_query = returned.get("user_query")
    if new_query is not None and new_query != composed["user_query"]:
        prefix_en, base_en = split_query(source_pair, bases)
        if not prefix_en:
            fixes[base_en] = new_query
        else:
            prefix_t, base_t = resolver.text(prefix_en), resolver.text(base_en)
            if new_query.startswith(prefix_t):
                fixes[base_en] = new_query[len(prefix_t):]
            elif new_query.endswith(base_t):
                fixes[prefix_en] = new_query[: -len(base_t)]
            else:
                problems.append(
                    f"{returned.get('id')}: user_query changed in both the pressure prefix "
                    "and the question — cannot tell which is which; correct them in "
                    "separate items, or edit only one part here"
                )

    return fixes, problems
