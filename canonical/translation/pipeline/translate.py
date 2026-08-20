"""CLI + per-language orchestration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from pipeline import compose, validators
from pipeline.client import DEFAULT_MODEL, Translator

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "translation.yaml"
DEFAULT_SOURCE = REPO_ROOT / "data" / "source" / "judgment_rules_expanded.verified.json"
STRINGS_DIR = REPO_ROOT / "data" / "strings"
TRANSLATED_DIR = REPO_ROOT / "data" / "translated"
REVIEW_DIR = REPO_ROOT / "data" / "review"
AUTHORED_DIR = REPO_ROOT / "data" / "authored"
CACHE_PATH = REPO_ROOT / ".cache" / "translations.sqlite"

STEM = "judgment_rules_expanded"

META_FIELDS = [
    "id", "language", "topic", "category", "grammar_type", "pressure_level",
    "pair_type", "translator_model", "translated_at", "verification",
    "grammar_collapse", "in_review_set",
]


class TranslationCache:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, isolation_level=None)

    def key_from(self, parts: Iterable[Any]) -> str:
        canonical = json.dumps(list(parts), sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value))


# ============================================================================
# Loading and persistence
# ============================================================================


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_source(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def strings_path(lang: str) -> Path:
    return STRINGS_DIR / f"{STEM}.{lang}.strings.json"


def load_table_provenance(lang: str) -> tuple[str | None, str | None]:
    """How an existing string table was produced"""
    path = strings_path(lang)
    if not path.exists():
        return None, None
    saved = json.loads(path.read_text(encoding="utf-8"))
    return saved.get("model"), saved.get("translated_at")


def load_string_table(lang: str, pairs: list[dict]) -> dict[str, dict]:
    """Fresh table from the source, with previously saved machine translations carried over.
    Keyed by hash of the English text, so a source edit adds a new entry rather than
    silently reusing a translation of different text.
    """
    table = compose.decompose(pairs)
    path = strings_path(lang)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))["strings"]
        for key, entry in table.items():
            if key in saved:
                entry["mt"] = saved[key].get("mt")
    return table


def apply_corrections(
    table: dict[str, dict], lang: str, pairs: list[dict], lang_config: dict
) -> tuple[int, dict]:
    """Overlay reviewer corrections from every file for this language in data/authored/.
    """
    paths = sorted(AUTHORED_DIR.glob(f"{STEM}.{lang}.*.json"))
    if not paths:
        return 0, {}

    templates: dict[str, str] = {}
    fixes: dict[str, str] = {}
    origin: dict[str, str] = {}
    conflicts: list[str] = []
    problems: list[str] = []

    # The rows as they stand before any correction — what the reviewer was looking at.
    # Every comparison below is against these, so a reviewer's own edit reads as a change
    # rather than being masked by having already been applied.
    bases = compose._base_queries(pairs)
    by_id = {p["id"]: p for p in pairs}
    before = {r["id"]: r for r in compose.build_rows(pairs, table, lang_config, lang)}

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        templates.update((data.get("glossary") or {}).get("templates") or data.get("templates") or {})

        resolver = compose.Resolver(table, lang_config)

        for returned in data.get("items", []):
            row_id = returned.get("id")
            if row_id not in by_id:
                problems.append(f"{path.name}: unknown item id {row_id!r} — ignored")
                continue
            row_fixes, row_problems = compose.corrections_from_row(
                by_id[row_id], before[row_id], returned, bases, resolver
            )
            problems.extend(f"{path.name}: {m}" for m in row_problems)
            for english, corrected in row_fixes.items():
                if english in fixes and fixes[english] != corrected and origin[english] != path.name:
                    conflicts.append(
                        f"{english[:45]!r}: {origin[english]} and {path.name} disagree "
                        f"— using {path.name}"
                    )
                elif english in fixes and fixes[english] != corrected:
                    conflicts.append(
                        f"{english[:45]!r}: corrected two different ways within "
                        f"{path.name} — using the later item"
                    )
                fixes[english] = corrected
                origin[english] = path.name

    applied = 0
    for english, corrected in fixes.items():
        key = compose.key_for(english)
        if key in table:
            table[key]["corrected"] = corrected
            applied += 1


    hand_edited = set()
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for returned in data.get("items", []):
            was = before.get(returned.get("id"))
            if was is None:
                continue
            if any(f in returned and returned[f] != was[f] for f in compose.EDITABLE_FIELDS):
                continue  # they edited a real field; derived staleness is expected
            for field in compose.DERIVED_FIELDS:
                if field in returned and returned[field] != was[field]:
                    hand_edited.add((returned["id"], field))

    print(f"  [{lang}] read corrections from: {', '.join(p.name for p in paths)}")
    for message in problems[:10]:
        print(f"  [{lang}] SKIPPED {message}")
    if len(problems) > 10:
        print(f"  [{lang}] ... and {len(problems) - 10} more skipped")
    for message in conflicts[:10]:
        print(f"  [{lang}] CONFLICT {message}")
    if len(conflicts) > 10:
        print(f"  [{lang}] ... and {len(conflicts) - 10} more conflicts")
    if hand_edited:
        fields = sorted({f for _, f in hand_edited})
        print(
            f"  [{lang}] IGNORED hand-edits to derived field(s) {', '.join(fields)} in "
            f"{len({i for i, _ in hand_edited})} item(s) — these are rebuilt from "
            "rule_clause, the status words and the templates; edit those instead"
        )
    return applied, templates


def save_string_table(
    lang: str, table: dict[str, dict], model: str | None = None, translated_at: str | None = None
) -> None:
    STRINGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "language": lang,
        "model": model or DEFAULT_MODEL,
        "translated_at": translated_at or datetime.now(timezone.utc).isoformat(),
        "count": len(table),
        "strings": table,
    }
    strings_path(lang).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================================
# Per-language run
# ============================================================================

def translate_strings(
    table: dict[str, dict], translator: Translator, limit: int | None, lang: str
) -> int:
    """Fill in `mt` for every entry that needs one. Returns how many were translated."""
    pending = [
        (key, entry)
        for key, entry in compose.translatable(table)
        if entry["mt"] is None and not entry["corrected"]
    ]
    if limit:
        pending = pending[:limit]
    for _, entry in tqdm(pending, desc=f"[{lang}] translate", disable=not pending):
        entry["mt"] = translator.translate(entry["en"])
    return len(pending)

def write_english_reference(source: dict) -> None:
    """The review rows in English, for reviewers to compare against.

    Filtered straight from the source rather than composed. `build_rows` over an identity
    table reproduces the source exactly (test_english_round_trip_is_exact), so filtering
    gives the same rows without a second composition path that could drift from it.
    """
    pairs = source["pairs"]
    review_ids = set(compose.covering_rows(pairs))
    items = [pair for pair in pairs if pair["id"] in review_ids]
    write_json(
        REVIEW_DIR / f"{STEM}.en.review.json",
        {"language": "en", "count": len(items), "items": items},
    )

def process_language(
    lang: str,
    config: dict,
    source: dict,
    limit_strings: int | None,
    compose_only: bool,
    retranslate: bool = False,
) -> None:
    if lang not in config:
        raise ValueError(f"language {lang!r} not in {CONFIG_PATH}")
    lang_config = config[lang]
    pairs = source["pairs"]

    table = load_string_table(lang, pairs)
    producer, produced_at = load_table_provenance(lang)
    if retranslate:
        # Discard existing machine translations so every string is produced again.
        # Reviewer corrections are untouched — they live in data/authored/ and are
        # re-applied on top of whatever the new translation produces.
        for entry in table.values():
            entry["mt"] = None
        producer, produced_at = None, None
    # Translate first, then overlay corrections. Corrections are recovered by diffing the
    # reviewer's rows against composed ones, which requires the strings to exist — so this
    # order is load-bearing, not stylistic.
    if not compose_only:
        cache = TranslationCache(CACHE_PATH)
        translator = Translator(lang_config["name"], cache=cache, lang_config=lang_config)
        n = translate_strings(table, translator, limit_strings, lang)
        if n:
            producer, produced_at = DEFAULT_MODEL, datetime.now(timezone.utc).isoformat()
        else:
            print(f"  [{lang}] nothing to translate — every string already has one")
    save_string_table(lang, table, producer, produced_at)

    corrected_count, template_overrides = apply_corrections(table, lang, pairs, lang_config)
    if template_overrides:
        lang_config = {**lang_config, "templates": template_overrides}

    problems = {
        key: diagnostic
        for key, entry in table.items()
        if entry["kind"] != "label" and (diagnostic := validators.check_untranslated(entry))
    }
    if limit_strings or compose_only:
        # A partial run leaves most strings untranslated by design.
        problems = {k: v for k, v in problems.items() if v != "not translated"}
    for key, diagnostic in list(problems.items())[:10]:
        print(f"  [{lang}] {table[key]['en'][:50]!r}: {diagnostic}")

    untranslated = [k for k, e in table.items()
                    if e["kind"] != "label" and not (e["corrected"] or e["mt"])]
    if untranslated:
        print(
            f"[{lang}] {len(untranslated)} strings not yet translated — "
            "skipping composition (rerun without --limit-strings)"
        )
        return

    collapses = validators.check_grammar_collapse(pairs, table)
    collapsed_cells = {(c, t) for c, t, _ in collapses}

    rows = compose.build_rows(pairs, table, lang_config, lang)

    resolver = compose.Resolver(table, lang_config)
    bases = compose._base_queries(pairs)
    invariant_failures = []
    for pair, row in zip(pairs, rows):
        problem = validators.check_status_invariant(
            row,
            resolver.status_word(pair["pair_type"], pair["active_status"]),
            resolver.status_word(pair["pair_type"], pair["revoked_status"]),
        )
        if problem:
            invariant_failures.append((row["id"], problem))

    review_ids = set(compose.covering_rows(pairs))
    review_rows = [row for row in rows if row["id"] in review_ids]

    # The metadata block is the source's, unchanged — the translated file carries no keys
    # the English file does not have. 
    metadata = source["metadata"]

    write_json(TRANSLATED_DIR / f"{STEM}.{lang}.json", {"metadata": metadata, "pairs": rows})
    write_json(
        REVIEW_DIR / f"{STEM}.{lang}.review.json",
        {
            "language": lang,
            "count": len(review_rows),
            "items": review_rows,
        },
    )

    now = datetime.now(timezone.utc).isoformat()

    print(
        f"[{lang}] {len(rows)} items, {len(compose.translatable(table))} strings "
        f"({corrected_count} human-corrected), {len(review_rows)} review items"
    )
    if collapses:
        print(f"[{lang}] grammar collapse in {len(collapses)} cells:")
        for category, topic, types in collapses:
            print(f"    {category}/{topic}: {' == '.join(types)}")
    if invariant_failures:
        print(f"[{lang}] STATUS INVARIANT FAILED on {len(invariant_failures)} items:")
        for row_id, problem in invariant_failures[:5]:
            print(f"    {row_id}: {problem}")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", help="Target language code (e.g. de).")
    parser.add_argument("--all", action="store_true", help="Process every configured language.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source dataset JSON.")
    parser.add_argument("--limit-strings", type=int, default=None,
                        help="Cap strings translated per language (smoke tests).")
    parser.add_argument("--compose-only", action="store_true",
                        help="Rebuild items from saved strings and corrections; no API calls.")
    parser.add_argument("--retranslate", action="store_true",
                        help="Discard existing translations and translate every string again "
                             "(costs API calls). Reviewer corrections are preserved.")
    args = parser.parse_args()

    if args.retranslate and args.limit_strings:
        parser.error(
            "--retranslate with --limit-strings would clear every translation and refill "
            "only some, saving a mostly-empty table. Run --retranslate on its own."
        )
    if args.retranslate and args.compose_only:
        parser.error("--retranslate needs API calls; it cannot be combined with --compose-only")

    config = load_config()
    source = load_source(args.source)

    if args.all:
        languages = list(config)
    elif args.lang:
        languages = [args.lang]
    else:
        parser.error("must pass --lang <code> or --all")

    try:
        for lang in languages:
            process_language(lang, config, source, args.limit_strings,
                             args.compose_only, args.retranslate)
        write_english_reference(source)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
