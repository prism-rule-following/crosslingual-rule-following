"""
This file does 3 things:
    - CLI + per-language processing loop
    - `TranslationCache` (SQLite-backed key/value store, dedupes MT calls across runs)
    - `translate_pair` (per-field policy — see the section header below)

Usage:
    python -m pipeline.translate --lang de
    python -m pipeline.translate --lang de --limit 5       # smoke test
    python -m pipeline.translate --all
    python -m pipeline.translate --lang de --recalibrate   # recompute quality threshold

Outputs (relative to repo root):
    data/translated/rb_attrpatch_dataset.<lang>.json             — accepted rows (native language)
    data/translate_test/rb_attrpatch_dataset.<lang>.json         — translate-test artifact (back-translated to English)
    data/quarantine/rb_attrpatch_dataset.<lang>.quarantine.json  — rows that failed the QA gate
    data/review/rb_attrpatch_dataset.<lang>.sample.json          — stratified 10-15% sample of accepted rows for native review
    data/translation_meta.csv                                     — provenance for every attempted row (all languages)

If data/authored/rb_attrpatch_dataset.<lang>.authored.json exists, those rows
override any auto-translated pair with the same id (native-speaker completions).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from pipeline import validators
from pipeline.categories import HANDLERS
from pipeline.client import Translator

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "translation.yaml"
SOURCE_PATH = REPO_ROOT / "data" / "source" / "rb_attrpatch_dataset.json"
TRANSLATED_DIR = REPO_ROOT / "data" / "translated"
TRANSLATE_TEST_DIR = REPO_ROOT / "data" / "translate_test"
QUARANTINE_DIR = REPO_ROOT / "data" / "quarantine"
AUTHORED_DIR = REPO_ROOT / "data" / "authored"
REVIEW_DIR = REPO_ROOT / "data" / "review"
META_PATH = REPO_ROOT / "data" / "translation_meta.csv"
CACHE_PATH = REPO_ROOT / ".cache" / "translations.sqlite"

CALIBRATION_N = 5
REVIEW_FRACTION = 0.12
META_FIELDS = [
    "id", "en_pair_id", "language", "category", "translator_model",
    "quality_metric", "quality_score", "cosine",
    "lid_ok", "structural_ok", "checker_sanity_ok",
    "translated_at", "passed", "diagnostics",
    "in_review_sample", "authored_by",
]


# ============================================================================
# Persistent translation cache
# ============================================================================
#
# Keys hash every input that could change the output (backend, model, direction,
# prompt template version, text). Re-runs on the same inputs cost nothing;
# changing any input forces re-translation — the reproducibility guarantee.


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
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
            )


# ============================================================================
# Per-field translation policy
# ============================================================================
#
# Each pair has 12 fields; not all translate. See categories.py for the
# rule_text / non_rule_text / checker logic — this function just orchestrates
# the "simple" fields and delegates the category-specific work.
#
# Field policy:
#   id                          → rewritten with `_<lang>` suffix
#   en_pair_id                  → emitted as the original English id (join key)
#   category / topic /
#     grammar_type              → kept as-is
#   language                    → overwritten with target language code
#   context / user_query        → straight translation
#   rule_text / non_rule_text
#     / checker                 → delegated to category handler (each category
#                                 has different structural constraints)
#   system_rule / system_non_rule
#                               → reconstructed from context + rule/non-rule
#                                 to avoid drift between them


def translate_pair(
    source_pair: dict[str, Any],
    translator: Translator,
    lang_code: str,
    lang_config: dict,
) -> dict[str, Any]:
    """Translate one source pair into the target language.

    Raises KeyError if the category has no handler.
    """
    category = source_pair["category"]
    if category not in HANDLERS:
        raise KeyError(f"no handler registered for category {category!r}")

    result: dict[str, Any] = {
        "id": f"{source_pair['id']}_{lang_code}",
        "en_pair_id": source_pair["id"],
        "category": category,
        "topic": source_pair["topic"],
        "grammar_type": source_pair["grammar_type"],
        "language": lang_code,
    }

    result["context"] = translator.translate(source_pair["context"])
    result["user_query"] = translator.translate(source_pair["user_query"])

    category_out = HANDLERS[category](source_pair, translator, lang_config)
    result["rule_text"] = category_out["rule_text"]
    result["non_rule_text"] = category_out["non_rule_text"]
    result["checker"] = category_out["checker"]

    result["system_rule"] = f"{result['context']} {result['rule_text']}"
    result["system_non_rule"] = f"{result['context']} {result['non_rule_text']}"

    return result


# ============================================================================
# Config + source loading
# ============================================================================


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_source_pairs() -> list[dict]:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))["pairs"]


# ============================================================================
# Translator + backtranslator construction
# ============================================================================


def build_translators(lang_config: dict, lang_name: str, cache: TranslationCache):
    fwd = Translator(
        spec=lang_config["translator"],
        direction="forward",
        target_language_name=lang_name,
        source_language_name="English",
        cache=cache,
    )
    bwd = Translator(
        spec=lang_config["backtranslator"],
        direction="backward",
        target_language_name="English",
        source_language_name=lang_name,
        cache=cache,
    )
    return fwd, bwd


# ============================================================================
# Quality-threshold calibration (per-language, from first N pairs)
# ============================================================================


def calibrate_or_reuse(
    lang_code: str,
    lang_config: dict,
    source_pairs: list[dict],
    fwd: Translator,
    recalibrate: bool,
) -> float:
    stored = lang_config.get("quality_threshold")
    if stored is not None and not recalibrate:
        return float(stored)

    print(f"[{lang_code}] calibrating quality threshold on first {CALIBRATION_N} pairs...")
    scores: list[float] = []
    for pair in source_pairs[:CALIBRATION_N]:
        try:
            translated = translate_pair(pair, fwd, lang_code, lang_config)
            score = validators.score_comet(
                pair["rule_text"], translated["rule_text"], lang_config["quality_metric"]
            )
            scores.append(score)
        except Exception as e:
            print(f"  {pair['id']}: skipped ({e})")
    threshold = validators.calibrate_threshold(scores)
    print(
        f"[{lang_code}] threshold = {threshold:.3f} "
        f"(n={len(scores)}, min={min(scores, default=0):.3f}, max={max(scores, default=0):.3f})"
    )
    return threshold


# ============================================================================
# Authored-row overlay + review sample + translate-test artifact
# ============================================================================


def load_authored(lang_code: str) -> dict[str, dict]:
    path = AUTHORED_DIR / f"rb_attrpatch_dataset.{lang_code}.authored.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: row for row in data}


def stratified_review_sample(pairs: list[dict], fraction: float) -> list[dict]:
    """Deterministic stratified sample across categories, for native-reviewer cross-check."""
    by_category: dict[str, list[dict]] = {}
    for p in pairs:
        by_category.setdefault(p["category"], []).append(p)
    rng = random.Random(42)
    sample: list[dict] = []
    for rows in by_category.values():
        k = max(1, int(round(len(rows) * fraction)))
        sample.extend(rng.sample(rows, min(k, len(rows))))
    return sample


def build_translate_test_pair(
    translated: dict,
    en_source_by_id: dict[str, dict],
    bwd: Translator,
    lang_code: str,
) -> dict:
    """Back-translate an accepted target-language pair into English (IrokoBench-style artifact).

    Downstream evaluation computes `native_adherence − translate_test_adherence` per
    language to separate concept failure from language-processing failure.
    """
    en_source = en_source_by_id.get(translated["en_pair_id"], {})
    context_en = bwd.translate(translated["context"])
    rule_en = bwd.translate(translated["rule_text"])
    non_rule_en = bwd.translate(translated["non_rule_text"])
    user_query_en = bwd.translate(translated["user_query"])
    return {
        "id": f"{translated['en_pair_id']}_{lang_code}_bt",
        "en_pair_id": translated["en_pair_id"],
        "category": translated["category"],
        "topic": translated["topic"],
        "grammar_type": translated["grammar_type"],
        "language": f"{lang_code}_en",
        "context": context_en,
        "rule_text": rule_en,
        "non_rule_text": non_rule_en,
        "system_rule": f"{context_en} {rule_en}",
        "system_non_rule": f"{context_en} {non_rule_en}",
        "user_query": user_query_en,
        # Use the original English checker — this artifact is evaluated in English.
        "checker": en_source.get("checker", translated["checker"]),
    }


def append_meta(rows: list[dict]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not META_PATH.exists()
    with META_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in META_FIELDS})


# ============================================================================
# Per-language processing loop
# ============================================================================


def process_language(
    lang_code: str,
    config: dict,
    source_pairs: list[dict],
    limit: int | None,
    recalibrate: bool,
) -> None:
    if lang_code not in config:
        raise ValueError(f"language {lang_code!r} not in {CONFIG_PATH}")
    lang_config = config[lang_code]
    lang_name = lang_config["name"]

    cache = TranslationCache(CACHE_PATH)
    fwd, bwd = build_translators(lang_config, lang_name, cache)

    threshold = calibrate_or_reuse(lang_code, lang_config, source_pairs, fwd, recalibrate)

    authored = load_authored(lang_code)
    en_source_by_id = {p["id"]: p for p in source_pairs}
    pairs_to_process = source_pairs[:limit] if limit else source_pairs

    accepted: list[dict] = []
    quarantined: list[dict] = []
    translate_test: list[dict] = []
    meta_rows: list[dict] = []

    for source_pair in tqdm(pairs_to_process, desc=f"[{lang_code}] translate"):
        suffixed_id = f"{source_pair['id']}_{lang_code}"

        if suffixed_id in authored:
            authored_pair = authored[suffixed_id]
            accepted.append(authored_pair)
            translate_test.append(
                build_translate_test_pair(authored_pair, en_source_by_id, bwd, lang_code)
            )
            meta_rows.append({
                "id": suffixed_id, "en_pair_id": source_pair["id"], "language": lang_code,
                "category": source_pair["category"], "translator_model": "human",
                "translated_at": datetime.now(timezone.utc).isoformat(),
                "passed": True, "authored_by": authored_pair.get("authored_by", "unknown"),
            })
            continue

        try:
            translated = translate_pair(source_pair, fwd, lang_code, lang_config)
        except Exception as e:
            quarantined.append({**source_pair, "diagnostics": f"translate error: {e}"})
            meta_rows.append({
                "id": suffixed_id, "en_pair_id": source_pair["id"], "language": lang_code,
                "category": source_pair["category"],
                "translator_model": lang_config["translator"]["model"],
                "translated_at": datetime.now(timezone.utc).isoformat(),
                "passed": False, "diagnostics": f"translate error: {e}",
            })
            continue

        result = validators.evaluate_pair(
            source_pair=source_pair, translated_pair=translated,
            lang_code=lang_code, lang_config=lang_config,
            quality_threshold=threshold, backtranslator=bwd,
        )
        row_meta = {
            "id": translated["id"], "en_pair_id": source_pair["id"], "language": lang_code,
            "category": source_pair["category"],
            "translator_model": lang_config["translator"]["model"],
            "quality_metric": lang_config["quality_metric"],
            "quality_score": f"{result['quality_score']:.4f}",
            "cosine": f"{result['cosine']:.4f}",
            "lid_ok": result["lid_ok"], "structural_ok": result["structural_ok"],
            "checker_sanity_ok": result["checker_sanity_ok"],
            "translated_at": datetime.now(timezone.utc).isoformat(),
            "passed": result["passed"], "diagnostics": result["diagnostics"],
        }

        if result["passed"]:
            accepted.append(translated)
            translate_test.append(
                build_translate_test_pair(translated, en_source_by_id, bwd, lang_code)
            )
        else:
            quarantined.append({**translated, "diagnostics": result["diagnostics"]})
        meta_rows.append(row_meta)

    # Write outputs
    TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)
    TRANSLATE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    main_path = TRANSLATED_DIR / f"rb_attrpatch_dataset.{lang_code}.json"
    main_path.write_text(
        json.dumps(
            {"metadata": {"language": lang_code, "count": len(accepted)}, "pairs": accepted},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    tt_path = TRANSLATE_TEST_DIR / f"rb_attrpatch_dataset.{lang_code}.json"
    tt_path.write_text(
        json.dumps(
            {"metadata": {"language": f"{lang_code}_en", "count": len(translate_test)}, "pairs": translate_test},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    q_path = QUARANTINE_DIR / f"rb_attrpatch_dataset.{lang_code}.quarantine.json"
    q_path.write_text(
        json.dumps({"count": len(quarantined), "pairs": quarantined}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    review = stratified_review_sample(accepted, REVIEW_FRACTION)
    review_ids = {r["id"] for r in review}
    for row in meta_rows:
        row["in_review_sample"] = row["id"] in review_ids
    review_path = REVIEW_DIR / f"rb_attrpatch_dataset.{lang_code}.sample.json"
    review_path.write_text(
        json.dumps({"count": len(review), "pairs": review}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    append_meta(meta_rows)

    print(
        f"[{lang_code}] done: {len(accepted)} accepted, {len(quarantined)} quarantined, "
        f"{len(review)} in review sample, {len(translate_test)} translate-test rows"
    )


# ============================================================================
# CLI entry point
# ============================================================================


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", type=str, help="Target language code (e.g. de).")
    parser.add_argument("--all", action="store_true", help="Process every language in the config.")
    parser.add_argument("--limit", type=int, default=None, help="Cap pairs processed per language (smoke tests).")
    parser.add_argument("--recalibrate", action="store_true", help="Ignore stored threshold and recompute.")
    args = parser.parse_args()

    config = load_config()
    source_pairs = load_source_pairs()

    if args.all:
        for lang_code in config:
            process_language(lang_code, config, source_pairs, args.limit, args.recalibrate)
    elif args.lang:
        process_language(args.lang, config, source_pairs, args.limit, args.recalibrate)
    else:
        parser.error("must pass --lang <code> or --all")


if __name__ == "__main__":
    main()
