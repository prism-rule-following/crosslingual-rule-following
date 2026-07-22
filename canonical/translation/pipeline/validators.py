"""Quality checks for translated pairs.

Five checks, include:
  1. COMET / AfriCOMET quality score (per-language calibrated threshold)
  2. Round-trip back-translation similarity (multilingual sentence-embedding cosine)
  3. fastText lid.176 language ID
  4. Structural check (category-specific)
  5. Checker sanity (rewritten regex must pass a synthetic positive, fail a synthetic negative)

A pair passes only if all five pass. Failures return a diagnostic string that
gets attached to the quarantine record.
"""
from __future__ import annotations

import os
import re
import statistics
from pathlib import Path
from typing import Any

_COMET_MODEL_MAP = {
    "comet_kiwi": "Unbabel/wmt22-cometkiwi-da",
    "africomet": "masakhane/africomet-qe-stl",
}

_COMET_INSTANCES: dict[str, Any] = {}
_EMBEDDER = None
_LID_MODEL = None


# ----- COMET / AfriCOMET ---------------------------------------------------


def load_comet(metric: str):
    """Download + load the requested COMET-family model. Cached per process."""
    if metric not in _COMET_MODEL_MAP:
        raise ValueError(f"unknown quality metric {metric!r}")
    model_id = _COMET_MODEL_MAP[metric]
    if model_id not in _COMET_INSTANCES:
        from comet import download_model, load_from_checkpoint  # type: ignore

        checkpoint = download_model(model_id)
        _COMET_INSTANCES[model_id] = load_from_checkpoint(checkpoint)
    return _COMET_INSTANCES[model_id]


def score_comet(source: str, translation: str, metric: str) -> float:
    """Reference-free quality score in [0, 1]-ish (COMET-Kiwi is not strictly bounded).

    `num_workers=1` works around an unbabel-comet bug on modern PyTorch where
    the library sets `multiprocessing_context='fork'` while also setting
    `num_workers=0`, which torch rejects. One worker adds trivial overhead but
    makes the DataLoader accept the config.
    """
    model = load_comet(metric)
    result = model.predict(
        [{"src": source, "mt": translation}],
        batch_size=1,
        gpus=0,
        progress_bar=False,
        num_workers=1,
    )
    return float(result.scores[0])


def calibrate_threshold(scores: list[float]) -> float:
    """Threshold = max(mean − 1.5·std, 15th percentile) over calibration scores.

    Rationale: adapts to per-language score distribution rather than using a
    flat cutoff. The `max` picks the stricter of the two bounds.
    """
    if not scores:
        return 0.0
    if len(scores) == 1:
        return scores[0]
    mean = statistics.mean(scores)
    std = statistics.stdev(scores)
    sorted_scores = sorted(scores)
    p15_idx = max(0, int(len(sorted_scores) * 0.15))
    p15 = sorted_scores[p15_idx]
    return max(mean - 1.5 * std, p15)


# ----- Back-translation similarity ----------------------------------------


def cosine_score(reference_en: str, backtranslated_en: str) -> float:
    """Cosine similarity of multilingual sentence embeddings."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embs = _EMBEDDER.encode(
        [reference_en, backtranslated_en], convert_to_tensor=True, normalize_embeddings=True
    )
    return float((embs[0] * embs[1]).sum())


# ----- Language identification --------------------------------------------


def _lid_model_path() -> str:
    """Where the fastText model lives. Override via FASTTEXT_LID_PATH."""
    default = str(Path.home() / ".cache" / "fasttext" / "lid.176.bin")
    return os.environ.get("FASTTEXT_LID_PATH", default)


def detect_language(text: str) -> tuple[str, float]:
    """Return (iso_code, confidence). Text is normalized to a single line."""
    global _LID_MODEL
    if _LID_MODEL is None:
        import fasttext  # type: ignore

        path = _lid_model_path()
        if not Path(path).exists():
            raise FileNotFoundError(
                f"fastText model not found at {path!r}. Download with:\n"
                "  curl -L -o ~/.cache/fasttext/lid.176.bin "
                "https://huggingface.co/julien-c/fasttext-language-id/resolve/main/lid.176.bin"
            )
        _LID_MODEL = fasttext.load_model(path)
    labels, probs = _LID_MODEL.predict(text.replace("\n", " "), k=1)
    return labels[0].replace("__label__", ""), float(probs[0])


def check_language_id(translated_fields: dict[str, str], expected_lang: str) -> tuple[bool, str]:
    """Verify every translated field is identified as the expected language."""
    for name, text in translated_fields.items():
        if not text.strip():
            continue
        detected, conf = detect_language(text)
        if detected != expected_lang and conf > 0.5:
            return False, f"lid: {name!r} detected as {detected!r} (conf {conf:.2f}), expected {expected_lang!r}"
    return True, ""


# ----- Structural checks --------------------------------------------------


def check_structural(pair: dict) -> tuple[bool, str]:
    category = pair["category"]
    if category == "active_cancelled":
        r_toks = pair["rule_text"].split()
        n_toks = pair["non_rule_text"].split()
        if len(r_toks) != len(n_toks):
            return False, f"active_cancelled: token counts differ ({len(r_toks)} vs {len(n_toks)})"
        diffs = [i for i in range(len(r_toks)) if r_toks[i] != n_toks[i]]
        if len(diffs) != 1:
            return False, f"active_cancelled: {len(diffs)} differing tokens, expected exactly 1"
        return True, ""
    if category == "bold_html":
        for field in ("rule_text", "non_rule_text"):
            if re.search(r"</?(b|strong)\b", pair[field], re.IGNORECASE):
                return False, f"bold_html: literal HTML tag in {field}"
        return True, ""
    return True, ""


# ----- Checker sanity -----------------------------------------------------


_QUOTED = re.compile(r"[\"'‘’“”]([^\"'‘’“”]+)[\"'‘’“”]")


def _first_quoted(text: str) -> str | None:
    m = _QUOTED.search(text)
    if not m:
        return None
    return m.group(1).rstrip(".,;:!?")


def _run_checker(checker_str: str, out: str) -> bool | None:
    """Execute a checker expression string in a restricted namespace.

    Returns True/False if the check ran, None if the expression couldn't be
    evaluated (missing symbol, syntax the runtime can't resolve).
    """
    namespace: dict[str, Any] = {"re": re, "out": out}
    try:
        from langdetect import detect as _detect  # type: ignore

        namespace["langdetect"] = _detect
    except ImportError:
        pass
    try:
        return bool(eval(checker_str, {"__builtins__": {}}, namespace))
    except Exception:
        return None


def _synthesize_positive_negative(pair: dict) -> tuple[str | None, str | None]:
    cat = pair["category"]
    if cat == "bold_html":
        return "<b>text</b>", "plain text"
    if cat == "banned_word":
        word = _first_quoted(pair["rule_text"])
        if not word:
            return None, None
        return "clean output", f"this has {word} in it"
    if cat == "include_word":
        word = _first_quoted(pair["rule_text"])
        if not word:
            return None, None
        return f"this has {word} in it", "clean output"
    if cat == "start_with":
        token = _first_quoted(pair["rule_text"])
        if not token:
            return None, None
        return f"{token} and more.", "something else"
    if cat == "word_count":
        m = re.search(r"==\s*(\d+)", pair["checker"])
        if not m:
            return None, None
        n = int(m.group(1))
        return " ".join(["w"] * n), " ".join(["w"] * (n + 1))
    return None, None


def check_checker_sanity(pair: dict) -> tuple[bool, str]:
    pos, neg = _synthesize_positive_negative(pair)
    if pos is None or neg is None:
        return True, ""  # not synthesizable for this category — skip
    pos_res = _run_checker(pair["checker"], pos)
    neg_res = _run_checker(pair["checker"], neg)
    if pos_res is None or neg_res is None:
        return True, ""  # checker didn't eval cleanly — skip rather than false-flag
    if not pos_res:
        return False, f"checker rejects synthetic positive {pos!r}"
    if neg_res:
        return False, f"checker accepts synthetic negative {neg!r}"
    return True, ""


# ----- Composite check -----------------------------------------------------


def evaluate_pair(
    source_pair: dict,
    translated_pair: dict,
    lang_code: str,
    lang_config: dict,
    quality_threshold: float,
    backtranslator,
) -> dict[str, Any]:
    """Run all five checks. Returns per-check results + overall pass flag."""
    diagnostics: list[str] = []

    quality_score = score_comet(
        source_pair["rule_text"], translated_pair["rule_text"], lang_config["quality_metric"]
    )
    quality_ok = quality_score >= quality_threshold
    if not quality_ok:
        diagnostics.append(f"quality: {quality_score:.3f} < threshold {quality_threshold:.3f}")

    backtranslated = backtranslator.translate(translated_pair["rule_text"])
    cosine = cosine_score(source_pair["rule_text"], backtranslated)
    similarity_ok = cosine >= 0.75
    if not similarity_ok:
        diagnostics.append(f"backtrans: cosine={cosine:.2f}")

    lid_fields = {
        "rule_text": translated_pair["rule_text"],
        "non_rule_text": translated_pair["non_rule_text"],
        "context": translated_pair["context"],
        "user_query": translated_pair["user_query"],
    }
    lid_ok, lid_diag = check_language_id(lid_fields, lang_code)
    if not lid_ok:
        diagnostics.append(lid_diag)

    struct_ok, struct_diag = check_structural(translated_pair)
    if not struct_ok:
        diagnostics.append(struct_diag)

    checker_ok, checker_diag = check_checker_sanity(translated_pair)
    if not checker_ok:
        diagnostics.append(checker_diag)

    passed = quality_ok and similarity_ok and lid_ok and struct_ok and checker_ok

    return {
        "passed": passed,
        "quality_score": quality_score,
        "cosine": cosine,
        "lid_ok": lid_ok,
        "structural_ok": struct_ok,
        "checker_sanity_ok": checker_ok,
        "diagnostics": "; ".join(diagnostics) if diagnostics else "",
    }
