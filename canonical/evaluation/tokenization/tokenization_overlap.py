import os
import gc
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from datasets import load_dataset as hf_load_dataset
from itertools import combinations
from typing import NamedTuple, Optional
from transformers import AutoTokenizer
from eflomal import Aligner

SCRIPT_MAP = {
    "en": "Latin",
    "de": "Latin",
    "it": "Latin",
    "yo": "Latin",
    "ig": "Latin",
    "tr": "Latin",
    "ru": "Cyrillic",
    "ko": "Hangul",
    "hi": "Devanagari",
    "ur": "Arabic",
}

# Mapping to flores_plus's real iso_639_3 values -- plain 3-letter codes, NOT the
# "eng_Latn" compound style (verified against the dataset's actual schema: script
# lives in a separate iso_15924 column, e.g. {"iso_639_3": "gla", "iso_15924":
# "Latn", ...}). None of our 10 languages have multiple script variants in
# flores_plus, so filtering on iso_639_3 alone is enough -- not joining iso_15924
# too is a deliberate simplification, not an oversight.
FLORES_CODE_MAP = {
    "en": "eng",
    "de": "deu",
    "it": "ita",
    "yo": "yor",
    "ig": "ibo",
    "tr": "tur",
    "ru": "rus",
    "ko": "kor",
    "hi": "hin",
    "ur": "urd",
}

TOKENIZER_IDS = {
    "gemma3": "google/gemma-3-4b-it",
    "llama3": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
}

# canonical rule-following dataset.
CANONICAL_DATASET_ID = "crosslingual-rule-following/canonical-dataset"
CANONICAL_TEXT_FIELDS = ["system_rule", "system_non_rule", "user_query"]

CANONICAL_PROMPT_FIELDS = ["prompt_active", "prompt_revoked"]
CANONICAL_PROMPT_SEP = "\n\n"

CANONICAL_LANGUAGES = [
    "en",
    "am",
    "de",
    "hi",
    "ig",
    "it",
    "ko",
    "ru",
    "sw",
    "ta",
    "tr",
    "ur",
    "yo",
]

# --- compute/memory knobs -----------------------------------------------.
MAX_SENTENCES_PER_LANG = None
EFLOMAL_MODEL = 3  # 1=IBM1, 2=+HMM, 3=+fertility (eflomal's default)
EFLOMAL_N_SAMPLERS = 3
TOKENIZE_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# 1. Vocabulary & Fertility Functions
# ---------------------------------------------------------------------------


def compute_fertility_and_vocab(
    df: pd.DataFrame,
    text_column: str,
    tokenizer,
    lang_column: str = "language",
    batch_size: int = 64,
):
    """Computes unique vocabulary sets and token fertility metrics per language."""
    lang_vocab = defaultdict(set)
    fertility_records = []

    for lang, group in df.groupby(lang_column):
        texts = group[text_column].dropna().astype(str).tolist()
        total_tokens, total_words, total_chars = 0, 0, 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(batch, add_special_tokens=False)

            for text, input_ids in zip(batch, encoded["input_ids"]):
                lang_vocab[lang].update(input_ids)
                total_tokens += len(input_ids)
                total_words += len(text.strip().split())
                total_chars += len(text.strip())

        fertility_records.append(
            {
                "language": lang,
                "script": SCRIPT_MAP.get(lang, "Unknown"),
                "tokens_per_word": (
                    total_tokens / total_words if total_words > 0 else 0.0
                ),
                "tokens_per_char": (
                    total_tokens / total_chars if total_chars > 0 else 0.0
                ),
                "vocab_used": len(lang_vocab[lang]),
            }
        )

    fertility_df = pd.DataFrame(fertility_records).set_index("language")
    return lang_vocab, fertility_df


def compute_jaccard_matrix(lang_vocab: dict[str, set[int]]) -> pd.DataFrame:
    """Computes pairwise Jaccard similarity matrix across languages."""
    languages = sorted(lang_vocab.keys())
    matrix = pd.DataFrame(index=languages, columns=languages, dtype=float)

    for i, l1 in enumerate(languages):
        for l2 in languages[i:]:
            s1, s2 = lang_vocab[l1], lang_vocab[l2]
            if not s1 or not s2:
                value = 0.0
            else:
                intersection = len(s1.intersection(s2))
                union = len(s1.union(s2))
                value = intersection / union if union > 0 else 0.0
            matrix.loc[l1, l2] = value
            matrix.loc[l2, l1] = value

    return matrix


# ---------------------------------------------------------------------------
# 2. eflomal Subword Token Alignability Functions
# ---------------------------------------------------------------------------


def tokenize_lines(
    texts: list[str], tokenizer, batch_size: int = TOKENIZE_BATCH_SIZE
) -> list[str]:
    """
    Batch-tokenizes texts into eflomal-ready, space-joined token-string lines.
    Same token-string construction as the original (convert_ids_to_tokens +
    space/newline escaping.
    """
    lines = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_ids = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in batch_ids:
            toks = tokenizer.convert_ids_to_tokens(ids)
            lines.append(" ".join(t.replace(" ", "_").replace("\n", "↵") for t in toks))
    return lines


def align_lines(src_lines: list[str], tgt_lines: list[str]) -> dict:
    """
    Runs eflomal on already-tokenized, matched line lists and computes
    coverage/density.
    """
    min_len = min(len(src_lines), len(tgt_lines))
    if min_len == 0:
        return {"alignability_coverage": 0.0, "alignment_density": 0.0}
    src_lines, tgt_lines = src_lines[:min_len], tgt_lines[:min_len]

    with tempfile.TemporaryDirectory() as tmpdir:
        links_path = os.path.join(tmpdir, "links.align")
        Aligner(model=EFLOMAL_MODEL, n_samplers=EFLOMAL_N_SAMPLERS).align(
            src_lines, tgt_lines, links_filename_fwd=links_path, quiet=True
        )

        total_src_tokens = 0
        aligned_src_tokens = set()
        total_alignments = 0

        with open(links_path, "r", encoding="utf-8") as flink:
            for line_idx, (src_line, link_line) in enumerate(zip(src_lines, flink)):
                src_part = src_line.strip().split()
                total_src_tokens += len(src_part)

                links = link_line.strip().split()
                total_alignments += len(links)

                for link in links:
                    if "-" in link:
                        s_idx, _ = map(int, link.split("-"))
                        aligned_src_tokens.add((line_idx, s_idx))

    coverage = (
        len(aligned_src_tokens) / total_src_tokens if total_src_tokens > 0 else 0.0
    )
    density = total_alignments / total_src_tokens if total_src_tokens > 0 else 0.0

    return {"alignability_coverage": coverage, "alignment_density": density}


def compute_eflomal_alignability(
    df_parallel: pd.DataFrame, src_lang: str, tgt_lang: str, tokenizer
) -> dict:
    src_fl_code = FLORES_CODE_MAP.get(src_lang, src_lang)
    tgt_fl_code = FLORES_CODE_MAP.get(tgt_lang, tgt_lang)
    src_texts = df_parallel[df_parallel["iso_639_3"] == src_fl_code][
        "sentence"
    ].tolist()
    tgt_texts = df_parallel[df_parallel["iso_639_3"] == tgt_fl_code][
        "sentence"
    ].tolist()
    if not src_texts or not tgt_texts:
        return {"alignability_coverage": 0.0, "alignment_density": 0.0}
    return align_lines(
        tokenize_lines(src_texts, tokenizer), tokenize_lines(tgt_texts, tokenizer)
    )


# ---------------------------------------------------------------------------
# 3. Visualization Pipeline
# ---------------------------------------------------------------------------


def generate_visualizations(
    model_slug: str,
    overlap_df: pd.DataFrame,
    fertility_df: pd.DataFrame,
    alignability_df: pd.DataFrame,
):
    """Generates 3-panel plot: Jaccard Overlap, Token Fertility, and eflomal Alignability."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Panel 1: Jaccard Overlap Matrix
    sns.heatmap(
        overlap_df.astype(float),
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": "Jaccard Overlap Index"},
        linewidths=0.5,
    )
    axes[0].set_title(
        f"Subword Overlap Matrix ({model_slug})", fontsize=12, fontweight="bold"
    )

    # Panel 2: Token Fertility
    fertility_sorted = fertility_df.reset_index().sort_values(by="tokens_per_word")
    barplot = sns.barplot(
        data=fertility_sorted,
        x="language",
        y="tokens_per_word",
        hue="script",
        dodge=False,
        palette="Set2",
        ax=axes[1],
    )
    axes[1].set_title(
        f"Token Fertility (Tokens/Word) ({model_slug})", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel("Avg Subwords per Word")

    # Panel 3: eflomal Cross-Script Alignability (vs English)
    align_sorted = alignability_df.sort_values(
        by="alignability_coverage", ascending=False
    )
    alignplot = sns.barplot(
        data=align_sorted,
        x="target_lang",
        y="alignability_coverage",
        hue="script",
        dodge=False,
        palette="Blues_d",
        ax=axes[2],
    )
    axes[2].set_title(
        f"eflomal Alignability from English ({model_slug})",
        fontsize=12,
        fontweight="bold",
    )
    axes[2].set_ylabel("Source Token Coverage Ratio")
    axes[2].set_ylim(0, 1.0)

    plt.tight_layout()
    plot_filename = f"full_crosslingual_analysis_{model_slug}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved complete visual report to {plot_filename}")


SUMMARY_SOURCE_ORDER = [
    "flores",
    "canonical:system_rule",
    "canonical:system_non_rule",
    "canonical:user_query",
    "canonical:prompt_active",
    "canonical:prompt_revoked",
]


def _source_facet_label(source: str) -> str:
    return "FLORES" if source == "flores" else source.split(":", 1)[1]


def plot_summary_visualizations(
    fertility_stats: pd.DataFrame,
    jaccard_stats: pd.DataFrame,
    alignability_stats: pd.DataFrame,
    src_lang: str = "en",
):
    """ """
    sns.set_theme(style="whitegrid")

    models = sorted(
        set(fertility_stats["model"])
        | set(jaccard_stats["model"])
        | set(alignability_stats["model"])
    )
    palette = dict(zip(models, sns.color_palette("deep", n_colors=len(models))))

    def _faceted_barplot(data, x, y, ylabel, suptitle, filename, ylim=None):
        sources = [s for s in SUMMARY_SOURCE_ORDER if s in data["source"].unique()]
        sources += [
            s for s in sorted(data["source"].unique()) if s not in sources
        ]  # any leftovers, stable fallback

        ncols = 3
        nrows = -(-len(sources) // ncols)  # ceil
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6.5 * ncols, 5 * nrows), squeeze=False
        )

        lang_order = data.groupby(x)[y].mean().sort_values().index.tolist()
        y_max = ylim[1] if ylim is not None else data[y].max() * 1.1

        for i, source in enumerate(sources):
            ax = axes[i // ncols][i % ncols]
            sub = data[data["source"] == source]
            sns.barplot(
                data=sub,
                x=x,
                y=y,
                hue="model",
                order=lang_order,
                hue_order=models,
                palette=palette,
                ax=ax,
            )
            ax.set_title(_source_facet_label(source), fontsize=11, fontweight="bold")
            ax.set_ylabel(ylabel if i % ncols == 0 else "")
            ax.set_xlabel("")
            ax.set_ylim(0, y_max)
            ax.tick_params(axis="x", rotation=45)
            if i == 0:
                ax.legend(title="Model", fontsize=8, title_fontsize=9)
            else:
                ax.get_legend().remove()

        for j in range(len(sources), nrows * ncols):  # hide any unused grid cells
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(suptitle, fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")

    # 1. Fertility: tokens_per_word, one subplot per source
    _faceted_barplot(
        fertility_stats,
        x="language",
        y="tokens_per_word",
        ylabel="Tokens / Word",
        suptitle="Token Fertility by Language, Model, and Source",
        filename="summary_fertility_by_model.png",
    )

    # 2. Overlap: Jaccard vs. src_lang specifically (the one comparison every
    # other plot below is already anchored on).
    jac_vs_src = jaccard_stats[
        (jaccard_stats["language_1"] == src_lang)
        & (jaccard_stats["language_2"] != src_lang)
    ].rename(columns={"language_2": "language"})
    _faceted_barplot(
        jac_vs_src,
        x="language",
        y="jaccard",
        ylabel=f"Jaccard vs {src_lang}",
        suptitle=f"Vocabulary Overlap with {src_lang.upper()} by Language, Model, and Source",
        filename="summary_overlap_by_model.png",
        ylim=(0, 1.0),
    )

    # 3. eflomal alignability from src_lang, one subplot per source.
    _faceted_barplot(
        alignability_stats,
        x="target_lang",
        y="alignability_coverage",
        ylabel="Source Token Coverage Ratio",
        suptitle=f"eflomal Alignability from {src_lang.upper()} by Language, Model, and Source",
        filename="summary_alignability_by_model.png",
        ylim=(0, 1.0),
    )


# ---------------------------------------------------------------------------
# 4. Canonical dataset loading.
# ---------------------------------------------------------------------------


def cap_per_language(
    df: pd.DataFrame,
    lang_col: str = "language",
    max_n: int | None = MAX_SENTENCES_PER_LANG,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Subsamples each language group down to at most max_n rows.
    """
    if max_n is None:
        return df
    return (
        df.groupby(lang_col, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), max_n), random_state=seed))
        .reset_index(drop=True)
    )


def load_canonical_by_field(split: str = "test") -> dict[str, pd.DataFrame]:
    frames = []
    for lang in CANONICAL_LANGUAGES:
        raw = hf_load_dataset(CANONICAL_DATASET_ID, lang, split=split)
        lang_df = pd.DataFrame(raw)
        # the config name IS the language here.
        lang_df["language"] = lang
        frames.append(lang_df)
    df = pd.concat(frames, ignore_index=True)

    # positional alignment assumption: compute_eflomal_alignability zips src/tgt
    # lists by position (same as it already does for FLORES).
    df = df.sort_values("id")

    active_mask = df["system_rule"].notna() & df["user_query"].notna()
    df["prompt_active"] = pd.NA
    df.loc[active_mask, "prompt_active"] = (
        df.loc[active_mask, "system_rule"]
        + CANONICAL_PROMPT_SEP
        + df.loc[active_mask, "user_query"]
    )
    revoked_mask = df["system_non_rule"].notna() & df["user_query"].notna()
    df["prompt_revoked"] = pd.NA
    df.loc[revoked_mask, "prompt_revoked"] = (
        df.loc[revoked_mask, "system_non_rule"]
        + CANONICAL_PROMPT_SEP
        + df.loc[revoked_mask, "user_query"]
    )

    by_field = {}
    for text_field in CANONICAL_TEXT_FIELDS + CANONICAL_PROMPT_FIELDS:
        df_f = df.dropna(subset=[text_field]).copy()
        df_f["sentence"] = df_f[text_field]
        by_field[text_field] = df_f
    return by_field


# ---------------------------------------------------------------------------
# 5. Main Controller
# ---------------------------------------------------------------------------


def main():
    print("--- Loading FLORES-200 Dataset ---")
    raw_ds = hf_load_dataset("openlanguagedata/flores_plus", split="devtest")
    df_flores = pd.DataFrame(raw_ds)

    # Mapping FLORES columns
    df_flores["language"] = df_flores["iso_639_3"].map(
        {v: k for k, v in FLORES_CODE_MAP.items()}
    )
    df_flores["sentence"] = df_flores["text"]
    df_flores = cap_per_language(
        df_flores
    )  # no-op unless MAX_SENTENCES_PER_LANG is set
    df_eval = df_flores.dropna(subset=["language"])

    src_lang = "en"
    target_langs = [l for l in SCRIPT_MAP.keys() if l != src_lang]

    print("--- Loading canonical-dataset ---")
    canon_by_field = {
        field: cap_per_language(df) for field, df in load_canonical_by_field().items()
    }

    all_fertility_rows = []
    all_alignability_rows = []
    all_jaccard_rows = []

    for model_slug, model_id in TOKENIZER_IDS.items():
        print(f"\n==================================================")
        print(f"Evaluating Tokenizer: {model_slug} ({model_id})")
        print(f"==================================================")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # 1. Jaccard & Fertility Analysis (FLORES) -- unchanged
        lang_vocab, fertility_df = compute_fertility_and_vocab(
            df_eval, text_column="sentence", tokenizer=tokenizer, lang_column="language"
        )
        overlap_df = compute_jaccard_matrix(lang_vocab)

        # 2. eflomal Alignability Analysis (FLORES).
        src_fl_code = FLORES_CODE_MAP.get(src_lang, src_lang)
        src_texts = df_flores[df_flores["iso_639_3"] == src_fl_code][
            "sentence"
        ].tolist()
        src_lines_cached = tokenize_lines(src_texts, tokenizer)

        alignability_records = []
        for tgt in target_langs:
            print(f"Computing eflomal alignability: {src_lang} -> {tgt}...")
            tgt_fl_code = FLORES_CODE_MAP.get(tgt, tgt)
            tgt_texts = df_flores[df_flores["iso_639_3"] == tgt_fl_code][
                "sentence"
            ].tolist()
            res = align_lines(src_lines_cached, tokenize_lines(tgt_texts, tokenizer))
            alignability_records.append(
                {
                    "target_lang": tgt,
                    "script": SCRIPT_MAP.get(tgt, "Unknown"),
                    "alignability_coverage": res["alignability_coverage"],
                    "alignment_density": res["alignment_density"],
                }
            )

        alignability_df = pd.DataFrame(alignability_records)

        # 3. Print Unified Report (FLORES)
        print("\n--- Summary Report ---")
        print(fertility_df[["script", "tokens_per_word", "vocab_used"]].round(3))
        print("\n--- eflomal Alignability (from English) ---")
        print(alignability_df.to_string(index=False))

        # 4. Plotting moved out of the per-run loop.

        # 4b. Stash for CSV export.
        _fert = fertility_df.reset_index()
        _fert["model"] = model_slug
        _fert["source"] = "flores"
        all_fertility_rows.append(_fert)

        _align = alignability_df.copy()
        _align["model"] = model_slug
        _align["source"] = "flores"
        all_alignability_rows.append(_align)

        _jac = overlap_df.reset_index(names="language_1").melt(
            id_vars="language_1", var_name="language_2", value_name="jaccard"
        )
        _jac["model"] = model_slug
        _jac["source"] = "flores"
        all_jaccard_rows.append(_jac)

        # 5. Same three steps again, on the canonical dataset -- one pass per
        #    text field (system_rule / system_non_rule / user_query / the two
        #    concatenated prompt_active / prompt_revoked views).
        for text_field, df_canon in canon_by_field.items():
            canon_target_langs = [
                l for l in df_canon["language"].unique() if l != src_lang
            ]
            print(f"\n--- Canonical dataset field: {text_field} ---")

            lang_vocab_c, fertility_df_c = compute_fertility_and_vocab(
                df_canon,
                text_column="sentence",
                tokenizer=tokenizer,
                lang_column="language",
            )
            overlap_df_c = compute_jaccard_matrix(lang_vocab_c)

            # same src-caching as the FLORES pass above.
            src_texts_c = df_canon[df_canon["language"] == src_lang][
                "sentence"
            ].tolist()
            src_lines_cached_c = tokenize_lines(src_texts_c, tokenizer)

            alignability_records_c = []
            for tgt in canon_target_langs:
                print(
                    f"Computing eflomal alignability ({text_field}): {src_lang} -> {tgt}..."
                )
                tgt_texts_c = df_canon[df_canon["language"] == tgt]["sentence"].tolist()
                res = align_lines(
                    src_lines_cached_c, tokenize_lines(tgt_texts_c, tokenizer)
                )
                alignability_records_c.append(
                    {
                        "target_lang": tgt,
                        "script": SCRIPT_MAP.get(tgt, "Unknown"),
                        "alignability_coverage": res["alignability_coverage"],
                        "alignment_density": res["alignment_density"],
                    }
                )
            alignability_df_c = pd.DataFrame(alignability_records_c)

            print(f"\n--- Summary Report ({text_field}) ---")
            print(fertility_df_c[["script", "tokens_per_word", "vocab_used"]].round(3))
            print(f"\n--- eflomal Alignability ({text_field}, from English) ---")
            print(alignability_df_c.to_string(index=False))

            _fert_c = fertility_df_c.reset_index()
            _fert_c["model"] = model_slug
            _fert_c["source"] = f"canonical:{text_field}"
            all_fertility_rows.append(_fert_c)

            _align_c = alignability_df_c.copy()
            _align_c["model"] = model_slug
            _align_c["source"] = f"canonical:{text_field}"
            all_alignability_rows.append(_align_c)

            _jac_c = overlap_df_c.reset_index(names="language_1").melt(
                id_vars="language_1", var_name="language_2", value_name="jaccard"
            )
            _jac_c["model"] = model_slug
            _jac_c["source"] = f"canonical:{text_field}"
            all_jaccard_rows.append(_jac_c)

        # free this tokenizer before loading the next one
        del tokenizer
        gc.collect()

    # ---------------------------------------------------------------------
    # 6. CSV export.
    # ---------------------------------------------------------------------
    fertility_stats = pd.concat(all_fertility_rows, ignore_index=True)
    alignability_stats = pd.concat(all_alignability_rows, ignore_index=True)
    jaccard_stats = pd.concat(all_jaccard_rows, ignore_index=True)

    fertility_stats.to_csv("fertility_stats.csv", index=False)
    alignability_stats.to_csv("alignability_stats.csv", index=False)
    jaccard_stats.to_csv("jaccard_stats.csv", index=False)
    print(
        "\nSaved fertility_stats.csv, alignability_stats.csv, jaccard_stats.csv "
        f"({len(fertility_stats)}, {len(alignability_stats)}, {len(jaccard_stats)} rows)"
    )

    # ---------------------------------------------------------------------
    # 7. Exactly three summary plots, across every model.
    # ---------------------------------------------------------------------
    plot_summary_visualizations(
        fertility_stats, jaccard_stats, alignability_stats, src_lang=src_lang
    )


if __name__ == "__main__":
    main()
