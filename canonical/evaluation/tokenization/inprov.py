import gc
import os
import tempfile
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset as hf_load_dataset
from eflomal import Aligner
from transformers import AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

# The analysis is intentionally restricted to these 10 languages for BOTH
# FLORES and the canonical benchmark.
CANONICAL_LANGUAGES = [
    "en",
    "de",
    "hi",
    "ig",
    "it",
    "ko",
    "ru",
    "tr",
    "ur",
    "yo",
]

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

# FLORES uses ISO 639-3 codes.
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

CANONICAL_DATASET_ID = "crosslingual-rule-following/canonical-dataset"

# ---------------------------------------------------------------------------
# Canonical prompt conditions
#
# These are the ONLY two canonical conditions included in the main
# tokenizer analysis.
#
# The underlying columns remain in the dataset because they are required
# to construct the prompts.
# ---------------------------------------------------------------------------

CANONICAL_CONDITIONS = {
    "active": {
        "system_column": "system_rule",
        "output_column": "prompt_active",
        "source_label": "canonical_active",
    },
    "revoked": {
        "system_column": "system_non_rule",
        "output_column": "prompt_revoked",
        "source_label": "canonical_revoked",
    },
}

CANONICAL_PROMPT_SEP = "\n\n"

# ---------------------------------------------------------------------------
# Compute / memory knobs
# ---------------------------------------------------------------------------

# None = use all available examples.
#
# For monolingual statistics, this caps each language independently.
# For parallel alignment, sentence IDs are sampled ONCE and then retained
# across all languages.
MAX_SENTENCES_PER_LANG = None

EFLOMAL_MODEL = 3
EFLOMAL_N_SAMPLERS = 3

TOKENIZE_BATCH_SIZE = 64


# =============================================================================
# 1. Tokenization / Vocabulary / Fertility
# =============================================================================


def compute_fertility_and_vocab(
    df: pd.DataFrame,
    text_column: str,
    tokenizer,
    lang_column: str = "language",
    batch_size: int = TOKENIZE_BATCH_SIZE,
):
    """
    Compute tokenizer statistics independently for each language.

    Token fertility:

        number of tokenizer tokens
        --------------------------
        number of whitespace-delimited words

    Vocabulary:

        Set of tokenizer vocabulary IDs observed in the text.
    """

    lang_vocab = defaultdict(set)
    fertility_records = []

    for lang, group in df.groupby(lang_column):
        texts = group[text_column].dropna().astype(str).tolist()

        total_tokens = 0
        total_words = 0
        total_chars = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            encoded = tokenizer(
                batch,
                add_special_tokens=False,
            )

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
                "n_sentences": len(texts),
                "n_words": total_words,
                "n_tokens": total_tokens,
                "n_chars": total_chars,
            }
        )

    fertility_df = pd.DataFrame(fertility_records).set_index("language")

    return lang_vocab, fertility_df


def compute_jaccard_matrix(
    lang_vocab: dict[str, set[int]],
) -> pd.DataFrame:
    """
    Compute pairwise token-type vocabulary Jaccard similarity.

        J(A, B) = |A ∩ B| / |A ∪ B|

    A and B are sets of tokenizer vocabulary IDs observed in each language.
    """

    languages = sorted(lang_vocab.keys())

    matrix = pd.DataFrame(
        index=languages,
        columns=languages,
        dtype=float,
    )

    for i, lang_1 in enumerate(languages):
        for lang_2 in languages[i:]:
            set_1 = lang_vocab[lang_1]
            set_2 = lang_vocab[lang_2]

            if not set_1 or not set_2:
                value = 0.0
            else:
                intersection = len(set_1.intersection(set_2))
                union = len(set_1.union(set_2))

                value = intersection / union if union > 0 else 0.0

            matrix.loc[lang_1, lang_2] = value
            matrix.loc[lang_2, lang_1] = value

    return matrix


# =============================================================================
# 2. Tokenizer Subword Preparation
# =============================================================================


def tokenize_lines(
    texts: list[str],
    tokenizer,
    batch_size: int = TOKENIZE_BATCH_SIZE,
) -> list[str]:
    """
    Convert text into space-separated tokenizer subword strings suitable
    for eflomal.

    eflomal therefore operates on tokenizer-derived subword units rather
    than linguistically segmented words.
    """

    lines = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        batch_ids = tokenizer(
            batch,
            add_special_tokens=False,
        )["input_ids"]

        for ids in batch_ids:
            tokens = tokenizer.convert_ids_to_tokens(ids)

            line = " ".join(
                token.replace(" ", "_").replace("\n", "↵") for token in tokens
            )

            lines.append(line)

    return lines


# =============================================================================
# 3. Parallel Data Handling
# =============================================================================


def cap_per_language(
    df: pd.DataFrame,
    lang_col: str = "language",
    max_n: int | None = MAX_SENTENCES_PER_LANG,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Cap each language independently.

    Used ONLY for monolingual statistics such as fertility and vocabulary.

    Do not use this independently sampled dataframe for parallel alignment.
    """

    if max_n is None:
        return df.copy()

    return (
        df.groupby(lang_col, group_keys=False)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_n),
                random_state=seed,
            )
        )
        .reset_index(drop=True)
    )


def cap_parallel_by_id(
    df: pd.DataFrame,
    id_col: str = "id",
    max_n: int | None = MAX_SENTENCES_PER_LANG,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Cap a multilingual parallel dataset by sampling sentence IDs once.

    The same IDs are retained across all languages, preserving
    source-target correspondence.
    """

    if max_n is None:
        return df.copy()

    unique_ids = df[id_col].drop_duplicates()

    sampled_ids = unique_ids.sample(
        n=min(len(unique_ids), max_n),
        random_state=seed,
    )

    return df[df[id_col].isin(sampled_ids)].copy()


# =============================================================================
# 4. FLORES Parallel Data
# =============================================================================


def get_flores_parallel_texts(
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
) -> tuple[list[str], list[str]]:
    """
    Construct explicit FLORES source-target sentence pairs using sentence ID.

    This avoids relying on incidental row ordering.
    """

    src_code = FLORES_CODE_MAP[src_lang]
    tgt_code = FLORES_CODE_MAP[tgt_lang]

    src = df[df["iso_639_3"] == src_code][["id", "sentence"]].rename(
        columns={"sentence": "src_sentence"}
    )

    tgt = df[df["iso_639_3"] == tgt_code][["id", "sentence"]].rename(
        columns={"sentence": "tgt_sentence"}
    )

    if src.empty:
        raise ValueError(f"No FLORES sentences found for source language: {src_lang}")

    if tgt.empty:
        raise ValueError(f"No FLORES sentences found for target language: {tgt_lang}")

    parallel = src.merge(
        tgt,
        on="id",
        how="inner",
        validate="one_to_one",
    ).sort_values("id")

    if parallel.empty:
        raise ValueError(
            f"No parallel FLORES sentence pairs found for " f"{src_lang} -> {tgt_lang}"
        )

    return (
        parallel["src_sentence"].tolist(),
        parallel["tgt_sentence"].tolist(),
    )


# =============================================================================
# 5. Canonical Parallel Data
# =============================================================================


def get_canonical_parallel_texts(
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
) -> tuple[list[str], list[str]]:
    """
    Construct explicit source-target sentence pairs from the canonical
    dataset using the shared item ID.

    The canonical dataframe must contain:
        id
        language
        sentence
    """

    src = df[df["language"] == src_lang][["id", "sentence"]].rename(
        columns={"sentence": "src_sentence"}
    )

    tgt = df[df["language"] == tgt_lang][["id", "sentence"]].rename(
        columns={"sentence": "tgt_sentence"}
    )

    if src.empty:
        raise ValueError(
            f"No canonical sentences found for source language: {src_lang}"
        )

    if tgt.empty:
        raise ValueError(
            f"No canonical sentences found for target language: {tgt_lang}"
        )

    parallel = src.merge(
        tgt,
        on="id",
        how="inner",
        validate="one_to_one",
    ).sort_values("id")

    if parallel.empty:
        raise ValueError(
            f"No parallel canonical sentence pairs found for "
            f"{src_lang} -> {tgt_lang}"
        )

    return (
        parallel["src_sentence"].tolist(),
        parallel["tgt_sentence"].tolist(),
    )


# =============================================================================
# 6. eflomal Subword Alignment
# =============================================================================


def align_lines(
    src_lines: list[str],
    tgt_lines: list[str],
) -> dict:
    """
    Run eflomal on matched tokenizer-subword sequences.

    Metrics:

    subword_alignment_coverage
        Fraction of source tokenizer subwords participating in at least
        one source-to-target alignment link.

    subword_alignment_density
        Number of alignment links per source tokenizer subword.
    """

    if len(src_lines) != len(tgt_lines):
        raise ValueError(
            "Parallel inputs have different numbers of sentences: "
            f"{len(src_lines)} source vs {len(tgt_lines)} target."
        )

    if not src_lines:
        return {
            "subword_alignment_coverage": 0.0,
            "subword_alignment_density": 0.0,
            "n_sentences": 0,
            "n_source_subwords": 0,
            "n_target_subwords": 0,
            "n_aligned_source_subwords": 0,
            "n_alignment_links": 0,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        links_path = os.path.join(
            tmpdir,
            "links.align",
        )

        Aligner(
            model=EFLOMAL_MODEL,
            n_samplers=EFLOMAL_N_SAMPLERS,
        ).align(
            src_lines,
            tgt_lines,
            links_filename_fwd=links_path,
            quiet=True,
        )

        total_src_tokens = 0
        total_tgt_tokens = 0
        aligned_src_tokens = set()
        total_alignments = 0

        with open(
            links_path,
            "r",
            encoding="utf-8",
        ) as flink:

            for line_idx, (src_line, tgt_line, link_line) in enumerate(
                zip(src_lines, tgt_lines, flink)
            ):
                src_tokens = src_line.strip().split()
                tgt_tokens = tgt_line.strip().split()

                total_src_tokens += len(src_tokens)
                total_tgt_tokens += len(tgt_tokens)

                links = link_line.strip().split()

                total_alignments += len(links)

                for link in links:
                    if "-" not in link:
                        continue

                    src_idx, tgt_idx = map(
                        int,
                        link.split("-"),
                    )

                    if not (0 <= src_idx < len(src_tokens)):
                        raise ValueError(
                            f"Invalid source token index {src_idx} "
                            f"for sentence {line_idx}."
                        )

                    if not (0 <= tgt_idx < len(tgt_tokens)):
                        raise ValueError(
                            f"Invalid target token index {tgt_idx} "
                            f"for sentence {line_idx}."
                        )

                    aligned_src_tokens.add((line_idx, src_idx))

    aligned_source_count = len(aligned_src_tokens)

    coverage = aligned_source_count / total_src_tokens if total_src_tokens > 0 else 0.0

    density = total_alignments / total_src_tokens if total_src_tokens > 0 else 0.0

    return {
        "subword_alignment_coverage": coverage,
        "subword_alignment_density": density,
        "n_sentences": len(src_lines),
        "n_source_subwords": total_src_tokens,
        "n_target_subwords": total_tgt_tokens,
        "n_aligned_source_subwords": aligned_source_count,
        "n_alignment_links": total_alignments,
    }


def compute_flores_alignment(
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    tokenizer,
) -> dict:
    """
    Compute tokenizer-subword alignment metrics for FLORES.
    """

    src_texts, tgt_texts = get_flores_parallel_texts(
        df,
        src_lang,
        tgt_lang,
    )

    src_lines = tokenize_lines(
        src_texts,
        tokenizer,
    )

    tgt_lines = tokenize_lines(
        tgt_texts,
        tokenizer,
    )

    return align_lines(
        src_lines,
        tgt_lines,
    )


def compute_canonical_alignment(
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    tokenizer,
) -> dict:
    """
    Compute tokenizer-subword alignment metrics for one canonical condition.

    The condition is already encoded in the 'sentence' column, so this
    function deliberately has no knowledge of system_rule,
    system_non_rule, or user_query.
    """

    src_texts, tgt_texts = get_canonical_parallel_texts(
        df,
        src_lang,
        tgt_lang,
    )

    src_lines = tokenize_lines(
        src_texts,
        tokenizer,
    )

    tgt_lines = tokenize_lines(
        tgt_texts,
        tokenizer,
    )

    return align_lines(
        src_lines,
        tgt_lines,
    )


# =============================================================================
# 7. Canonical Dataset Loading
# =============================================================================


def load_canonical_dataset(
    split: str = "test",
) -> pd.DataFrame:
    """
    Load the canonical benchmark for exactly the 10 analysis languages.

    The underlying dataset retains:
        system_rule
        system_non_rule
        user_query

    These are used ONLY to construct:
        prompt_active
        prompt_revoked

    The tokenizer analysis itself operates only on those two constructed
    prompts.
    """

    frames = []

    for lang in CANONICAL_LANGUAGES:
        raw = hf_load_dataset(
            CANONICAL_DATASET_ID,
            lang,
            split=split,
        )

        lang_df = pd.DataFrame(raw)

        # The dataset configuration identifies the language.
        lang_df["language"] = lang

        frames.append(lang_df)

    df = pd.concat(
        frames,
        ignore_index=True,
    )

    required_columns = {
        "id",
        "system_rule",
        "system_non_rule",
        "user_query",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Canonical dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # -------------------------------------------------------------------------
    # Construct Active prompt
    #
    # Active = system_rule + user_query
    # -------------------------------------------------------------------------

    active_mask = df["system_rule"].notna() & df["user_query"].notna()

    df["prompt_active"] = pd.NA

    df.loc[active_mask, "prompt_active"] = (
        df.loc[active_mask, "system_rule"].astype(str)
        + CANONICAL_PROMPT_SEP
        + df.loc[active_mask, "user_query"].astype(str)
    )

    # -------------------------------------------------------------------------
    # Construct Revoked prompt
    #
    # Revoked = system_non_rule + user_query
    # -------------------------------------------------------------------------

    revoked_mask = df["system_non_rule"].notna() & df["user_query"].notna()

    df["prompt_revoked"] = pd.NA

    df.loc[revoked_mask, "prompt_revoked"] = (
        df.loc[revoked_mask, "system_non_rule"].astype(str)
        + CANONICAL_PROMPT_SEP
        + df.loc[revoked_mask, "user_query"].astype(str)
    )

    # -------------------------------------------------------------------------
    # Validate language coverage
    # -------------------------------------------------------------------------

    observed_languages = set(df["language"].unique())
    expected_languages = set(CANONICAL_LANGUAGES)

    missing_languages = expected_languages - observed_languages

    if missing_languages:
        raise ValueError(
            "Canonical dataset is missing languages: " f"{sorted(missing_languages)}"
        )

    unexpected_languages = observed_languages - expected_languages

    if unexpected_languages:
        raise ValueError(
            "Canonical dataset contains unexpected languages: "
            f"{sorted(unexpected_languages)}"
        )

    # Deterministic ordering.
    df = df.sort_values(["id", "language"]).reset_index(drop=True)

    return df


def prepare_canonical_conditions(
    df: pd.DataFrame,
) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Create the two canonical analysis conditions:

        active
        revoked

    Each condition contains:
        mono       -> independently capped data for monolingual statistics
        parallel   -> ID-preserving data for alignment
        source     -> clean reporting label

    No separate system_rule, system_non_rule, or user_query analyses
    are produced.
    """

    conditions = {}

    for condition_name, config in CANONICAL_CONDITIONS.items():
        prompt_column = config["output_column"]
        source_label = config["source_label"]

        condition_df = df.dropna(subset=[prompt_column]).copy()

        condition_df["sentence"] = condition_df[prompt_column].astype(str)

        conditions[condition_name] = {
            "mono": cap_per_language(condition_df),
            "parallel": cap_parallel_by_id(condition_df),
            "source": source_label,
        }

    return conditions


# =============================================================================
# 8. Summary Visualization
# =============================================================================

SUMMARY_SOURCE_ORDER = [
    "flores",
    "canonical_active",
    "canonical_revoked",
]


def source_label(source: str) -> str:
    """
    Convert internal source names into publication-friendly labels.
    """

    labels = {
        "flores": "FLORES",
        "canonical_active": "Canonical: Active",
        "canonical_revoked": "Canonical: Revoked",
    }

    return labels.get(source, source)


def plot_summary_visualizations(
    fertility_stats: pd.DataFrame,
    jaccard_stats: pd.DataFrame,
    alignability_stats: pd.DataFrame,
    src_lang: str = "en",
):
    """
    Generate three summary figures:

    1. Token fertility
    2. Token-type vocabulary Jaccard overlap with English
    3. Source-token subword alignment coverage

    Sources shown:
        FLORES
        Canonical: Active
        Canonical: Revoked
    """

    sns.set_theme(style="whitegrid")

    models = sorted(
        set(fertility_stats["model"])
        | set(jaccard_stats["model"])
        | set(alignability_stats["model"])
    )

    palette = dict(
        zip(
            models,
            sns.color_palette(
                "deep",
                n_colors=len(models),
            ),
        )
    )

    def _faceted_barplot(
        data,
        x,
        y,
        ylabel,
        suptitle,
        filename,
        ylim=None,
    ):
        sources = [
            source
            for source in SUMMARY_SOURCE_ORDER
            if source in data["source"].unique()
        ]

        # Include unexpected sources deterministically.
        sources += [
            source
            for source in sorted(data["source"].unique())
            if source not in sources
        ]

        ncols = 3
        nrows = max(
            1,
            -(-len(sources) // ncols),
        )

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(
                6.5 * ncols,
                5 * nrows,
            ),
            squeeze=False,
        )

        lang_order = data.groupby(x)[y].mean().sort_values().index.tolist()

        if ylim is not None:
            y_max = ylim[1]
        else:
            data_max = data[y].max()
            y_max = data_max * 1.1 if data_max > 0 else 1.0

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

            ax.set_title(
                source_label(source),
                fontsize=11,
                fontweight="bold",
            )

            ax.set_ylabel(ylabel if i % ncols == 0 else "")

            ax.set_xlabel("")

            ax.set_ylim(
                0,
                y_max,
            )

            ax.tick_params(
                axis="x",
                rotation=45,
            )

            if i == 0:
                ax.legend(
                    title="Model",
                    fontsize=8,
                    title_fontsize=9,
                )
            else:
                legend = ax.get_legend()

                if legend is not None:
                    legend.remove()

        # Hide unused subplot cells.
        for j in range(
            len(sources),
            nrows * ncols,
        ):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(
            suptitle,
            fontsize=15,
            fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plt.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Saved {filename}")

    # -------------------------------------------------------------------------
    # 1. Fertility
    # -------------------------------------------------------------------------

    _faceted_barplot(
        fertility_stats,
        x="language",
        y="tokens_per_word",
        ylabel="Tokenizer Tokens / Whitespace-Delimited Word",
        suptitle="Token Fertility by Language, Model, and Source",
        filename="summary_fertility_by_model.png",
    )

    # -------------------------------------------------------------------------
    # 2. Vocabulary Jaccard vs English
    # -------------------------------------------------------------------------

    jac_vs_src = jaccard_stats[
        (jaccard_stats["language_1"] == src_lang)
        & (jaccard_stats["language_2"] != src_lang)
    ].rename(columns={"language_2": "language"})

    _faceted_barplot(
        jac_vs_src,
        x="language",
        y="jaccard",
        ylabel=f"Token-Type Jaccard vs {src_lang}",
        suptitle=(
            "Tokenizer Vocabulary Overlap with "
            f"{src_lang.upper()} by Language, Model, and Source"
        ),
        filename="summary_overlap_by_model.png",
        ylim=(0, 1.0),
    )

    # -------------------------------------------------------------------------
    # 3. Subword Alignment Coverage
    # -------------------------------------------------------------------------

    _faceted_barplot(
        alignability_stats,
        x="target_lang",
        y="subword_alignment_coverage",
        ylabel="Source Subword Alignment Coverage",
        suptitle=(
            "Subword Alignment Coverage from "
            f"{src_lang.upper()} by Language, Model, and Source"
        ),
        filename="summary_alignability_by_model.png",
        ylim=(0, 1.0),
    )


# =============================================================================
# 9. Main Controller
# =============================================================================


def main():

    # =========================================================================
    # Load FLORES
    # =========================================================================

    print("--- Loading FLORES-200 Dataset ---")

    raw_flores = hf_load_dataset(
        "openlanguagedata/flores_plus",
        split="devtest",
    )

    df_flores_raw = pd.DataFrame(raw_flores)

    # Map FLORES ISO 639-3 codes to our 10-language codes.
    df_flores_raw["language"] = df_flores_raw["iso_639_3"].map(
        {value: key for key, value in FLORES_CODE_MAP.items()}
    )

    df_flores_raw["sentence"] = df_flores_raw["text"]

    # Restrict FLORES to exactly the same 10 languages.
    df_flores_raw = df_flores_raw[
        df_flores_raw["language"].isin(CANONICAL_LANGUAGES)
    ].copy()

    # -------------------------------------------------------------------------
    # Monolingual FLORES data
    #
    # Independent sampling is valid for fertility and vocabulary statistics.
    # -------------------------------------------------------------------------

    df_flores_mono = cap_per_language(df_flores_raw)

    # -------------------------------------------------------------------------
    # Parallel FLORES data
    #
    # IDs are sampled once, preserving sentence correspondence.
    # -------------------------------------------------------------------------

    df_flores_parallel = cap_parallel_by_id(df_flores_raw)

    src_lang = "en"

    target_langs = [lang for lang in CANONICAL_LANGUAGES if lang != src_lang]

    # =========================================================================
    # Load canonical dataset
    # =========================================================================

    print("--- Loading canonical-dataset ---")

    canonical_df = load_canonical_dataset()

    # Prepare ONLY:
    #
    #   Active
    #   Revoked
    #
    # for the tokenizer analysis.
    canonical_conditions = prepare_canonical_conditions(canonical_df)

    # =========================================================================
    # Validate language sets
    # =========================================================================

    expected_languages = set(CANONICAL_LANGUAGES)

    flores_languages = set(df_flores_raw["language"].unique())

    if flores_languages != expected_languages:
        raise ValueError(
            "FLORES language set does not match "
            "the expected 10-language set.\n"
            f"Expected: {sorted(expected_languages)}\n"
            f"Observed: {sorted(flores_languages)}"
        )

    for condition_name, condition in canonical_conditions.items():

        condition_languages = set(condition["mono"]["language"].unique())

        if condition_languages != expected_languages:
            missing = expected_languages - condition_languages

            unexpected = condition_languages - expected_languages

            raise ValueError(
                f"Canonical condition '{condition_name}' "
                "does not contain exactly the expected "
                "10 languages.\n"
                f"Missing: {sorted(missing)}\n"
                f"Unexpected: {sorted(unexpected)}"
            )

    # =========================================================================
    # Result containers
    # =========================================================================

    all_fertility_rows = []
    all_alignability_rows = []
    all_jaccard_rows = []

    # =========================================================================
    # Tokenizer loop
    # =========================================================================

    for model_slug, model_id in TOKENIZER_IDS.items():

        print("\n==================================================")
        print(f"Evaluating Tokenizer: " f"{model_slug} ({model_id})")
        print("==================================================")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # =====================================================================
        # FLORES
        # =====================================================================

        print("\n--- FLORES analysis ---")

        # ---------------------------------------------------------------------
        # Fertility + vocabulary
        # ---------------------------------------------------------------------

        lang_vocab, fertility_df = compute_fertility_and_vocab(
            df_flores_mono,
            text_column="sentence",
            tokenizer=tokenizer,
            lang_column="language",
        )

        overlap_df = compute_jaccard_matrix(lang_vocab)

        # ---------------------------------------------------------------------
        # Subword alignment
        # ---------------------------------------------------------------------

        alignability_records = []

        for tgt_lang in target_langs:

            print("Computing FLORES subword alignment: " f"{src_lang} -> {tgt_lang}...")

            result = compute_flores_alignment(
                df_flores_parallel,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                tokenizer=tokenizer,
            )

            alignability_records.append(
                {
                    "target_lang": tgt_lang,
                    "script": SCRIPT_MAP.get(
                        tgt_lang,
                        "Unknown",
                    ),
                    **result,
                }
            )

        alignability_df = pd.DataFrame(alignability_records)

        # ---------------------------------------------------------------------
        # FLORES report
        # ---------------------------------------------------------------------

        print("\n--- FLORES Token Statistics ---")

        print(
            fertility_df[
                [
                    "script",
                    "tokens_per_word",
                    "tokens_per_char",
                    "vocab_used",
                    "n_sentences",
                    "n_words",
                    "n_tokens",
                ]
            ].round(3)
        )

        print("\n--- FLORES Subword Alignment ---")

        print(alignability_df.to_string(index=False))

        # ---------------------------------------------------------------------
        # FLORES CSV rows
        # ---------------------------------------------------------------------

        fertility_flores = fertility_df.reset_index()

        fertility_flores["model"] = model_slug
        fertility_flores["source"] = "flores"

        all_fertility_rows.append(fertility_flores)

        align_flores = alignability_df.copy()

        align_flores["model"] = model_slug
        align_flores["source"] = "flores"

        all_alignability_rows.append(align_flores)

        jac_flores = overlap_df.reset_index(names="language_1").melt(
            id_vars="language_1",
            var_name="language_2",
            value_name="jaccard",
        )

        jac_flores["model"] = model_slug
        jac_flores["source"] = "flores"

        all_jaccard_rows.append(jac_flores)

        # =====================================================================
        # CANONICAL: ACTIVE + REVOKED
        # =====================================================================

        for condition_name, condition in canonical_conditions.items():

            source_label_value = condition["source"]

            df_canon_mono = condition["mono"]
            df_canon_parallel = condition["parallel"]

            print("\n--- Canonical dataset: " f"{condition_name.upper()} ---")

            # -----------------------------------------------------------------
            # Fertility + vocabulary
            # -----------------------------------------------------------------

            lang_vocab_c, fertility_df_c = compute_fertility_and_vocab(
                df_canon_mono,
                text_column="sentence",
                tokenizer=tokenizer,
                lang_column="language",
            )

            overlap_df_c = compute_jaccard_matrix(lang_vocab_c)

            # -----------------------------------------------------------------
            # Determine available target languages.
            # -----------------------------------------------------------------

            available_languages = set(df_canon_parallel["language"].unique())

            canon_target_langs = [
                lang for lang in target_langs if lang in available_languages
            ]

            # -----------------------------------------------------------------
            # Subword alignment
            # -----------------------------------------------------------------

            alignability_records_c = []

            for tgt_lang in canon_target_langs:

                print(
                    f"Computing canonical subword alignment "
                    f"({condition_name}): "
                    f"{src_lang} -> {tgt_lang}..."
                )

                result = compute_canonical_alignment(
                    df_canon_parallel,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    tokenizer=tokenizer,
                )

                alignability_records_c.append(
                    {
                        "target_lang": tgt_lang,
                        "script": SCRIPT_MAP.get(
                            tgt_lang,
                            "Unknown",
                        ),
                        **result,
                    }
                )

            alignability_df_c = pd.DataFrame(alignability_records_c)

            # -----------------------------------------------------------------
            # Canonical report
            # -----------------------------------------------------------------

            print("\n--- Canonical Token Statistics " f"({condition_name}) ---")

            print(
                fertility_df_c[
                    [
                        "script",
                        "tokens_per_word",
                        "tokens_per_char",
                        "vocab_used",
                        "n_sentences",
                        "n_words",
                        "n_tokens",
                    ]
                ].round(3)
            )

            print(
                "\n--- Canonical Subword Alignment "
                f"({condition_name}, from English) ---"
            )

            print(alignability_df_c.to_string(index=False))

            # -----------------------------------------------------------------
            # Canonical CSV rows
            # -----------------------------------------------------------------

            fertility_c = fertility_df_c.reset_index()

            fertility_c["model"] = model_slug
            fertility_c["source"] = source_label_value

            all_fertility_rows.append(fertility_c)

            align_c = alignability_df_c.copy()

            align_c["model"] = model_slug
            align_c["source"] = source_label_value

            all_alignability_rows.append(align_c)

            jac_c = overlap_df_c.reset_index(names="language_1").melt(
                id_vars="language_1",
                var_name="language_2",
                value_name="jaccard",
            )

            jac_c["model"] = model_slug
            jac_c["source"] = source_label_value

            all_jaccard_rows.append(jac_c)

        # =====================================================================
        # Free tokenizer before loading next model
        # =====================================================================

        del tokenizer
        gc.collect()

    # =========================================================================
    # CSV Export
    # =========================================================================

    fertility_stats = pd.concat(
        all_fertility_rows,
        ignore_index=True,
    )

    alignability_stats = pd.concat(
        all_alignability_rows,
        ignore_index=True,
    )

    jaccard_stats = pd.concat(
        all_jaccard_rows,
        ignore_index=True,
    )

    fertility_stats.to_csv(
        "fertility_stats.csv",
        index=False,
    )

    alignability_stats.to_csv(
        "alignability_stats.csv",
        index=False,
    )

    jaccard_stats.to_csv(
        "jaccard_stats.csv",
        index=False,
    )

    print("\nSaved:")

    print(f"  fertility_stats.csv " f"({len(fertility_stats)} rows)")

    print(f"  alignability_stats.csv " f"({len(alignability_stats)} rows)")

    print(f"  jaccard_stats.csv " f"({len(jaccard_stats)} rows)")

    # =========================================================================
    # Summary plots
    # =========================================================================

    plot_summary_visualizations(
        fertility_stats,
        jaccard_stats,
        alignability_stats,
        src_lang=src_lang,
    )


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
