"""
run_baseline_norule.py -- no-rule baseline generation.

Lives in canonical/baseline/. Run from the repo root (PYTHONPATH=.).

What it does
  Reuses ModelRunner from inference.py UNCHANGED (same model loading, chat
  template, sampling, batching, per-batch checkpointing, same config file), and
  only changes which prompts get generated:

    * system prompt  -> persona only (the row's `context` column, e.g. "You are
                        an assistant."). The rule, the status word and the
                        grammar_type wrapper ("Never...", "You must ensure:",
                        "Please make sure...") all live after it in `system`,
                        so they are all removed. Falls back to "first sentence
                        of system" if `context` is missing.
    * user query     -> unchanged, but NEUTRAL (L0) queries only. L1-L4 queries
                        contain pressure text like "ignore that rule", which
                        makes no sense when there is no rule.
    * rows           -> ACTIVE-status rows only, de-duplicated on
                        (language, category, topic, pressure_level). With no rule,
                        the active/cancelled pair, the status words and the
                        grammar types all collapse to the same prompt, so
                        generating them separately would just repeat identical
                        prompts. Keeping the active row means each baseline row
                        still carries the ACTIVE checker, so a baseline response
                        can later be scored as "would this have satisfied the
                        rule if it were active".

  Join baseline -> rule rows later on (language, category, topic, pressure_level).
  (If translations differ in wording between rows of one cell, the baseline
  keeps the first row's wording; the dry run reports how many cells differ.)
  Baseline ids look like  norule::<original id>.  The other metadata columns
  (rule_status, grammar_type, pair_type, ...) are inherited from the source row
  only so the output schema stays valid; they are NOT meaningful for the
  baseline.

Safety rails
  * Never reads hf_result_repo from the config. You must pass --result-repo
    explicitly, and it must differ from the config's repo. (inference.py skips
    any model/language already present in the repo, so pointing at the main
    results repo would silently skip everything.)
  * --limit never pushes and uses a separate checkpoint dir.
  * Stops before loading a model if the stripped system prompts look wrong.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from dotenv import load_dotenv

from canonical.evaluation.inference import ModelGenerationConfig, ModelRunner
from canonical.model.dataset import (
    ACTIVE_STATUSES,
    CrossLingualRuleFollowingDataset,
    DatasetLanguageCode,
    DatasetSource,
    HFDataHelper,
)

load_dotenv()

NORULE_PREFIX = "norule::"
# One baseline prompt per (language, category, topic, pressure_level) cell. The
# English source has exactly one user_query per cell; the translations are made
# per row, so wording could differ between rows of the same cell. Keying on the
# cell (not the text) keeps one prompt per cell either way, and build_baseline_df
# reports how many cells had differing wording.
DEDUP_KEYS = ["language", "category", "topic", "pressure_level"]
REQUIRED_COLS = DEDUP_KEYS + ["id", "system", "user_query"]
MAX_UNIQUE_PERSONAS_PER_LANG = 3
MAX_PERSONA_CHARS = 150

# First sentence ends at: Latin ".", CJK "。", Devanagari danda "।", Urdu "۔",
# Ethiopic full stop "።" (Amharic). Only a fallback; `context` is used first.
_SENTENCE_END = re.compile(r"[.。।۔።]")


def persona_only(system: str) -> str:
    """Keep only the first sentence of the system prompt (the persona line,
    e.g. 'Du bist ein Assistent.'); drop 'Rule: ...' and 'Rule status: ...'."""
    system = (system or "").strip()
    m = _SENTENCE_END.search(system)
    return system[: m.end()].strip() if m else system


def _norm(v) -> str:
    return str(getattr(v, "value", v))


def build_baseline_df(df: pd.DataFrame, pressure_levels=("L0",)) -> Tuple[pd.DataFrame, dict]:
    """df must already be language-filtered AND active-status-filtered."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"dataset is missing expected columns {missing}; got {list(df.columns)}"
        )
    stats = {"input_rows": len(df)}
    df = df[df["pressure_level"].map(_norm).isin(set(pressure_levels))]
    stats["after_pressure_filter"] = len(df)
    n_variants = df.groupby(DEDUP_KEYS)["user_query"].nunique()
    stats["cells"] = int(len(n_variants))
    stats["cells_with_differing_query_text"] = int((n_variants > 1).sum())
    df = df.sort_values("id", kind="stable")
    df = df.drop_duplicates(subset=DEDUP_KEYS, keep="first").reset_index(drop=True)
    stats["after_dedupe"] = len(df)
    df = df.copy()
    df["id"] = NORULE_PREFIX + df["id"].astype(str)

    # Preferred: the dataset keeps the persona in its own `context` column and
    # builds  system = context + " " + rule_text  (see RulePair), so use it
    # directly when system really starts with it. Fallback: first-sentence guess.
    def _strip(row) -> Tuple[str, bool]:
        ctx = str(row["context"]).strip() if "context" in row and row["context"] else ""
        sysm = str(row["system"])
        if ctx and sysm.strip().startswith(ctx):
            return ctx, True
        return persona_only(sysm), False

    pairs = [_strip(r) for _, r in df.iterrows()]
    df["system"] = [p for p, _ in pairs]
    stats["persona_from_context"] = sum(1 for _, ok in pairs if ok)
    stats["persona_from_heuristic"] = sum(1 for _, ok in pairs if not ok)
    return df, stats


def check_personas(df: pd.DataFrame, lang: str) -> List[str]:
    """Return a list of problems with the stripped system prompts. A correct
    strip collapses to one (maybe a few) persona sentences per language."""
    problems = []
    uniq = list(df["system"].unique())
    print(f"  distinct stripped system prompts [{lang}]: {len(uniq)}")
    for s in uniq[:5]:
        print(f"    - {s!r}")
    if not uniq:
        problems.append("no rows left after filtering")
    if len(uniq) > MAX_UNIQUE_PERSONAS_PER_LANG:
        problems.append(
            f"{len(uniq)} distinct system prompts (expected <= "
            f"{MAX_UNIQUE_PERSONAS_PER_LANG}); the first-sentence strip probably "
            "isn't isolating the persona line"
        )
    if any(len(s) > MAX_PERSONA_CHARS for s in uniq):
        problems.append(f"a stripped system prompt is longer than {MAX_PERSONA_CHARS} chars")
    if any(not s for s in uniq):
        problems.append("a stripped system prompt is empty")
    return problems


def load_canonical_dataset(config, lang_code) -> CrossLingualRuleFollowingDataset:
    """Same per-language loading logic as run() in inference.py."""
    ds_config = config.dataset_config
    if ds_config.data_dir:
        lang_file = f"{ds_config.data_dir}/{lang_code.value}/test.jsonl"
        update = (
            {"data_files": lang_file}
            if ds_config.source == DatasetSource.hf
            else {"url": lang_file}
        )
        ds_config = ds_config.model_copy(update=update)
    return CrossLingualRuleFollowingDataset(ds_config)


def build_for_language(config, lang_code, pressure_levels):
    ds = load_canonical_dataset(config, lang_code).subset(language=lang_code.value)
    ds = ds.subset(rule_status=[status.value for status in ACTIVE_STATUSES])
    return build_baseline_df(ds.df, pressure_levels)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="No-rule baseline generation")
    ap.add_argument("--hyperparameter-file", required=True,
                    help="an EXISTING generation config (Llama or Qwen); all "
                         "generation settings are inherited from it")
    ap.add_argument("--result-repo", default=None,
                    help="HF dataset repo for baseline responses. Must be a NEW "
                         "repo, not the main results repo. Required unless --no-push")
    ap.add_argument("--checkpoint-dir", default="/workspace/baseline_ckpt")
    ap.add_argument("--out-dir", default="/workspace/baseline_out",
                    help="local copy of the responses is always written here")
    ap.add_argument("--langs", default=None,
                    help="comma-separated language codes, e.g. en,de (default: config's)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model ids (default: config's)")
    ap.add_argument("--pressure-levels", default="L0",
                    help="comma-separated pressure levels to keep (default L0 = neutral)")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--generation-batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: only the first N rows; never pushes")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + inspect prompts only; no model, no generation")
    ap.add_argument("--allow-persona-warnings", action="store_true",
                    help="continue even if the stripped system prompts look wrong")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.set_grad_enabled(False)

    with open(args.hyperparameter_file) as f:
        config = ModelGenerationConfig.model_validate_json(f.read())

    main_repo = getattr(config, "hf_result_repo", None)
    config.run_inference_response = True
    config.run_inference_activations = False
    if args.n_samples is not None:
        if args.n_samples < 1:
            raise ValueError("--n-samples must be at least 1")
        config.n_samples = args.n_samples
    if args.generation_batch_size is not None:
        config.generation_batch_size = args.generation_batch_size

    ckpt_dir = Path(args.checkpoint_dir)
    if args.limit:
        ckpt_dir = ckpt_dir / "smoke"  # keep smoke rows out of the real run's checkpoints
    config.checkpoint_dir = str(ckpt_dir)

    push = (not args.no_push) and (not args.limit) and (not args.dry_run)
    if push:
        if not args.result_repo:
            raise SystemExit("Pass --result-repo <NEW HF dataset repo>, or --no-push.")
        if main_repo and args.result_repo == main_repo:
            raise SystemExit(
                f"--result-repo equals the config's hf_result_repo ({main_repo}). "
                "inference.py skips model/language pairs already in the repo, so "
                "this would skip everything or mix baseline with rule data. Use a new repo."
            )
    config.hf_result_repo = args.result_repo if push else None
    config.push_to_hf = push

    langs = (
        [DatasetLanguageCode(x.strip()) for x in args.langs.split(",")]
        if args.langs
        else [DatasetLanguageCode(getattr(c, "value", c)) for c in config.language_codes]
    )
    models = (
        [m.strip() for m in args.models.split(",")] if args.models else list(config.model_ids)
    )
    pressure_levels = tuple(x.strip() for x in args.pressure_levels.split(","))

    print(f"models={models} langs={[l.value for l in langs]} pressure_levels={pressure_levels}")
    print(f"n_samples={config.n_samples} temperature={config.temperature} "
          f"max_new_tokens={config.max_new_tokens} batch={config.generation_batch_size}")
    print(f"push={push} result_repo={config.hf_result_repo} checkpoint_dir={config.checkpoint_dir}")

    # ---- build + inspect prompts for every language BEFORE loading any model ----
    built = {}
    all_ok = True
    for lang in langs:
        df, stats = build_for_language(config, lang, pressure_levels)
        print(f"\n[{lang.value}] rows: {stats}")
        problems = check_personas(df, lang.value)
        for p in problems:
            print(f"  !! PROBLEM: {p}")
        all_ok = all_ok and not problems
        if args.limit:
            df = df.head(args.limit)
        built[lang] = df
        if args.dry_run:
            for _, r in df.head(3).iterrows():
                print(f"  sample id={r['id']}\n    system: {r['system']!r}\n    user:   {r['user_query']!r}")

    if args.dry_run:
        print("\nDry run done. Nothing generated.")
        return
    if not all_ok and not args.allow_persona_warnings:
        raise SystemExit(
            "Stripped system prompts look wrong (see PROBLEM lines above). Fix the "
            "stripping, or re-run with --allow-persona-warnings if you checked and it's fine."
        )

    helper = HFDataHelper(config.hf_result_repo) if push else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_idx, model_name in enumerate(models, start=1):
        print(f"\n{'=' * 70}\n[model {model_idx}/{len(models)}] {model_name}\n{'=' * 70}", flush=True)
        runner = ModelRunner(config)
        runner.load(model_id=model_name)

        for lang in langs:
            t0 = time.time()
            if helper and helper.exists(model_id=model_name, lang_code=lang):
                print(f"Skipping {model_name}/{lang.value}: already in {config.hf_result_repo}")
                continue

            responses = runner.generate_response(dataset=built[lang], lang_code=lang)
            out_df = pd.DataFrame(responses)

            slug = model_name.replace("/", "__")
            local_path = out_dir / f"{slug}_{lang.value}_norule_responses.jsonl"
            with open(local_path, "w", encoding="utf-8") as f:
                for row in responses:
                    f.write(json.dumps(row, ensure_ascii=False,
                                       default=lambda o: getattr(o, "value", str(o))) + "\n")
            print(f"Wrote local copy: {local_path} ({len(out_df)} rows)")

            if helper:
                helper.upload(df=out_df, model_id=model_name, lang_code=lang)
                runner.clear_checkpoint(lang, "responses")
                print(f"Uploaded to {config.hf_result_repo}")
            else:
                print("Not pushing (--no-push / --limit); checkpoint kept.")

            n_empty = int((out_df["response"].astype(str).str.strip() == "").sum())
            print(f"--- {model_name}/{lang.value}: {len(out_df)} responses, "
                  f"{n_empty} empty, mean length "
                  f"{out_df['response'].astype(str).str.len().mean():.0f} chars, "
                  f"{(time.time() - t0) / 60:.1f} min ---", flush=True)

        del runner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
