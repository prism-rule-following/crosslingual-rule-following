"""
generate_grammar_variants.py
============================
Expand the imperative-only dataset into matched grammar triples
(imperative + modal_obligation + polite_asking) via LLM re-registration.

This script is a PURE GENERATOR. It produces rows and writes them; it does not
do model-based verification and does not stamp per-row provenance. All
verification (schema load, deterministic gate, second-model equivalence) is the
job of verify_dataset.py, run separately on the output.


Outputs:
  judgment_rules_expanded.json   (imperative sources + accepted variants)
  rejects.jsonl                  (every rejected attempt + reasons)
  generation_report.json         (coverage, balance, pass rates)

Env:
  OPENROUTER_API_KEY   required to actually call.
  DATASET_GEN_MODEL   default 'anthropic/claude-3.5-sonnet'
Dry-run without a key: pass --dry-run to exercise gate/assembly on mock output.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from prompts import build_batch_transform_prompt, cell_id
from quality_gate import check_clause
from model.dataset import (
    Dataset,
    GenerationRecord,
    RulePair,
    build_id,
    carry_metadata,
    compute_row_counts,
)

# Must run before any os.environ.get() below — this script may be imported
# (e.g. by run_pipeline.py) rather than run standalone, and module-level
# constants are read at import time, before a caller's own load_dotenv()
# (if any) would run.
load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent

SRC = DATA_DIR / "v2/judgment_rules.json"
OUT = DATA_DIR / "v2/judgment_rules_expanded.json"
REJECTS = DATA_DIR / "v2/rejects.jsonl"
REPORT = DATA_DIR / "v2/generation_report.json"

DATASET_GEN_MODEL = os.environ.get("DATASET_GEN_MODEL") or "anthropic/claude-3.5-sonnet"
PROMPT_VERSION = "grammar-transform-v1"


# --------------------------------------------------------------------------- #
# OpenRouter SDK call
# --------------------------------------------------------------------------- #
from openrouter import OpenRouter, errors


def _client() -> OpenRouter:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenRouter(api_key=key)


def call_json(
    client: OpenRouter,
    model: str,
    messages: list,
    max_retries: int = 3,
    max_tokens: int = 1200,
) -> Optional[dict]:
    """Call OpenRouter chat completion and parse strict JSON from the reply."""
    last = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.send(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                text = text[text.find("{") :]
            return json.loads(text[text.find("{") : text.rfind("}") + 1])
        except errors.OpenRouterError as e:
            # rate limit, bad model, etc. — body carries the useful detail
            last = f"HTTP {e.status_code}: {e.body[:200]}"
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa - network, JSON, or shape errors
            last = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  ! JSON call failed after {max_retries}: {last}", file=sys.stderr)
    return None


# --------------------------------------------------------------------------- #
# Row assembly
# --------------------------------------------------------------------------- #
def cell_key(row: dict) -> tuple:
    """Group rows that must share ONE generated clause."""
    return (row["category"], row["topic"])


def assemble_row(source: dict, grammar: str, clause: str) -> dict:
    """Build a full row for ONE pair_type sibling from a shared clause."""
    clause = clause.strip().rstrip(" .!?")
    row = copy.deepcopy(source)
    row["id"] = build_id(
        source["category"],
        source["topic"],
        source["active_status"],
        grammar,
        source["pressure_level"],
    )
    row["grammar_type"] = grammar
    row["rule_clause"] = clause
    row["rule_text"] = f"{clause}. Rule status: {source['active_status']}."
    row["non_rule_text"] = f"{clause}. Rule status: {source['revoked_status']}."
    row["system_rule"] = f"{source['context']} Rule: {row['rule_text']}"
    row["system_non_rule"] = f"{source['context']} Rule: {row['non_rule_text']}"

    for side in ("active_checker", "revoked_checker"):
        ch = row[side]
        if ch.get("checker_type") == "llm_judge" and "rule_clause" in ch.get(
            "instruction", ""
        ):
            ch["instruction"] = ch["instruction"].replace(source["rule_clause"], clause)
        rub = ch.get("rubric", {})
        for k, v in list(rub.items()):
            if isinstance(v, str) and source["rule_clause"] in v:
                rub[k] = v.replace(source["rule_clause"], clause)

    return row


def log_reject(fh, source_id: str, grammar: str, stage: str, reasons, payload):
    fh.write(
        json.dumps(
            {
                "source_id": source_id,
                "grammar": grammar,
                "stage": stage,
                "reasons": reasons,
                "payload": payload,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        + "\n"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="No API calls; mock the generator to exercise the gate.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only N scenario cells (0 = all). A cell is one "
        "(category, topic) group, covering all its pressure_level x "
        "pair_type siblings (rule_clause/context don't vary across either).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Scenario cells packed into one model call, per grammar "
        "(0 = one call for every cell). Each call still only costs "
        "output tokens proportional to its batch, not one call per cell.",
    )
    args = ap.parse_args(argv)

    data = json.loads(SRC.read_text())
    sources = [p for p in data["pairs"] if p["grammar_type"] == "imperative"]

    by_cell: dict = {}
    for p in sources:
        by_cell.setdefault(cell_key(p), []).append(p)
    cells = sorted(by_cell.items(), key=lambda kv: kv[0])
    if args.limit:
        cells = cells[: args.limit]

    cell_list = [(c, sorted(sibs, key=lambda s: s["id"])) for c, sibs in cells]

    batch_size = args.batch_size or len(cell_list) or 1
    batches = [
        cell_list[i : i + batch_size] for i in range(0, len(cell_list), batch_size)
    ]

    client = None if args.dry_run else _client()

    accepted = list(data["pairs"])  # keep the imperative originals unchanged
    stats = Counter()
    stage_fail = Counter()

    with REJECTS.open("w") as rj:
        for grammar in ("modal_obligation", "polite_asking"):
            for batch in batches:
                reps = [sibs[0] for _, sibs in batch]
                stats["model_calls"] += 1
                stats["cells_attempted"] += len(reps)
                messages = build_batch_transform_prompt(reps, grammar)
                out_tokens = 300 + 150 * len(reps)

                # ---- generate: one call for the whole batch ----
                if args.dry_run:
                    mapping = {
                        cell_id(r["category"], r["topic"]): _mock_clause(r, grammar)
                        for r in reps
                    }
                else:
                    gen = call_json(
                        client, DATASET_GEN_MODEL, messages, max_tokens=out_tokens
                    )
                    mapping = gen if isinstance(gen, dict) else {}

                # ---- return-side gate, per item ----
                clauses: dict = {}
                failed: list = []  # [(rep, reasons)]
                for rep in reps:
                    key = cell_id(rep["category"], rep["topic"])
                    clause = mapping.get(key)
                    if not clause:
                        failed.append((rep, ["missing from batch response"]))
                        continue
                    res = check_clause(clause, rep, grammar)
                    if res.ok:
                        clauses[key] = clause
                    else:
                        failed.append((rep, res.reasons))

                # ---- one batched repair call for whatever failed ----
                if failed and not args.dry_run:
                    repair_reps = [rep for rep, _ in failed]
                    reasons_block = "\n".join(
                        f"- {cell_id(rep['category'], rep['topic'])}: "
                        + "; ".join(reasons)
                        for rep, reasons in failed
                    )
                    repair_messages = build_batch_transform_prompt(
                        repair_reps, grammar
                    ) + [
                        {
                            "role": "user",
                            "content": "Your previous output failed these "
                            "checks for these keys — return corrected STRICT "
                            "JSON for ONLY these keys:\n" + reasons_block,
                        }
                    ]
                    stats["model_calls"] += 1
                    gen2 = call_json(
                        client,
                        DATASET_GEN_MODEL,
                        repair_messages,
                        max_tokens=300 + 150 * len(repair_reps),
                    )
                    mapping2 = gen2 if isinstance(gen2, dict) else {}
                    still_failed = []
                    for rep, reasons in failed:
                        key = cell_id(rep["category"], rep["topic"])
                        clause2 = mapping2.get(key)
                        res2 = check_clause(clause2, rep, grammar) if clause2 else None
                        if res2 and res2.ok:
                            clauses[key] = clause2
                        else:
                            still_failed.append((rep, reasons))
                    failed = still_failed

                for rep, reasons in failed:
                    stage_fail["gate"] += 1
                    log_reject(
                        rj,
                        rep["id"],
                        grammar,
                        "gate",
                        reasons,
                        mapping.get(cell_id(rep["category"], rep["topic"])),
                    )

                stats["cells_accepted"] += len(clauses)

                # ---- assemble + schema-validate EVERY pair_type sibling ----
                for _, sibs in batch:
                    key = cell_id(sibs[0]["category"], sibs[0]["topic"])
                    clause = clauses.get(key)
                    if clause is None:
                        continue
                    for sib in sibs:
                        row = assemble_row(sib, grammar, clause)
                        try:
                            RulePair.model_validate(row)
                        except Exception as e:
                            stage_fail["schema"] += 1
                            log_reject(
                                rj,
                                sib["id"],
                                grammar,
                                "schema",
                                [str(e).splitlines()[-1]],
                                row,
                            )
                            continue
                        accepted.append(row)
                        stats["rows_accepted"] += 1

            print(
                f"  {grammar}: cells_accepted={stats['cells_accepted']} "
                f"rows_accepted={stats['rows_accepted']} so far "
                f"({stats['model_calls']} model calls total)"
            )

    # ---- write outputs ----
    ds = Dataset(
        metadata=carry_metadata(
            data.get("metadata", {}),
            total=len(accepted),
            note=data["metadata"]["note"]
            + " | grammar variants added via matched-triple transform "
            f"({PROMPT_VERSION}).",
            generation=GenerationRecord(
                all_synthetic=True,
                seed_method="prior_synthetic_draft",
                transform_method="grammar_transform",
                gen_model="mock" if args.dry_run else DATASET_GEN_MODEL,
                prompt_version=PROMPT_VERSION,
                ts=datetime.now(timezone.utc).isoformat(),
            ),
            counts=compute_row_counts(accepted),
        ),
        pairs=[RulePair.model_validate(r) for r in accepted],
    )
    OUT.write_text(ds.model_dump_json(indent=2))

    report = _report(accepted, stats, stage_fail)
    REPORT.write_text(json.dumps(report, indent=2))

    print(
        f"\nDone. model_calls={stats['model_calls']}; "
        f"cells_accepted={stats['cells_accepted']} / "
        f"cells_attempted={stats['cells_attempted']}; "
        f"rows_accepted={stats['rows_accepted']}"
    )
    print(f"stage failures: {dict(stage_fail)}")
    print(f"wrote {OUT}, {REJECTS}, {REPORT}")


def _report(accepted, stats, stage_fail) -> dict:
    def dist(key):
        return dict(Counter(r[key] for r in accepted))

    from collections import defaultdict

    grammars_by_scenario = defaultdict(set)
    for r in accepted:
        key = (r["category"], r["topic"], r["pressure_level"], r["pair_type"])
        grammars_by_scenario[key].add(r["grammar_type"])
    complete_triples = sum(
        1
        for gs in grammars_by_scenario.values()
        if {"imperative", "modal_obligation", "polite_asking"} <= gs
    )

    return {
        "totals": dict(stats),
        "stage_failures": dict(stage_fail),
        "pass_rate": round(
            stats["cells_accepted"] / max(stats["cells_attempted"], 1), 4
        ),
        "grammar_distribution": dist("grammar_type"),
        "category_distribution": dist("category"),
        "topic_distribution": dist("topic"),
        "pressure_distribution": dist("pressure_level"),
        "scenarios_with_complete_triples": complete_triples,
        "gen_model": DATASET_GEN_MODEL,
        "prompt_version": PROMPT_VERSION,
        "note": "Verification (schema, deterministic, model) is performed "
        "separately by verify_dataset.py.",
        "TODO_for_paper": [
            "Run verify_dataset.py --with-model over this output.",
            "Human-validate a stratified sample (>=100 rows); report kappa.",
            "Report judge-human agreement on the same sample.",
            "Attach datasheet + license + synthetic-content disclaimer.",
        ],
    }


def _mock_clause(src: dict, grammar: str) -> str:
    """Deterministic fake clause for --dry-run gate testing."""
    core = src["rule_clause"]
    if grammar == "modal_obligation":
        # naive re-register: front with a modal
        if core.lower().startswith("never"):
            clause = "You must not " + core[len("never") :].strip()
        elif core.lower().startswith("only"):
            clause = "You must only " + core[len("only") :].strip()
        else:
            clause = "You must ensure that you " + core[0].lower() + core[1:]
    else:  # polite_asking
        clause = "Please make sure you " + core[0].lower() + core[1:]
    return clause.rstrip(".")


if __name__ == "__main__":
    main()
