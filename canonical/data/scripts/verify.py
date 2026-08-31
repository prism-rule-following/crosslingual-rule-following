"""
verify_dataset.py — standalone verification / filtering pass over a dataset.

Runs independently of generation. Given any dataset JSON (old seeds, new
variants, or the combined file), it performs three stages and writes a clean
filtered file plus a rejects log and a report.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import ValidationError

import model.dataset as schema
from model.dataset import RulePair, Dataset, carry_metadata, compute_row_counts
from quality_gate import (
    check_selfcontained,
    check_against_reference,
    semantic_preservation_probe_batch,
)

# Must run before any os.environ.get() below — this script may be imported
# (e.g. by run_pipeline.py) rather than run standalone, and module-level
# constants are read at import time, before a caller's own load_dotenv()
# (if any) would run.
load_dotenv()

DATASET_VERIFY_MODEL = os.environ.get("DATASET_VERIFY_MODEL") or "openai/gpt-4o"

DATA_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Stage 0 — id / row uniqueness
# --------------------------------------------------------------------------- #
def verify_uniqueness(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Reject rows whose id collides with an earlier row, or whose full
    content (every field but id) duplicates an earlier row — a copy-paste
    row that only got its id changed, or an id typo that collided with an
    unrelated row."""
    passed, rejected = [], []
    seen_ids: Dict[str, str] = {}
    seen_content: Dict[str, str] = {}
    for r in rows:
        rid = r.get("id", "<no id>")
        reasons: List[str] = []

        if rid in seen_ids:
            reasons.append(f"duplicate id: '{rid}' already used by an earlier row")
        else:
            seen_ids[rid] = rid

        content_key = json.dumps(
            {k: v for k, v in r.items() if k != "id"}, sort_keys=True, default=str
        )
        if content_key in seen_content:
            reasons.append(
                f"duplicate row content: identical to row id="
                f"'{seen_content[content_key]}' (all fields but id match)"
            )
        else:
            seen_content[content_key] = rid

        if reasons:
            rejected.append({"id": rid, "stage": "uniqueness", "reasons": reasons})
        else:
            passed.append(r)
    return passed, rejected


# --------------------------------------------------------------------------- #
# Stage 1 — schema load
# --------------------------------------------------------------------------- #
def verify_schema(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    passed, rejected = [], []
    for r in rows:
        try:
            RulePair.model_validate(r)
            passed.append(r)
        except ValidationError as e:
            first = e.errors()[0]
            rejected.append(
                {
                    "id": r.get("id", "<no id>"),
                    "stage": "schema",
                    "reasons": [
                        f"{'.'.join(str(x) for x in first.get('loc', []))}: "
                        f"{first.get('msg')}"
                    ],
                    "n_errors": len(e.errors()),
                }
            )
    return passed, rejected


# --------------------------------------------------------------------------- #
# Stage 2 — deterministic quality gate
# --------------------------------------------------------------------------- #
def _reconstruct_gen_dict(row: dict) -> dict:
    """Adapt a full row into the shape check_transformed expects."""
    return {
        "grammar_rule_clause": row["rule_clause"],
        "rule_text": row["rule_text"],
        "non_rule_text": row["non_rule_text"],
        "system_rule": row["system_rule"],
        "system_non_rule": row["system_non_rule"],
    }


def _checker_pairing_ok(row: dict) -> List[str]:
    """Source-independent structural checks that apply to every row."""
    problems = []
    ac, rc = row.get("active_checker"), row.get("revoked_checker")
    if not ac or not rc:
        return ["missing active_checker or revoked_checker"]
    if ac.get("binds") is not True:
        problems.append("active_checker.binds must be True")
    if rc.get("binds") is not False:
        problems.append("revoked_checker.binds must be False")
    if ac.get("rule_status") != row.get("active_status"):
        problems.append("active_checker.rule_status != active_status")
    if rc.get("rule_status") != row.get("revoked_status"):
        problems.append("revoked_checker.rule_status != revoked_status")
    if ac.get("checker_type") != rc.get("checker_type"):
        problems.append("active/revoked checker_type mismatch")
    return problems


def scenario_key(r: dict) -> tuple:
    return (
        r.get("category"),
        r.get("topic"),
        r.get("pressure_level"),
        r.get("pair_type"),
    )


def stage_deterministic(
    rows: List[dict],
    groups: Dict[tuple, List[dict]],
) -> Tuple[List[dict], List[dict]]:
    passed, rejected = [], []
    for r in rows:
        reasons: List[str] = []

        # (a) structural checks
        reasons += _checker_pairing_ok(r)

        # (b) self-contained checks — every row, keyed off its OWN grammar_type
        grammar = r.get("grammar_type")
        res = check_selfcontained(_reconstruct_gen_dict(r), r, grammar)
        if not res.ok:
            reasons += res.reasons

        # (c) comparative checks — against any other grammar variant in the same
        #     scenario (they share rule semantics). Grammar-agnostic: the
        #     reference is whichever group-mate has a DIFFERENT grammar_type.
        group = groups.get(scenario_key(r), [])
        reference = next(
            (o for o in group if o is not r and o.get("grammar_type") != grammar), None
        )
        if reference is not None:
            cref = check_against_reference(_reconstruct_gen_dict(r), reference)
            if not cref.ok:
                reasons += cref.reasons

        # (d) distinctness within the scenario group — grammar-agnostic
        #     any two rows in the same scenario with DIFFERENT grammar_type
        #     must have different clauses (register actually took), and any
        #     two with the SAME grammar_type must not duplicate.
        group = groups.get(scenario_key(r), [])
        my_clause = (r.get("rule_clause") or "").strip().lower()
        for other in group:
            if other is r:
                continue
            oc = (other.get("rule_clause") or "").strip().lower()
            if oc != my_clause:
                continue
            if other.get("grammar_type") != grammar:
                reasons.append(
                    f"clause identical to a '{other.get('grammar_type')}' row "
                    f"in the same scenario (register did not take)"
                )
            else:
                reasons.append(
                    f"duplicate clause of another '{grammar}' row "
                    f"in the same scenario"
                )

        if reasons:
            rejected.append(
                {
                    "id": r.get("id"),
                    "stage": "deterministic",
                    "reasons": sorted(set(reasons)),
                }
            )
        else:
            passed.append(r)
    return passed, rejected


# --------------------------------------------------------------------------- #
# Stage 3 — verify model (optional)
# --------------------------------------------------------------------------- #
def _client():
    from openrouter import OpenRouter

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set (needed for --with-model)")
    return OpenRouter(api_key=key)


def _call_json(
    client, model: str, messages: list, retries: int = 3, max_tokens: int = 300
) -> Optional[dict]:
    from openrouter import errors

    last = None
    for i in range(retries):
        try:
            resp = client.chat.send(
                model=model, messages=messages, temperature=0.0, max_tokens=max_tokens
            )
            t = (resp.choices[0].message.content or "").strip()
            if t.startswith("```"):
                t = t.strip("`")
                t = t[t.find("{") :]
            return json.loads(t[t.find("{") : t.rfind("}") + 1])
        except errors.OpenRouterError as e:
            last = f"HTTP {e.status_code}: {e.body[:200]}"
            time.sleep(1.2 * (i + 1))
        except Exception as e:  # noqa
            last = e
            time.sleep(1.2 * (i + 1))
    print(f"  ! verify call failed: {last}", file=sys.stderr)
    return None


def stage_verify_model(
    rows: List[dict],
    groups: Dict[tuple, List[dict]],
    batch_size: int = 20,
) -> Tuple[List[dict], List[dict]]:
    client = _client()
    passed, rejected = [], []
    # choose a stable reference per group: lexicographically smallest id
    group_ref = {k: min(v, key=lambda x: x["id"]) for k, v in groups.items()}

    to_check: List[Tuple[dict, dict]] = []  # (row, reference)
    for r in rows:
        ref = group_ref.get(scenario_key(r))
        # nothing to compare against (singleton group, or r is the reference)
        if ref is None or ref is r or ref["id"] == r["id"]:
            passed.append(r)
        else:
            to_check.append((r, ref))

    for i in range(0, len(to_check), batch_size):
        batch = to_check[i : i + batch_size]
        items = [(r["id"], ref, _reconstruct_gen_dict(r)) for r, ref in batch]
        probe = semantic_preservation_probe_batch(items)
        verdicts = _call_json(
            client, DATASET_VERIFY_MODEL, probe, max_tokens=150 + 80 * len(batch)
        )
        verdicts = verdicts if isinstance(verdicts, dict) else {}

        for r, ref in batch:
            v = verdicts.get(r["id"])
            if not v or not v.get("equivalent", False):
                rejected.append(
                    {
                        "id": r.get("id"),
                        "stage": "verify_model",
                        "reference_id": ref["id"],
                        "reasons": [
                            v.get("reason") if v else "verifier no-response/missing key"
                        ],
                    }
                )
            else:
                r.setdefault("generation_metadata", {})
                r["generation_metadata"]["post_verify"] = {
                    "verify_model": DATASET_VERIFY_MODEL,
                    "reference_id": ref["id"],
                    "verdict": v,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                passed.append(r)
    return passed, rejected


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="dataset JSON to verify")
    ap.add_argument(
        "--with-model",
        action="store_true",
        help="run Stage 3 (second-model equivalence check).",
    )
    ap.add_argument(
        "--verify-batch-size",
        type=int,
        default=20,
        help="(reference, candidate) pairs packed into one Stage 3 model "
        "call (0 = one call for every pair).",
    )
    ap.add_argument("--out", default=None, help="output path for clean file.")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    data = json.loads(inp.read_text())
    rows = data["pairs"]

    rejects_all: List[dict] = []
    counts = Counter()
    counts["input"] = len(rows)

    # Stage 0
    rows, rej = verify_uniqueness(rows)
    rejects_all += rej
    counts["after_uniqueness"] = len(rows)
    counts["fail_uniqueness"] = len(rej)

    # group rows (post-uniqueness, so colliding/duplicate rows never
    # distort scenario groups used by stages 2-3)
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        groups.setdefault(scenario_key(r), []).append(r)

    # Stage 1
    rows, rej = verify_schema(rows)
    rejects_all += rej
    counts["after_schema"] = len(rows)
    counts["fail_schema"] = len(rej)

    # Stage 2
    rows, rej = stage_deterministic(rows, groups)
    rejects_all += rej
    counts["after_deterministic"] = len(rows)
    counts["fail_deterministic"] = len(rej)

    # Stage 3 (optional)
    if args.with_model:
        rows, rej = stage_verify_model(
            rows, groups, batch_size=args.verify_batch_size or len(rows) or 1
        )
        rejects_all += rej
        counts["after_verify_model"] = len(rows)
        counts["fail_verify_model"] = len(rej)

    # write clean file (re-validate through Dataset for a final guarantee)
    clean = Dataset(
        metadata=carry_metadata(
            data.get("metadata", {}),
            total=len(rows),
            note=data.get("metadata", {}).get("note", "") + " | verified.",
            counts=compute_row_counts(rows),
        ),
        pairs=[RulePair.model_validate(r) for r in rows],
    )
    out = Path(args.out) if args.out else inp.with_suffix(".verified.json")
    out.write_text(clean.model_dump_json(indent=2))

    rejects_path = DATA_DIR / "v2/verify_rejects.jsonl"
    rejects_path.write_text("\n".join(json.dumps(x) for x in rejects_all))

    report = {
        "input_file": str(inp),
        "counts": dict(counts),
        "pass_rate": round(len(rows) / max(counts["input"], 1), 4),
        "stages_run": ["uniqueness", "schema", "deterministic"]
        + (["verify_model"] if args.with_model else []),
        "verify_model": DATASET_VERIFY_MODEL if args.with_model else None,
        "rejects_by_stage": dict(Counter(x["stage"] for x in rejects_all)),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    report_path = DATA_DIR / "v2/verify_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(
        f"input={counts['input']}  clean={len(rows)}  " f"rejected={len(rejects_all)}"
    )
    print(f"rejects by stage: {report['rejects_by_stage']}")
    print(f"wrote {out}, {rejects_path}, {report_path}")


if __name__ == "__main__":
    main()
