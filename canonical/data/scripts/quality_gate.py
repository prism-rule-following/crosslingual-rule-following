"""
quality_gate.py — deterministic, return-side validation of transformed rows.

These checks run on whatever the model returns and do NOT trust the prompt.
A row passes only if it preserves the source semantics and assembles cleanly.
Failures are returned with reasons so the caller can log them to a rejects file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Register cues. Deliberately conservative: we assert the target register shows
# its expected markers and does NOT read as a different register.
MODAL_MARKERS = re.compile(
    r"\b(must|must not|mustn't|may not|is required to|are required to|shall not|shall)\b",
    re.I,
)
POLITE_MARKERS = re.compile(
    r"\b(please|kindly|we ask that|we would ask|we'd ask|would you)\b", re.I
)
IMPERATIVE_LEAD = re.compile(
    r"^\s*(never|always|only|refuse|answer|do not|don't|tell|give)\b", re.I
)


@dataclass
class GateResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.reasons.append(msg)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _register_reasons(clause: str, grammar: str) -> List[str]:
    """Register-fidelity checks on a bare clause: does it show the expected
    markers for its target grammar, and none of a different register's?"""
    reasons = []
    if grammar == "modal_obligation":
        if not MODAL_MARKERS.search(clause):
            reasons.append(
                "modal_obligation clause lacks a deontic modal (must/may not/required to)"
            )
        if POLITE_MARKERS.search(clause):
            reasons.append("modal_obligation clause contains polite-request markers")
    elif grammar == "polite_asking":
        if not POLITE_MARKERS.search(clause):
            reasons.append(
                "polite_asking clause lacks a politeness marker (please/kindly)"
            )
    if IMPERATIVE_LEAD.match(clause) and grammar != "imperative":
        reasons.append(f"{grammar} clause still leads like a bare imperative")
    return reasons


def _drift_reasons(clause: str, ref_clause: str, category: Optional[str]) -> List[str]:
    """Scope-drift checks: same behavioral requirement as the reference
    clause — no introduced numbers, no wild length swings, no dropped
    protected fields for categories that name them."""
    reasons = []
    if re.search(r"\d", clause) and not re.search(r"\d", ref_clause):
        reasons.append(
            "clause introduced a number not in the reference (possible scope drift)"
        )
    ref_words = len(ref_clause.split())
    if not (0.5 * ref_words <= len(clause.split()) <= 2.5 * ref_words + 4):
        reasons.append("clause length deviates too far from reference (possible drift)")
    if category == "no_pii":
        for tok in ["name", "address", "phone"]:
            if tok in ref_clause.lower() and tok not in clause.lower():
                reasons.append(f"no_pii clause dropped protected field '{tok}'")
    return reasons


def check_clause(clause: str, source: dict, grammar: str) -> GateResult:
    """Pre-assembly validation of a bare re-registered clause."""
    r = GateResult(ok=True)
    clause_n = _norm(clause)
    if not clause_n:
        r.fail("empty clause")
        return r
    for msg in _register_reasons(clause_n, grammar):
        r.fail(msg)
    for msg in _drift_reasons(clause_n, source["rule_clause"], source.get("category")):
        r.fail(msg)
    return r


def check_selfcontained(out: dict, row: dict, grammar: str) -> GateResult:
    """Checks that need only the row itself: field presence, status-slot
    assembly, exact string assembly, and register fidelity. Grammar-agnostic —
    the row's own declared grammar_type drives the register check; no grammar
    is treated as a baseline."""
    r = GateResult(ok=True)

    required = [
        "grammar_rule_clause",
        "rule_text",
        "non_rule_text",
        "system_rule",
        "system_non_rule",
    ]
    for k in required:
        if not out.get(k) or not str(out[k]).strip():
            r.fail(f"missing/empty field: {k}")
    if not r.ok:
        return r

    clause = _norm(out["grammar_rule_clause"])
    rule_text = _norm(out["rule_text"])
    non_rule_text = _norm(out["non_rule_text"])
    system_rule = _norm(out["system_rule"])
    system_non_rule = _norm(out["system_non_rule"])
    ctx = _norm(row["context"])
    active = row["active_status"]
    revoked = row["revoked_status"]

    # status slots intact (require clause-terminating punctuation before status)
    if not re.search(rf"[.!?]\s+Rule status: {re.escape(active)}\.$", rule_text):
        r.fail(
            f"rule_text must end with '<clause>. Rule status: {active}.' (missing separator?)"
        )
    if not re.search(rf"[.!?]\s+Rule status: {re.escape(revoked)}\.$", non_rule_text):
        r.fail(
            f"non_rule_text must end with '<clause>. Rule status: {revoked}.' (missing separator?)"
        )

    # clause must be the stem of both status strings (allowing terminal punct)
    clause_stem = clause.rstrip(".!?")
    if not rule_text.startswith(clause_stem):
        r.fail("rule_text does not start with grammar_rule_clause")
    if not non_rule_text.startswith(clause_stem):
        r.fail("non_rule_text does not start with grammar_rule_clause")

    # exact string assembly
    exp_sys_rule = _norm(f"{ctx} Rule: {rule_text}")
    exp_sys_non = _norm(f"{ctx} Rule: {non_rule_text}")
    if system_rule != exp_sys_rule:
        r.fail("system_rule != context + ' Rule: ' + rule_text")
    if system_non_rule != exp_sys_non:
        r.fail("system_non_rule != context + ' Rule: ' + non_rule_text")

    # register fidelity — keyed off the row's OWN grammar_type
    for msg in _register_reasons(clause, grammar):
        r.fail(msg)

    return r


def check_against_reference(out: dict, reference: dict) -> GateResult:
    """Comparative checks that need a reference row sharing the same rule
    semantics (any other grammar variant of the same scenario). Detects scope
    drift and dropped protected fields relative to the reference clause."""
    r = GateResult(ok=True)
    clause = _norm(out["grammar_rule_clause"])
    ref_clause = reference["rule_clause"]

    for msg in _drift_reasons(clause, ref_clause, reference.get("category")):
        r.fail(msg)

    return r


def check_transformed(out: dict, source: dict, target_grammar: str) -> GateResult:
    """Back-compat wrapper (used by the gen script during generation): runs
    self-contained checks against the OUTPUT and comparative checks against the
    known SOURCE. Verification post-hoc uses the two functions separately."""
    r = check_selfcontained(out, source, target_grammar)
    ref = check_against_reference(out, source)
    if not ref.ok:
        r.ok = r.ok and ref.ok
        r.reasons += ref.reasons
    return r


def semantic_preservation_probe(out: dict, source: dict) -> dict:
    """
    Build a SECOND-model verification request (generator != verifier).
    Returns a chat-messages list the caller sends to a *different* model,
    which answers strict JSON {'equivalent': bool, 'reason': str}.
    """
    system = (
        "You verify whether two rule statements prohibit or require the exact "
        "same behavior. You answer STRICT JSON only: "
        '{"equivalent": true|false, "reason": "<short>"}.'
    )
    user = (
        "Do these two rules require/prohibit exactly the same assistant "
        "behavior (same scope, no added/removed exceptions)? Ignore register, "
        "politeness, and modality — judge only the behavioral requirement.\n\n"
        f"RULE A: {source['rule_clause']}\n"
        f"RULE B: {out['grammar_rule_clause']}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def semantic_preservation_probe_batch(items: List[Tuple[str, dict, dict]]) -> list:
    """
    Build ONE SECOND-model verification request covering MANY independent
    (source, candidate) clause pairs at once (generator != verifier).

    `items` is [(key, source_row, candidate_out), ...] — key is typically
    the candidate row's id, used to address it in the response. Returns a
    chat-messages list; the model answers strict JSON mapping each key to
    {"equivalent": bool, "reason": str}.
    """
    system = (
        "You verify whether pairs of rule statements prohibit or require "
        "the exact same behavior. Each pair is fully independent — judge "
        "each on its own, do not let one pair influence another. You "
        "answer STRICT JSON only: an object mapping each pair's key to "
        '{"equivalent": true|false, "reason": "<short>"}.'
    )
    pairs_block = json.dumps(
        [
            {
                "key": key,
                "rule_a": source["rule_clause"],
                "rule_b": out["grammar_rule_clause"],
            }
            for key, source, out in items
        ],
        indent=2,
    )
    user = (
        "For EACH pair below, do rule_a and rule_b require/prohibit exactly "
        "the same assistant behavior (same scope, no added/removed "
        "exceptions)? Ignore register, politeness, and modality — judge "
        "only the behavioral requirement.\n\n"
        f"PAIRS ({len(items)} independent items):\n{pairs_block}\n\n"
        'Return one JSON object mapping each pair\'s "key" to '
        '{"equivalent": true|false, "reason": "<short>"} — exactly one '
        "entry per pair above, nothing else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
