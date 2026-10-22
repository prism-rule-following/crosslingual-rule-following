"""Explicitly bounded live pilot/scoring runner; --dry-run never needs API keys."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from .core import PROTOCOL_VERSION, build_prompt, content_hash, digest, normalize_row, response_key
from .data import file_hash, read_jsonl, write_json, write_jsonl
from .providers import Provider, evaluate_reply, normalize_openai, normalize_vertex, as_dict


def config_check(config):
    if config.get("scale") not in ("binary", "numeric"):
        raise ValueError("config scale must be binary or numeric")
    if not 1 <= config.get("top_k", 20) <= 20 or not 0 <= config.get("min_scoring_mass", 0.9) <= 1:
        raise ValueError("invalid top_k or min_scoring_mass")
    if not 1 <= config.get("max_output_tokens", 16) <= 4096:
        raise ValueError("invalid max_output_tokens")
    judges = config.get("judges", [])
    ids = [j.get("id") for j in judges]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("unique nonempty judge IDs required")
    for j in judges:
        if j.get("provider") not in ("azure_v1", "azure_classic", "openrouter", "vertex"):
            raise ValueError("invalid provider")
        if not j.get("model") and not j.get("model_env"):
            raise ValueError("each judge needs an explicit model or model_env")
        if j["provider"].startswith("azure") and not j.get("env_prefix"):
            raise ValueError("Azure judge needs env_prefix")
        if j.get("token_limit_parameter", "max_completion_tokens") not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("invalid token limit parameter")
        if not 1 <= j.get("top_k", config.get("top_k", 20)) <= 20:
            raise ValueError("invalid per-judge top_k")
        if not 1 <= j.get("max_output_tokens", config.get("max_output_tokens", 16)) <= 4096:
            raise ValueError("invalid per-judge output-token budget")


def run_rows(rows, config, out, max_rows, dry_run=True, allow_resume=False, provider_factory=Provider):
    config_check(config)
    if max_rows < 1:
        raise ValueError("max_rows must be positive; unlimited API runs are not implicit")
    rows = [normalize_row(r) for r in rows]
    sources = sorted({(r.get("source_repo") or "UNSPECIFIED", r.get("source_revision") or "UNSPECIFIED")
                      for r in rows})
    seen = set()
    for r in rows:
        key = response_key(r)
        if key in seen:
            raise ValueError(f"duplicate input identity {key}")
        seen.add(key)
    # Explicit deterministic categories remain in the source audit but are not
    # silently assigned a different, invented judging rubric.
    eligible = []
    for row in rows:
        if row.get("confidence_judge_eligible") is False:
            continue
        build_prompt(row, config["scale"])
        eligible.append(row)
    rows = sorted(eligible, key=lambda r: json.dumps(response_key(r)))[:max_rows]
    if not rows:
        raise ValueError("no input rows")
    # Resolve model identities only for live requests, never replace config values with secrets.
    resolved = json.loads(json.dumps(config))
    if not dry_run:
        for j in resolved["judges"]:
            j["model"] = j.get("model") or os.environ.get(j["model_env"])
            if not j["model"]:
                raise ValueError(f"missing model setting {j['model_env']}")
    implementation = {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py")) if not p.name.startswith("test_")}
    fingerprint = digest({"protocol": PROTOCOL_VERSION, "config": resolved, "dry_run": dry_run,
                          "implementation": implementation, "source_snapshots": sources})
    out = Path(out)
    if out.exists() and not allow_resume:
        raise ValueError(f"output exists; use --resume explicitly: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_file = out / "manifest.json"
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("resume config/protocol/model differs from saved run")
    else:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        write_json(manifest_file, {"fingerprint": fingerprint, "config": resolved,
                   "protocol": PROTOCOL_VERSION, "implementation_sha256": implementation,
                   "input_source_snapshots": sources,
                   "dry_run": dry_run, "git_commit": commit,
                   "git_dirty": dirty, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "input_rows_this_invocation": len(rows), "planned_requests": 0 if dry_run else len(rows) * len(config["judges"]),
                   "temperature": 1, "top_p": 1, "retries": 0})
    results_file = out / "results.jsonl"
    existing = read_jsonl(results_file) if results_file.exists() else []
    done = set()
    hashes = {}
    for r in existing:
        key = (r["judge_id"], response_key(r))
        if r["run_fingerprint"] != fingerprint:
            raise ValueError("mixed run fingerprints in checkpoint")
        hashes[key] = r["input_sha256"]
        if r.get("scoring", {}).get("status") == "ok":
            done.add(key)
    outcomes = []
    for j in resolved["judges"]:
        top_k = j.get("top_k", config.get("top_k", 20))
        output_budget = j.get("max_output_tokens", config.get("max_output_tokens", 16))
        pending = []
        for row in rows:
            key = (j["id"], response_key(row))
            if key in hashes and hashes[key] != content_hash(row):
                raise ValueError("response content changed under an existing checkpoint identity")
            if key not in done:
                pending.append(row)
        client = provider_factory(j) if pending and not dry_run else None
        for row in pending:
            prompt = build_prompt(row, config["scale"])
            base = {**row, "judge_id": j["id"], "judge_model": j.get("model"),
                    "provider": j["provider"], "protocol": PROTOCOL_VERSION,
                    "scale": config["scale"], "run_fingerprint": fingerprint,
                    "scoring_request": {"top_k": top_k, "max_output_tokens": output_budget},
                    "input_sha256": content_hash(row), "prompt_sha256": digest(prompt),
                    "judge_prompt": prompt, "created_utc": datetime.now(timezone.utc).isoformat()}
            if dry_run:
                result = {**base, "scoring": {"status": "dry_run"}, "verdict": None, "error": None}
            else:
                # Call separately from normalization so even unsupported/malformed replies are retained.
                try:
                    raw = client.request(prompt, top_k, output_budget)
                except Exception as exc:
                    # Do not persist SDK exception strings, which can contain
                    # endpoints, headers or sensitive request diagnostics.
                    result = {**base, "scoring": {"status": "provider_error"}, "verdict": None,
                              "error": {"type": type(exc).__name__, "status_code": getattr(exc, "status_code", None)}}
                    with open(results_file, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    raise RuntimeError("provider call failed; sanitized error saved; no automatic retry") from None
                base["raw_provider_response"] = as_dict(raw)
                try:
                    normalized = normalize_vertex(raw) if j["provider"] == "vertex" else normalize_openai(raw)
                    scoring = evaluate_reply(normalized, config["scale"], config.get("min_scoring_mass", 0.9))
                    result = {**base, "resolved_model": normalized["resolved_model"] or "UNREPORTED",
                              "model_version_reported": bool(normalized["resolved_model"]),
                              "scoring": scoring, "verdict": scoring["weighted_verdict"], "error": None}
                except (ValueError, KeyError, TypeError) as exc:
                    result = {**base, "scoring": {"status": "invalid_provider_response"},
                              "verdict": None, "error": str(exc)}
            with open(results_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            outcomes.append(result)
            print(f'{j["id"]}: {len(outcomes)} completed; status={result["scoring"]["status"]}', flush=True)
            # Pilot is a gate: never spend a full run on silently missing confidence.
            if not dry_run and result["scoring"]["status"] != "ok" and config.get("stop_on_invalid_score", True):
                raise ValueError(f"stopped at invalid score ({result['scoring']['status']}); inspect saved reply before continuing")
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-rows", type=int, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="actually call paid judge APIs; default is dry-run")
    mode.add_argument("--dry-run", action="store_false", dest="live", help="validate locally without API access (default)")
    p.set_defaults(live=False)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--languages", nargs="+")
    p.add_argument("--judge", nargs="+", help="run selected configured judge IDs")
    args = p.parse_args()
    if args.live:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    config = json.loads(Path(args.config).read_text())
    if args.judge:
        known = {j["id"] for j in config["judges"]}
        if set(args.judge) - known:
            raise ValueError("unknown judge ID")
        config["judges"] = [j for j in config["judges"] if j["id"] in args.judge]
    rows = read_jsonl(args.input)
    if args.languages:
        rows = [r for r in rows if r.get("language") in args.languages]
    run_rows(rows, config, args.output_dir, args.max_rows, not args.live, args.resume)


if __name__ == "__main__":
    main()
