"""Bounded live logprob capability checks on one pinned response per route."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .core import build_prompt, content_hash, digest, PROTOCOL_VERSION
from .data import write_json, write_jsonl
from .providers import Provider, normalize_openai, normalize_vertex, evaluate_reply


def redact_error(exc):
    text = str(exc)
    for name, value in os.environ.items():
        if any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and value and len(value) > 4:
            text = text.replace(value, "[REDACTED]")
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return {"type": type(exc).__name__, "http_status": code if isinstance(code, (int, str)) else None,
            "detail": text[:1600]}


def pilot(source, out, selected=None, budget=512, top_k=20):
    from dotenv import load_dotenv
    load_dotenv()
    with open(source, encoding="utf-8") as f:
        row = next(json.loads(line) for line in f
                   if (r := json.loads(line)).get("language") == "en" and r.get("confidence_judge_eligible"))
    prompt = build_prompt(row, "binary")
    cases = [
        {"id": "gpt_mini", "provider": "azure_v1", "env_prefix": "AZURE_GPT",
         "model": os.environ.get("AZURE_GPT_DEPLOYMENT")},
        {"id": "deepseek", "provider": "azure_v1", "env_prefix": "AZURE_DEEPSEEK" if os.environ.get("AZURE_DEEPSEEK_API_KEY") else "AZURE_GPT",
         "model": os.environ.get("AZURE_DEEPSEEK_DEPLOYMENT", "DeepSeek-V4-Pro")},
        {"id": "gemini_openrouter", "provider": "openrouter", "model": os.environ.get("OPENROUTER_GEMINI_MODEL")},
        {"id": "gemini_vertex", "provider": "vertex", "model": os.environ.get("VERTEX_MODEL", "").rstrip("%")},
    ]
    if selected:
        if set(selected) - {c["id"] for c in cases}:
            raise ValueError("unknown pilot judge")
        cases = [c for c in cases if c["id"] in selected]
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "input.jsonl", [row])
    write_json(output / "manifest.json", {"protocol": PROTOCOL_VERSION, "prompt_sha256": digest(prompt),
               "input_sha256": content_hash(row), "source_revision": row.get("source_revision"),
               "cases": cases, "max_requests": len(cases), "max_output_tokens_per_request": budget,
               "temperature": 1, "top_p": 1, "top_k": top_k,
               "vertex_name_cleanup_in_memory": os.environ.get("VERTEX_MODEL", "").endswith("%"),
               "created_utc": datetime.now(timezone.utc).isoformat()})
    results = []
    for case in cases:
        result = {"judge_id": case["id"], "provider": case["provider"], "requested_model": case["model"]}
        try:
            raw = Provider(case).request(prompt, top_k=top_k, max_output_tokens=budget)
            write_json(output / (case["id"] + "_raw.json"), raw)
            result["request_succeeded"] = True
        except Exception as exc:
            result.update(request_succeeded=False, error=redact_error(exc))
        else:
            try:
                reply = normalize_vertex(raw) if case["provider"] == "vertex" else normalize_openai(raw)
                result.update(logprobs_returned=True, returned_model=reply["resolved_model"],
                              finish_reason=reply["finish_reason"], returned_text=reply["text"],
                              decoding_positions=len(reply["positions"]))
                result["scoring"] = evaluate_reply(reply, "binary", 0.9)
            except Exception as exc:
                result["validation_error"] = redact_error(exc)
                result.setdefault("logprobs_returned", False)
        results.append(result)
        write_json(output / (case["id"] + "_summary.json"), result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    write_json(output / "summary.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge", nargs="+")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.max_output_tokens <= 2048:
        parser.error("output-token budget must be 1..2048")
    if not 1 <= args.top_k <= 20:
        parser.error("top-k must be 1..20")
    pilot(args.input, args.output_dir, args.judge, args.max_output_tokens, args.top_k)
