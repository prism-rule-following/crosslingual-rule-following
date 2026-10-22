"""Small live credential checks. Never outputs credential values."""

import argparse
import json
import os
from pathlib import Path


def check_access():
    from dotenv import load_dotenv
    import httpx
    from openai import OpenAI, AzureOpenAI
    load_dotenv()
    secrets = [v for k, v in os.environ.items() if any(w in k.upper() for w in ("KEY", "TOKEN", "SECRET", "PASSWORD")) and v and len(v) > 4]

    def safe_error(exc):
        text = str(exc)
        for secret in sorted(secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        return {"type": type(exc).__name__, "http_status": code if isinstance(code, (int, str)) else None,
                "detail": text[:1000]}

    def chat(client, model):
        try:
            raw = client.chat.completions.create(model=model,
                messages=[{"role": "user", "content": "Reply with exactly OK."}], max_completion_tokens=64)
        except Exception as exc:
            # One bounded compatibility retry, only if the parameter is rejected.
            if getattr(exc, "status_code", None) == 400 and "max_completion_tokens" in str(exc):
                raw = client.chat.completions.create(model=model,
                    messages=[{"role": "user", "content": "Reply with exactly OK."}], max_tokens=16)
            else:
                raise
        return {"request_succeeded": True, "returned_model": raw.model,
                "has_text_response": bool(raw.choices and raw.choices[0].message.content),
                "finish_reason": raw.choices[0].finish_reason if raw.choices else None}

    results = []
    gpt_key = os.environ.get("AZURE_GPT_API_KEY")
    for prefix, default_model in (("AZURE_GPT", "gpt-5.4-mini"), ("AZURE_DEEPSEEK", "DeepSeek-V4-Pro")):
        key = os.environ.get(prefix + "_API_KEY") or (gpt_key if prefix == "AZURE_DEEPSEEK" else None)
        key_name = prefix + "_API_KEY" if os.environ.get(prefix + "_API_KEY") else "AZURE_GPT_API_KEY (shared-resource test)"
        model = os.environ.get(prefix + "_DEPLOYMENT", default_model)
        endpoint = os.environ.get(prefix + "_ENDPOINT_V1") or os.environ.get("AZURE_GPT_ENDPOINT_V1")
        if key and endpoint:
            result = {"service": prefix, "route": "azure_v1", "credential": key_name, "requested_model": model}
            try:
                result.update(chat(OpenAI(api_key=key, base_url=endpoint, max_retries=0, timeout=25), model))
            except Exception as exc:
                result.update(request_succeeded=False, error=safe_error(exc))
            results.append(result)
            print(json.dumps(result), flush=True)
        classic = os.environ.get(prefix + "_ENDPOINT_CLASSIC") or os.environ.get("AZURE_GPT_ENDPOINT_CLASSIC")
        version = os.environ.get(prefix + "_API_VERSION") or os.environ.get("AZURE_GPT_API_VERSION")
        if key and classic and version and (not results or not results[-1].get("request_succeeded")):
            result = {"service": prefix, "route": "azure_classic", "credential": key_name, "requested_model": model}
            try:
                client = AzureOpenAI(api_key=key, azure_endpoint=classic, api_version=version, max_retries=0, timeout=25)
                result.update(chat(client, model))
            except Exception as exc:
                result.update(request_succeeded=False, error=safe_error(exc))
            results.append(result)
            print(json.dumps(result), flush=True)

    for name in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY_3"):
        key = os.environ.get(name)
        if not key:
            continue
        model = os.environ.get("OPENROUTER_GEMINI_MODEL", "google/gemini-3.7-flash")
        result = {"service": "OPENROUTER", "credential": name, "requested_model": model}
        try:
            response = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"}, timeout=20)
            result["key_check_http_status"] = response.status_code
            result["key_authenticated"] = response.status_code == 200
            if response.status_code == 200:
                result.update(chat(OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=25), model))
            else:
                result["request_succeeded"] = False
        except Exception as exc:
            result.update(request_succeeded=False, error=safe_error(exc))
        results.append(result)
        print(json.dumps(result), flush=True)

    if os.environ.get("VERTEX_AI_API_KEY"):
        from google import genai
        from google.genai import types
        model = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
        result = {"service": "VERTEX", "credential": "VERTEX_AI_API_KEY", "requested_model": model}
        try:
            client = genai.Client(vertexai=True, api_key=os.environ["VERTEX_AI_API_KEY"],
                                  http_options={"timeout": 25000, "retry_options": {"attempts": 1}})
            raw = client.models.generate_content(model=model, contents="Reply with exactly OK.",
                                                  config=types.GenerateContentConfig(max_output_tokens=16))
            result.update(request_succeeded=True, has_text_response=bool(raw.text), returned_model=raw.model_version)
        except Exception as exc:
            result.update(request_succeeded=False, error=safe_error(exc))
        results.append(result)
        print(json.dumps(result), flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    path = Path(args.output)
    if path.exists():
        raise SystemExit("output already exists; choose a new path")
    results = check_access()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
