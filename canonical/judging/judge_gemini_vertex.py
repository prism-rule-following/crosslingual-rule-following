"""
Judges inference responses with Gemini via direct Google Cloud (Vertex AI).

Uses vertexai=True + api_key, NOT OpenRouter -- bypasses OpenRouter's
routing layer, which was measured at ~17 rows/min even at zero concurrency.
Same 27-field schema, same join key, same prompt as judge_gemini.py, so the
two outputs can be merged with merge_gemini_sources.py.

SETUP: .env with
    VERTEX_AI_API_KEY=your-key-here
    VERTEX_MODEL=gemini-2.5-flash   (no "google/" prefix; ignored if judge_model is set in the config)
Also needs: pip install google-genai

RUN (from the repo root):
    caffeinate -i python -m canonical.judging.judge_gemini_vertex \
        --config canonical/judging/configs/judge_gemini_vertex.json

Output: {results_dir}/gemini_vertex/results.jsonl -- separate from the
OpenRouter results. Set `skip_succeeded_from` in the config to skip rows
the OpenRouter run already judged.

NOTE ON STRUCTURED OUTPUT: tries response_schema/response_mime_type first,
and falls back to plain prompt-for-JSON + manual parsing if that errors.
"""

import json
import random
import time

from google import genai
from google.genai import types as genai_types

from canonical.judging.common import (
    _configured,
    error_result,
    get_secret,
    ok_result,
    parse_json_judgment,
    run_main_pass,
)
from canonical.judging.config import JudgeConfig, parse_config_arg

VERTEX_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["HELD", "VIOLATED"]},
    },
    "required": ["reasoning", "verdict"],
}


def is_rate_limit_error(exc):
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status == 429:
        return True
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or "resourceexhausted" in s


class GeminiVertexJudge:
    def __init__(self, config: JudgeConfig):
        api_key = get_secret("VERTEX_AI_API_KEY")
        self.judge_model = config.judge_model or get_secret(
            "VERTEX_MODEL", "gemini-2.5-flash"
        )
        if not _configured(api_key, self.judge_model):
            raise SystemExit("Not configured -- check .env has VERTEX_AI_API_KEY set.")
        print(f"Gemini (direct Google Cloud): ready ({self.judge_model})")
        self.max_retries = config.max_retries
        self.client = genai.Client(vertexai=True, api_key=api_key)
        self._schema_supported = None

    def _call_raw(self, prompt):
        """Same RPM-window-aware backoff as the OpenRouter judge (base 5s,
        doubling, capped at 65s, full jitter)."""
        base_delay, max_delay, factor = 5.0, 65.0, 2.0
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.judge_model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VERTEX_RESPONSE_SCHEMA,
                    ),
                )
            except Exception as e:
                last_error = e
                if is_rate_limit_error(e) and attempt < self.max_retries - 1:
                    sleep_time = random.uniform(0, min(max_delay, base_delay * (factor**attempt)))
                    print(
                        f"  [rate limit] backing off {sleep_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(sleep_time)
                    continue
                break
        raise last_error

    def judge(self, prompt):
        if self._schema_supported is not False:
            try:
                resp = self._call_raw(prompt)
                parsed = json.loads(resp.text)
                self._schema_supported = True
                return ok_result(parsed)
            except Exception as e:
                if self._schema_supported is None:
                    self._schema_supported = False
                else:
                    return error_result(e)

        json_prompt = (
            prompt
            + '\n\nRespond ONLY with valid JSON, no other text, with exactly two keys: "reasoning" (string) and "verdict" (either "HELD" or "VIOLATED").'
        )
        try:
            resp = self.client.models.generate_content(
                model=self.judge_model, contents=json_prompt
            )
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return parse_json_judgment(raw.strip())
        except Exception as e:
            return error_result(e)


if __name__ == "__main__":
    run_main_pass(parse_config_arg(__doc__.strip().splitlines()[0]))
