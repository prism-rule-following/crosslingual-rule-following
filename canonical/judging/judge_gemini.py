"""
Judges inference responses with Gemini via OpenRouter.

Same 27-field schema as the other judges (join key: id + model_id +
language + sample_idx), with judge_model set to the Gemini model slug.

Baked in from the start:
  1. Real exponential backoff on rate limits, tuned for a per-MINUTE limit
     (the 429s are Google's own upstream RPM limit for this model).
  2. Multiple OpenRouter keys: on credit exhaustion (HTTP 402) the next key
     is tried immediately; once every key is exhausted the run PAUSES and
     rechecks every 60s, rather than silently logging errors for the rest
     of the dataset.

SETUP: .env with
    OPENROUTER_API_KEY=your-key-here
    OPENROUTER_API_KEY_2=optional-backup-key   (and _3)
    OPENROUTER_GEMINI_MODEL=google/gemini-3.7-flash   (ignored if judge_model is set in the config)

RUN (from the repo root):
    caffeinate -i python -m canonical.judging.judge_gemini \
        --config canonical/judging/configs/judge_gemini.json

Output: {results_dir}/gemini/results.jsonl
"""

import json
import random
import threading
import time

from openai import OpenAI

from canonical.judging.common import (
    JSON_FALLBACK_INSTRUCTION,
    STRICT_RESPONSE_FORMAT,
    error_result,
    get_secret,
    ok_result,
    parse_json_judgment,
    run_main_pass,
)
from canonical.judging.config import JudgeConfig, parse_config_arg

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CREDIT_ERROR_MARKERS = ("402", "insufficient credit", "insufficient_quota")
RATE_LIMIT_MARKERS = ("429", "rate limit", "ratelimiterror")


def is_credit_exhaustion_error(exc):
    """Checks the ACTUAL HTTP status code the openai SDK attaches to the
    exception, not a substring match on the error text -- OpenRouter error
    bodies contain request IDs that can coincidentally contain "402", which
    once caused a false "credits exhausted" pause on a plain rate limit."""
    status_code = getattr(exc, "status_code", None)
    if status_code == 402:
        return True
    if status_code is None:
        s = str(exc).lower()
        return any(m in s for m in CREDIT_ERROR_MARKERS)
    return False


def is_rate_limit_error(exc):
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    if status_code is None:
        s = str(exc).lower()
        return any(m in s for m in RATE_LIMIT_MARKERS)
    return False


class GeminiOpenRouterJudge:
    def __init__(self, config: JudgeConfig):
        keys = [
            k
            for k in (
                get_secret("OPENROUTER_API_KEY"),
                get_secret("OPENROUTER_API_KEY_2"),
                get_secret("OPENROUTER_API_KEY_3"),
            )
            if k
        ]
        self.judge_model = config.judge_model or get_secret(
            "OPENROUTER_GEMINI_MODEL", "google/gemini-3.7-flash"
        )
        if not keys or not self.judge_model:
            raise SystemExit(
                "Gemini isn't configured -- check .env has OPENROUTER_API_KEY set."
            )
        print(f"Gemini (OpenRouter): ready ({self.judge_model}), {len(keys)} key(s) configured")
        self.max_retries = config.max_retries
        self._clients = [OpenAI(base_url=OPENROUTER_BASE_URL, api_key=k) for k in keys]
        self._current_key_index = 0
        self._key_lock = threading.Lock()
        self._paused = threading.Event()
        self._strict_schema_supported = None

    def _get_current_client(self):
        with self._key_lock:
            return self._clients[self._current_key_index], self._current_key_index

    def _advance_to_next_key(self, failed_index):
        """Moves to the next key ONLY if we're still on the one that just
        failed (another thread may have already advanced it)."""
        with self._key_lock:
            if (
                self._current_key_index == failed_index
                and self._current_key_index < len(self._clients) - 1
            ):
                self._current_key_index += 1
                print(
                    f"\nSwitching to backup OpenRouter key ({self._current_key_index + 1}/{len(self._clients)}) "
                    f"-- key {failed_index + 1} exhausted.\n"
                )
                return True
            return False

    def _call_raw(self, messages, response_format):
        """Base 5s, doubling, capped at 65s (enough to cross into a fresh
        minute), with full jitter."""
        client, _ = self._get_current_client()
        base_delay, max_delay, factor = 5.0, 65.0, 2.0
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return client.chat.completions.create(
                    model=self.judge_model,
                    messages=messages,
                    response_format=response_format,
                )
            except Exception as e:
                last_error = e
                if is_credit_exhaustion_error(e):
                    raise  # let judge() handle key-switching / pause
                if is_rate_limit_error(e) and attempt < self.max_retries - 1:
                    sleep_time = random.uniform(0, min(max_delay, base_delay * (factor**attempt)))
                    print(
                        f"  [rate limit] backing off {sleep_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(sleep_time)
                    continue
                break
        raise last_error

    def _judge_raw(self, prompt):
        """Tries strict json_schema first, falls back to json_object + local
        validation if the model/OpenRouter doesn't support strict mode."""
        if self._strict_schema_supported is not False:
            try:
                resp = self._call_raw(
                    messages=[{"role": "user", "content": prompt}],
                    response_format=STRICT_RESPONSE_FORMAT,
                )
                parsed = json.loads(resp.choices[0].message.content)
                self._strict_schema_supported = True
                return ok_result(parsed)
            except Exception as e:
                if is_credit_exhaustion_error(e):
                    raise
                if self._strict_schema_supported is None:
                    self._strict_schema_supported = False
                else:
                    return error_result(e)

        try:
            resp = self._call_raw(
                messages=[{"role": "user", "content": prompt + JSON_FALLBACK_INSTRUCTION}],
                response_format={"type": "json_object"},
            )
            return parse_json_judgment(resp.choices[0].message.content)
        except Exception as e:
            if is_credit_exhaustion_error(e):
                raise
            return error_result(e)

    def judge(self, prompt):
        """On credit exhaustion: try the next configured key first, and only
        pause-and-wait once every key is exhausted."""
        while True:
            _, key_index = self._get_current_client()
            try:
                result = self._judge_raw(prompt)
            except Exception as e:
                if is_credit_exhaustion_error(e):
                    if self._advance_to_next_key(key_index):
                        continue  # retry immediately on the new key, no wait
                    if not self._paused.is_set():
                        self._paused.set()
                        print(
                            f"\n{'!'*70}\nALL OPENROUTER KEYS EXHAUSTED -- pausing.\n"
                            f"Recharge OpenRouter (or add another key to .env and restart),\n"
                            f"then this resumes automatically (rechecking every 60s).\n{'!'*70}\n"
                        )
                    time.sleep(60)
                    continue
                result = error_result(e)
            if self._paused.is_set():
                self._paused.clear()
                print("\nCredits confirmed available again -- resuming normally.\n")
            return result


if __name__ == "__main__":
    run_main_pass(parse_config_arg(__doc__.strip().splitlines()[0]))
