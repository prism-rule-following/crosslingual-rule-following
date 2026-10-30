"""
Judges inference responses with GPT-5.4-mini (Azure).

Output is saved with a CLEAN, judge-specific schema (no gpt_mini_/deepseek_/
gemini_ prefixes) -- one folder per JUDGE under `results_dir`, mirroring how
the inference dataset is organized one folder per MODEL. All judges share
the composite key (id, model_id, language, sample_idx), which is what lets
you merge them back together later for comparison/consensus.

SETUP: .env with
    AZURE_GPT_API_KEY=your-key-here
    AZURE_GPT_ENDPOINT_V1=https://<resource>.services.ai.azure.com/openai/v1
    AZURE_GPT_ENDPOINT_CLASSIC=https://<resource>.openai.azure.com
    AZURE_GPT_API_VERSION=2024-02-01
    AZURE_GPT_DEPLOYMENT=gpt-5.4-mini   (ignored if judge_model is set in the config)

RUN (from the repo root):
    caffeinate -i python -m canonical.judging.judge_gpt_mini \
        --config canonical/judging/configs/judge_gpt_mini.json

Output: {results_dir}/gpt_mini/results.jsonl. Checkpointed and resumable --
safe to Ctrl-C and re-run; already-judged rows are skipped automatically.
"""

import json
import random
import time

from openai import AzureOpenAI, OpenAI

from canonical.judging.common import (
    STRICT_RESPONSE_FORMAT,
    _configured,
    error_result,
    get_secret,
    ok_result,
    run_main_pass,
)
from canonical.judging.config import JudgeConfig, parse_config_arg


def is_rate_limit_error(exc):
    s = str(exc).lower()
    return (
        "429" in s
        or "rate limit" in s
        or "ratelimiterror" in type(exc).__name__.lower()
    )


class AzureChatClient:
    """v1-then-classic Azure endpoint fallback (remembering whichever one
    worked), with exponential backoff on rate limits."""

    def __init__(self, deployment, api_key, endpoint_v1, endpoint_classic, api_version, max_retries):
        self.deployment = deployment
        self.max_retries = max_retries
        self._clients = {
            "v1": OpenAI(base_url=endpoint_v1, api_key=api_key),
            "classic": AzureOpenAI(
                azure_endpoint=endpoint_classic,
                api_key=api_key,
                api_version=api_version,
            ),
        }
        self._working_client_name = None

    def create(self, messages, response_format=None):
        order = ["v1", "classic"]
        if self._working_client_name is not None:
            order = [self._working_client_name] + [
                n for n in order if n != self._working_client_name
            ]
        last_error = None
        for name in order:
            client = self._clients[name]
            for attempt in range(self.max_retries):
                try:
                    kwargs = {"model": self.deployment, "messages": messages}
                    if response_format is not None:
                        kwargs["response_format"] = response_format
                    resp = client.chat.completions.create(**kwargs)
                    self._working_client_name = name
                    return resp
                except Exception as e:
                    last_error = e
                    if is_rate_limit_error(e) and attempt < self.max_retries - 1:
                        delay = (2**attempt) + random.uniform(0, 1)
                        print(
                            f"  [rate limit] backing off {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(delay)
                        continue
                    break  # non-rate-limit error, or out of retries -- try the other endpoint
        raise last_error


class GptMiniJudge:
    def __init__(self, config: JudgeConfig):
        api_key = get_secret("AZURE_GPT_API_KEY")
        self.judge_model = config.judge_model or get_secret(
            "AZURE_GPT_DEPLOYMENT", "gpt-5.4-mini"
        )
        if not _configured(api_key, self.judge_model):
            raise SystemExit(
                "GPT-5.4-mini isn't configured -- check .env has AZURE_GPT_API_KEY set."
            )
        print(f"GPT-mini (Azure): ready ({self.judge_model})")
        self.client = AzureChatClient(
            deployment=self.judge_model,
            api_key=api_key,
            endpoint_v1=get_secret(
                "AZURE_GPT_ENDPOINT_V1",
                "https://prismmechinterp.services.ai.azure.com/openai/v1",
            ),
            endpoint_classic=get_secret(
                "AZURE_GPT_ENDPOINT_CLASSIC", "https://prismmechinterp.openai.azure.com"
            ),
            api_version=get_secret("AZURE_GPT_API_VERSION", "2024-02-01"),
            max_retries=config.max_retries,
        )

    def judge(self, prompt):
        try:
            resp = self.client.create(
                messages=[{"role": "user", "content": prompt}],
                response_format=STRICT_RESPONSE_FORMAT,
            )
            return ok_result(json.loads(resp.choices[0].message.content))
        except Exception as e:
            return error_result(e)


if __name__ == "__main__":
    run_main_pass(parse_config_arg(__doc__.strip().splitlines()[0]))
