"""
Judges inference responses with DeepSeek-V4-Pro (Azure).

Same 27-field schema as the other judges (join key: id + model_id +
language + sample_idx). Keep max_workers low in the config: 8 workers was
measured ~4x SLOWER than 3 (60/min vs 245/min) due to rate-limit backoff
overhead -- don't raise it without re-measuring throughput first.

SETUP: .env with
    AZURE_DEEPSEEK_API_KEY=... (defaults to AZURE_GPT_API_KEY if same resource)
    AZURE_DEEPSEEK_ENDPOINT_V1=https://<resource>.services.ai.azure.com/openai/v1
    AZURE_DEEPSEEK_ENDPOINT_CLASSIC=https://<resource>.openai.azure.com
    AZURE_DEEPSEEK_API_VERSION=2024-02-01
    AZURE_DEEPSEEK_DEPLOYMENT=DeepSeek-V4-Pro   (ignored if judge_model is set in the config)

RUN (from the repo root):
    caffeinate -i python -m canonical.judging.judge_deepseek \
        --config canonical/judging/configs/judge_deepseek.json

Output: {results_dir}/deepseek/results.jsonl
"""

import json
import time

from canonical.judging.common import (
    JSON_FALLBACK_INSTRUCTION,
    STRICT_RESPONSE_FORMAT,
    _configured,
    error_result,
    get_secret,
    ok_result,
    parse_json_judgment,
    run_main_pass,
)
from canonical.judging.config import JudgeConfig, parse_config_arg
from canonical.judging.judge_gpt_mini import AzureChatClient


class DeepSeekJudge:
    # Azure's content-safety filter blocks DETERMINISTICALLY -- the same
    # prompt gets blocked every time, so these rows count as "done" on
    # restart instead of being re-attempted (which only wrote duplicates).
    permanent_failure_markers = ("jailbreak", "content_filter")

    def __init__(self, config: JudgeConfig):
        api_key = get_secret("AZURE_DEEPSEEK_API_KEY", get_secret("AZURE_GPT_API_KEY"))
        self.judge_model = config.judge_model or get_secret(
            "AZURE_DEEPSEEK_DEPLOYMENT", "DeepSeek-V4-Pro"
        )
        if not _configured(api_key, self.judge_model):
            raise SystemExit(
                "DeepSeek isn't configured -- check .env has AZURE_DEEPSEEK_API_KEY (or AZURE_GPT_API_KEY) set."
            )
        print(f"DeepSeek (Azure): ready ({self.judge_model})")
        self.client = AzureChatClient(
            deployment=self.judge_model,
            api_key=api_key,
            endpoint_v1=get_secret(
                "AZURE_DEEPSEEK_ENDPOINT_V1",
                "https://prismmechinterp.services.ai.azure.com/openai/v1",
            ),
            endpoint_classic=get_secret(
                "AZURE_DEEPSEEK_ENDPOINT_CLASSIC",
                "https://prismmechinterp.openai.azure.com",
            ),
            api_version=get_secret("AZURE_DEEPSEEK_API_VERSION", "2024-02-01"),
            max_retries=config.max_retries,
        )
        self._strict_schema_supported = None

    def _judge(self, prompt):
        """Tries strict json_schema first, falls back to json_object + local
        validation if the deployment doesn't support strict mode."""
        if self._strict_schema_supported is not False:
            try:
                resp = self.client.create(
                    messages=[{"role": "user", "content": prompt}],
                    response_format=STRICT_RESPONSE_FORMAT,
                )
                parsed = json.loads(resp.choices[0].message.content)
                self._strict_schema_supported = True
                return ok_result(parsed)
            except Exception as e:
                if self._strict_schema_supported is None:
                    self._strict_schema_supported = False
                else:
                    return error_result(e)

        try:
            resp = self.client.create(
                messages=[{"role": "user", "content": prompt + JSON_FALLBACK_INSTRUCTION}],
                response_format={"type": "json_object"},
            )
            return parse_json_judgment(resp.choices[0].message.content)
        except Exception as e:
            return error_result(e)

    def judge(self, prompt):
        start = time.time()
        result = self._judge(prompt)
        duration = time.time() - start
        if duration > 10:  # only print for genuinely slow calls, not routine noise
            print(f"  [slow call] {duration:.1f}s")
        return result


if __name__ == "__main__":
    run_main_pass(parse_config_arg(__doc__.strip().splitlines()[0]), progress_every=25)
