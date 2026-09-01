"""Shared constants for the vector-patching pipeline."""

from typing import Dict, List

CANONICAL_DATASET_REPO = "crosslingual-rule-following/canonical-dataset"
JUDGE_RESULTS_REPO = "crosslingual-rule-following/judge-results-active-only"
ACTIVATIONS_REPO = "crosslingual-rule-following/model-inference-activations"
EXPORT_REPO = "crosslingual-rule-following/vector-patching-responses"

JUDGES: List[str] = ["gpt_mini", "gemini", "deepseek"]

LANGUAGES: List[str] = ["en", "de", "hi", "ig", "it", "ko", "ru", "tr", "ur", "yo"]

# model_id -> activations-repo directory name ("/" -> "__")
MODEL_SLUGS: Dict[str, str] = {
    "Qwen/Qwen3-8B": "Qwen__Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "meta-llama__Llama-3.1-8B-Instruct",
}
PRIMARY_MODEL = "Qwen/Qwen3-8B"

HELD_VERDICT = "HELD"  # judge-results-active-only verdict values: HELD / VIOLATED / null

# extraction convention shared by every cached activation in this project:
# prompt-only, last token, hook_resid_post
HOOK_NAME = "hook_resid_post"
POSITION = -1

PRIMARY_PRESSURE_LEVEL = "L0"

# default cut points for classify_tiers(); recomputed rates should be
# eyeballed against these, not trusted blindly
HIGH_RESOURCE_HELD_RATE = 0.65
LOW_RESOURCE_HELD_RATE = 0.35
