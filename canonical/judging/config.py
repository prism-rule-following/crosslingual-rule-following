"""Config for the judging pipeline -- one JSON file per judge run, loaded the
same way canonical/evaluation/inference.py loads its hyperparameter files.

Secrets (API keys, endpoints) stay in .env; everything that decides WHAT is
judged and WHERE it goes lives here, so pointing the judges at a new
inference run is a config change, not an edit to five scripts.
"""

import argparse
import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

JudgeName = Literal["gpt_mini", "deepseek", "gemini", "gemini_vertex"]


class JudgeConfig(BaseModel):
    judge: JudgeName = Field(..., description="which judge backend to run")
    judge_model: Optional[str] = Field(
        default=None,
        description="model / deployment name for the judge; overrides the "
        "provider's env var (AZURE_GPT_DEPLOYMENT, AZURE_DEEPSEEK_DEPLOYMENT, "
        "OPENROUTER_GEMINI_MODEL, VERTEX_MODEL) when set",
    )
    dataset_repo: str = Field(
        ...,
        description="HF dataset repo holding inference responses (the "
        "hf_result_repo of the inference config)",
    )
    dataset_subset: str = Field(
        default="",
        description="folder prefix inside dataset_repo; '' reads the root "
        "{model}/{lang}.parquet layout written directly by inference.py, "
        "'active_only_768_n3' reads the assembled export",
    )
    model_ids: List[str] = Field(
        ...,
        description="HuggingFace model ids whose responses to judge (same ids "
        "as the inference config; '/' is mapped to '__' for the repo path)",
    )
    language_codes: List[str] = Field(..., description="languages to judge")
    sample_indices: Optional[List[int]] = Field(
        default=None,
        description="only judge rows with these sample_idx values; None judges "
        "every sample present",
    )
    results_dir: str = Field(
        default="results", description="local root for all judge outputs"
    )
    output_name: Optional[str] = Field(
        default=None,
        description="subfolder of results_dir for this judge; defaults to `judge`",
    )
    max_workers: int = Field(
        default=5, description="parallel API calls for the main judging pass"
    )
    retry_max_workers: int = Field(
        default=5, description="parallel API calls for retry_failed_rows.py"
    )
    max_retries: int = Field(
        default=5, description="attempts per request on rate-limit errors"
    )
    skip_succeeded_from: List[str] = Field(
        default_factory=list,
        description="extra results files (relative to results_dir) whose "
        "successful rows are skipped, e.g. a Vertex run skipping rows the "
        "OpenRouter Gemini run already judged",
    )
    merge_sources: List[str] = Field(
        default_factory=list,
        description="results files (relative to results_dir) merged into this "
        "judge's final.jsonl by merge_gemini_sources.py",
    )
    expected_rows: Optional[int] = Field(
        default=None,
        description="row count of a complete run, printed as a sanity check "
        "by dedupe/merge",
    )
    hf_results_repo: Optional[str] = Field(
        default=None,
        description="HF dataset repo to upload final.jsonl to; upload is refused if unset",
    )
    hf_path_prefix: str = Field(
        default="",
        description="folder inside hf_results_repo to upload under, so separate "
        "runs don't overwrite each other",
    )
    hf_private: bool = Field(
        default=True, description="create hf_results_repo as private if missing"
    )

    @property
    def output_dir(self) -> str:
        return os.path.join(self.results_dir, self.output_name or self.judge)

    @property
    def results_file(self) -> str:
        return os.path.join(self.output_dir, "results.jsonl")

    @property
    def retry_file(self) -> str:
        return os.path.join(self.output_dir, "retry.jsonl")

    @property
    def final_file(self) -> str:
        return os.path.join(self.output_dir, "final.jsonl")

    def data_files(self) -> List[str]:
        prefix = f"{self.dataset_subset.strip('/')}/" if self.dataset_subset else ""
        return [
            f"{prefix}{model_id.replace('/', '__')}/{lang}.parquet"
            for model_id in self.model_ids
            for lang in self.language_codes
        ]

    def hf_remote_path(self) -> str:
        parts = [self.hf_path_prefix.strip("/"), self.output_name or self.judge]
        return "/".join(p for p in parts if p) + "/results.jsonl"


def load_config(path: str) -> JudgeConfig:
    with open(path) as f:
        return JudgeConfig.model_validate_json(f.read())


def parse_config_arg(description: str) -> JudgeConfig:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--config", required=True, help="path to a judge config JSON file"
    )
    return load_config(ap.parse_args().config)
