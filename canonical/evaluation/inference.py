"""Run generation and/or activation-extraction inference passes over the
cross-lingual rule-following dataset for one or more models, optionally
uploading results (responses, activations) to the Hugging Face Hub.
"""

import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer
from transformer_lens.model_bridge import TransformerBridge
import transformer_lens.utilities as utils
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import argparse
from model.dataset import (
    CrossLingualRuleFollowingDataset,
    DataCategories,
    DatasetConfig,
    DatasetLanguageCode,
    HFDataHelper,
)


class ModelResponse(BaseModel):
    """One model-generated response, keyed back to its source dataset row."""

    id: str = Field(..., description="row id from the source dataset")
    category: DataCategories = Field(..., description="rule category of the source row")
    topic: str = Field(..., description="topic of the source row")
    grammar_type: str = Field(..., description="grammar type of the source row")
    language: DatasetLanguageCode = Field(..., description="language of the source row")
    system: str = Field(default=..., description="System prompt for the row")
    user_query: str = Field(
        ..., description="user query the model was asked to respond to"
    )
    response: str = Field(..., description="model-generated response text")
    sample_idx: int = Field(
        default=0,
        description="index of this stochastic sample among the n_samples "
        "generated for this prompt row",
    )
    checker: Optional[str] = Field(
        default=None, description="checker spec used to grade the response, if any"
    )
    pair_type: Optional[str] = Field(
        default=None, description="contrastive pair type of the source row, if any"
    )


class ModelGenerationConfig(BaseModel):
    model_ids: List[str] = Field(
        ...,
        description="HuggingFace model ids to run; one full generation/activation "
        "pass is performed per id",
    )
    dataset_config: DatasetConfig = Field(
        ...,
        description="config for loading/filtering the source rule-following dataset",
    )
    n_samples: int = Field(
        default=1,
        description="number of independent stochastic responses to generate per "
        "prompt row (self-consistency style); each output row is tagged with a sample_idx",
    )
    push_to_hf: bool = Field(
        default=True,
        description="whether to upload results to the Hugging Face Hub after each model finishes",
    )
    language_code: DatasetLanguageCode = Field(
        ...,
        description="language of the dataset rows being run, used to key the uploaded output path",
    )
    max_new_tokens: int = Field(
        default=100, description="maximum number of tokens to generate per response"
    )
    temperature: float = Field(
        default=1.0, description="sampling temperature for response generation"
    )
    generation_batch_size: int = Field(
        default=30, description="batch size for response generation"
    )
    activation_batch_size: int = Field(
        default=20,
        description="batch size for hidden-state extraction; kept separate from "
        "generation_batch_size since caching activations is far more memory-hungry per example",
    )
    checkpoint_dir: str = Field(
        default="/content/drive/MyDrive/crosslingual-rule-following/inference",
        description="local directory for per-batch response/activation checkpoints, "
        "so a crash mid-run resumes from the last completed batch instead of redoing it",
    )
    random_seed: int = Field(
        default=42, description="torch RNG seed, for reproducible stochastic sampling"
    )
    hf_result_repo: Optional[str] = Field(
        default=None,
        description="HF dataset repo id to upload generated responses to; upload is skipped if unset",
    )
    hf_activations_repo: Optional[str] = Field(
        default=None,
        description="HF dataset repo id to upload extracted activations to; upload is skipped if unset",
    )
    run_inference_response: bool = Field(
        default=True, description="whether to run the response-generation pass"
    )
    run_inference_activations: bool = Field(
        default=False, description="whether to run the activation-extraction pass"
    )

    @property
    def device(self) -> str:
        return utils.get_device()


class ModelRunner:
    """Stateful, single-model runner: loads a model/tokenizer, then generates
    responses and/or extracts hidden-state activations over a dataset.
    """

    def __init__(self, config: ModelGenerationConfig) -> None:
        self.config = config
        self.model: Optional[TransformerBridge] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.supports_system_role: bool = True
        self.model_id: Optional[str] = None

    def load(self, model_id: str) -> None:
        print(f"Loading: {model_id}")
        self.model_id = model_id
        torch.manual_seed(self.config.random_seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = TransformerBridge.boot_transformers(
            model_id, device=self.config.device
        )
        self.model.enable_compatibility_mode(disable_warnings=True)
        self.model.original_model.eval()
        # Required for patching hooks (qkv_input) to actually be populated during run_with_cache.
        self.model.cfg.use_attn_result = True
        self.model.cfg.use_split_qkv_input = True
        self.model.cfg.use_hook_mlp_in = True
        self.supports_system_role = self._check_system_role_support()

        n_layers = self.model.cfg.n_layers
        # cfg.n_params is None under TransformerBridge; count directly instead.
        n_parameters = sum(p.numel() for p in self.model.parameters())
        n_heads = self.model.cfg.n_heads
        d_vocab = self.model.cfg.d_vocab
        architecture = self.model.cfg.architecture
        print(
            f"Model: {model_id} | {n_parameters / 1e9:.2f}B params | {n_layers} layers | "
            f"{n_heads} heads | {d_vocab} vocabulary | {architecture} architecture"
        )

    def _check_system_role_support(self) -> bool:
        """Some chat templates (e.g. Gemma's) reject a system-role message.
        Probe once at load time instead of hardcoding a model-family check.
        """
        try:
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": ""}, {"role": "user", "content": ""}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception:
            return False

    def format_chat_prompt(self, system: str, user: str) -> str:
        if self.supports_system_role:
            chat = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        else:
            # Fold the system instruction into the user turn instead.
            chat = [{"role": "user", "content": f"{system}\n\n{user}"}]
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    def get_hook_filter(self) -> List[str]:
        """Hook names to cache: per-layer attention/MLP outputs, needed as the
        injectable values for edge activation patching (see attribution_patching.py).
        """
        n_layers = self.model.cfg.n_layers
        names = [
            f"blocks.{i}.{hook}"
            for i in range(n_layers)
            for hook in ["attn.qkv.hook_in", "hook_out"]
        ]
        names.append("hook_embed")
        return names

    def _checkpoint_path(self, kind: str) -> Path:
        model_slug = self.model_id.replace("/", "__")
        lang = self.config.language_code.value
        return Path(self.config.checkpoint_dir) / f"{model_slug}_{lang}_{kind}.jsonl"

    def _load_checkpoint(self, kind: str) -> List[Dict[str, Any]]:
        """Read back rows already checkpointed for this model/language/kind."""
        path = self._checkpoint_path(kind)
        if not path.exists():
            return []
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows:
            print(f"Resuming {kind}: {len(rows)} rows already checkpointed at {path}")
        return rows

    def _append_checkpoint(self, kind: str, rows: List[Dict[str, Any]]) -> None:
        """Append newly-completed rows so a crash doesn't lose this batch."""
        path = self._checkpoint_path(kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    @torch.no_grad
    def generate_response(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        assert self.model is not None, ValueError(
            "Initialize model by calling load() first"
        )
        results: List[Dict[str, Any]] = self._load_checkpoint("responses")
        completed_ids = {row["id"] for row in results}
        dataset = dataset[~dataset["id"].isin(completed_ids)].reset_index(drop=True)

        # Each row is repeated n_samples times within a batch: do_sample=True
        # gives every batch slot its own independent draw even for identical
        # inputs, so this yields n_samples stochastic completions per row in
        # a single generate() call. Shrink rows-per-batch accordingly so the
        # actual GPU batch size stays close to generation_batch_size.
        rows_per_batch = max(
            1, self.config.generation_batch_size // self.config.n_samples
        )

        for start in tqdm(
            range(0, len(dataset), rows_per_batch),
            desc="Generating responses...",
        ):
            batch_rows = dataset.iloc[start : start + rows_per_batch].to_dict(
                orient="records"
            )
            expanded_rows = [
                row for row in batch_rows for _ in range(self.config.n_samples)
            ]
            prompts = [
                self.format_chat_prompt(row["system"], row["user_query"])
                for row in expanded_rows
            ]
            input_prompt = self.tokenizer(
                prompts, padding=True, return_tensors="pt"
            ).to(self.config.device)

            outputs = self.model.original_model.generate(
                **input_prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
            )

            input_len = input_prompt["input_ids"].shape[1]
            batch_results = []
            for i, (row, output) in enumerate(zip(expanded_rows, outputs)):
                response = self.tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                )
                batch_results.append(
                    ModelResponse(
                        **row, response=response, sample_idx=i % self.config.n_samples
                    ).model_dump()
                )
            results.extend(batch_results)
            self._append_checkpoint("responses", batch_results)

            del outputs, input_prompt
            torch.cuda.empty_cache()

        return results

    @torch.no_grad
    def extract_hidden_states(self, dataset: pd.DataFrame) -> List[Dict[str, Any]]:
        assert self.model is not None, ValueError(
            "Initialize model by calling load() first"
        )
        results: List[Dict[str, Any]] = self._load_checkpoint("activations")
        completed_ids = {row["id"] for row in results}
        dataset = dataset[~dataset["id"].isin(completed_ids)].reset_index(drop=True)

        for start in tqdm(
            range(0, len(dataset), self.config.activation_batch_size),
            desc="Extracting hidden states...",
        ):
            batch_rows = dataset.iloc[
                start : start + self.config.activation_batch_size
            ].to_dict(orient="records")
            prompts = [
                self.format_chat_prompt(row["system"], row["user_query"])
                for row in batch_rows
            ]
            input_prompt = self.tokenizer(
                prompts, padding=True, return_tensors="pt"
            ).to(self.config.device)

            _, cache = self.model.run_with_cache(
                input_prompt["input_ids"],
                attention_mask=input_prompt["attention_mask"],
                names_filter=self.get_hook_filter(),
            )

            last_token_indices = (
                input_prompt["attention_mask"].flip(1).cumsum(1).bool().int().sum(1) - 1
            ).to(device=input_prompt["input_ids"].device, dtype=torch.long)
            batch_indices = torch.arange(
                len(input_prompt["input_ids"]), device=input_prompt["input_ids"].device
            )

            batch_activations = {
                name: acts[batch_indices, last_token_indices].cpu().numpy().tolist()
                for name, acts in cache.items()
            }
            batch_results = []
            for i, row in enumerate(batch_rows):
                batch_results.append(
                    {
                        "id": row["id"],
                        **{
                            name: values[i]
                            for name, values in batch_activations.items()
                        },
                    }
                )
            results.extend(batch_results)
            self._append_checkpoint("activations", batch_results)

            del cache, input_prompt
            torch.cuda.empty_cache()

        return results


def run(config: ModelGenerationConfig) -> None:
    torch.set_grad_enabled(False)
    try:
        dataset = CrossLingualRuleFollowingDataset(config.dataset_config)

        for model_name in config.model_ids:
            model_runner = ModelRunner(config)
            model_runner.load(model_id=model_name)

            if config.run_inference_response:
                result_helper = (
                    HFDataHelper(config.hf_result_repo)
                    if config.push_to_hf and config.hf_result_repo
                    else None
                )
                if result_helper and result_helper.exists(
                    model_id=model_name, lang_code=config.language_code
                ):
                    print(
                        f"Skipping response generation for {model_name}/{config.language_code}: "
                        f"already uploaded to {config.hf_result_repo}"
                    )
                else:
                    responses = model_runner.generate_response(dataset=dataset.df)
                    if result_helper:
                        result_helper.upload(
                            df=pd.DataFrame(responses),
                            model_id=model_name,
                            lang_code=config.language_code,
                        )
                    else:
                        print(
                            f"Skipping response upload for {model_name}/{config.language_code}"
                        )

            if config.run_inference_activations:
                activations_helper = (
                    HFDataHelper(config.hf_activations_repo)
                    if config.push_to_hf and config.hf_activations_repo
                    else None
                )
                if activations_helper and activations_helper.exists(
                    model_id=model_name, lang_code=config.language_code
                ):
                    print(
                        f"Skipping activation extraction for {model_name}/{config.language_code}: "
                        f"already uploaded to {config.hf_activations_repo}"
                    )
                else:
                    activations = model_runner.extract_hidden_states(dataset=dataset.df)
                    if activations_helper:
                        activations_helper.upload(
                            df=pd.DataFrame(activations),
                            model_id=model_name,
                            lang_code=config.language_code,
                        )
                    else:
                        print(
                            f"Skipping activation upload for {model_name}/{config.language_code}"
                        )

            del model_runner
            torch.cuda.empty_cache()
    except Exception:
        print("An error occurred during the generation run:")
        raise


def main(hyperparameter_path: str) -> None:
    with open(hyperparameter_path) as f:
        config = ModelGenerationConfig.model_validate_json(f.read())
    run(config=config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyperparameter-file", required=True)
    args = ap.parse_args()
    main(args.hyperparameter_file)
