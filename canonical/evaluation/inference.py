"""Run generation and/or activation-extraction inference passes over the
cross-lingual rule-following dataset for one or more models, optionally
uploading results (responses, activations) to the Hugging Face Hub.
"""

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoTokenizer

import transformer_lens.utilities as utils
from transformer_lens.model_bridge import TransformerBridge

from canonical.model.dataset import (
    Category,
    Checker,
    CrossLingualRuleFollowingDataset,
    DatasetConfig,
    DatasetLanguageCode,
    GrammarType,
    HFDataHelper,
    PairType,
    PressureLevel,
    PressureName,
    RuleStatus,
    Topic,
    nan_to_none,
)

load_dotenv()


# --------------------------------------------------------------------------- #
# Activation hook groups
# --------------------------------------------------------------------------- #
# Each group maps to one file on the Hub. `hook_embed` is a single top-level
# hook; every other group is a per-layer hook (suffix appended to blocks.{i}.)
# and is stored as a (n_rows, n_layers, *hook_shape) array.
PROBING_HOOK_SUFFIXES = ["hook_resid_post", "hook_attn_out", "hook_mlp_out"]
# TransformerLens 3.x/TransformerBridge splits the old concatenated
# `attn.qkv.hook_in` into per-QKV pre-projection input hooks; use those.
PATCHING_HOOK_SUFFIXES = [
    "attn.q.hook_in",
    "attn.k.hook_in",
    "attn.v.hook_in",
    "hook_out",
]

# (file group name, per-layer hook suffix, or "" for the top-level hook_embed)
ACTIVATION_GROUPS: List[tuple] = [
    ("hook_embed", ""),
    *[(s, s) for s in PROBING_HOOK_SUFFIXES],
    *[
        ("attn_q_input", "attn.q.hook_in"),
        ("attn_k_input", "attn.k.hook_in"),
        ("attn_v_input", "attn.v.hook_in"),
        ("hook_out", "hook_out"),
    ],
]

# Columns carried in the activations index so a probe can be built without
# touching the responses file: labels + the join key back to responses.
ACTIVATION_INDEX_COLUMNS = [
    "id",
    "rule_status",
    "grammar_type",
    "category",
    "topic",
    "pair_type",
    "pressure_level",
    "pressure_name",
    "language",
]


def _activation_filename(group: str, dtype: str) -> str:
    return f"{group}.{dtype}.npy"


def _activation_hf_dir(model_id: str, lang_code: "DatasetLanguageCode") -> str:
    return f"{model_id.replace('/', '__')}/{lang_code.value}"


@dataclass
class ActivationOutput:
    """Assembled activations for one (model, language), ready to upload.

    arrays: group name -> fp16 numpy array, row order == index row order.
    index:  DataFrame of label columns + row_idx (join key / array row order).
    n_rows: number of dataset rows the arrays/index cover.
    """

    arrays: Dict[str, "np.ndarray"]
    index: "pd.DataFrame"
    n_rows: int


class ModelResponse(BaseModel):
    """One model-generated response, keyed back to its source dataset row."""

    id: str = Field(..., description="row id from the source dataset")
    model_id: str = Field(
        ..., description="HuggingFace model id that generated this response"
    )
    category: Category = Field(..., description="rule category of the source row")
    topic: Topic = Field(..., description="topic of the source row")
    grammar_type: GrammarType = Field(..., description="grammar type of the source row")
    language: DatasetLanguageCode = Field(..., description="language of the source row")
    system: str = Field(..., description="System prompt for the row")
    user_query: str = Field(
        ..., description="user query the model was asked to respond to"
    )
    response: str = Field(..., description="model-generated response text")
    sample_idx: int = Field(
        default=0,
        description="index of this stochastic sample among the n_samples "
        "generated for this prompt row",
    )
    pair_type: PairType = Field(
        ..., description="contrastive pair type of the source row"
    )
    rule_status: RuleStatus = Field(
        ...,
        description="which side of the pair this response was generated under "
        "(active_status if the rule binds, revoked_status if it's lifted)",
    )
    checker: Checker = Field(
        ...,
        description="the checker matching rule_status - active_checker if "
        "rule_status is active-family, revoked_checker if revoked-family",
    )
    pressure_level: PressureLevel = Field(
        ...,
        description="pressure_level of the source row",
    )
    pressure_name: PressureName = Field(
        ...,
        description="pressure_name of the source row",
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
    language_codes: List[DatasetLanguageCode] = Field(
        ...,
        description="languages to run inference for; each loaded model is reused "
        "across all of them, uploading a separate output file per language",
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
    activation_dtype: Literal["float16", "float32"] = Field(
        default="float16",
        description="dtype to store cached activations in; float16 halves size "
        "and is sufficient for linear probes / logit-lens",
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
    enable_thinking: Optional[bool] = Field(
        default=None,
        description="chat-template toggle for reasoning-capable models (e.g. "
        "Qwen3); None leaves the tokenizer default. Set false to run without "
        "the thinking block (injects ' thinking\\n\\n response\\n\\n' for Qwen3).",
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

        dtype = torch.bfloat16 if self.config.device == "cuda" else torch.float32
        self.model = TransformerBridge.boot_transformers(
            model_id, device=self.config.device, dtype=dtype
        )
        self.model.enable_compatibility_mode(disable_warnings=True, no_processing=True)
        self.model.original_model.eval()
        # Required for patching hooks (qkv_input) to actually be populated during run_with_cache.
        self.model.cfg.use_attn_result = True
        self.model.cfg.use_split_qkv_input = True
        self.model.cfg.use_hook_mlp_in = True
        self.supports_system_role = self._check_system_role_support()
        self._validate_hooks()

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
        kwargs: Dict[str, Any] = {}
        if self.config.enable_thinking is not None:
            kwargs["enable_thinking"] = self.config.enable_thinking
        return self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def get_probing_hooks(self) -> List[str]:
        """Probing hooks: residual stream + per-layer attention/MLP outputs at
        the decision position, for linear probes / logit lens."""
        n_layers = self.model.cfg.n_layers
        return ["hook_embed"] + [
            f"blocks.{i}.{hook}"
            for i in range(n_layers)
            for hook in PROBING_HOOK_SUFFIXES
        ]

    def get_patching_hooks(self) -> List[str]:
        """Patching hooks: injectable values for edge activation patching."""
        n_layers = self.model.cfg.n_layers
        return [
            f"blocks.{i}.{hook}"
            for i in range(n_layers)
            for hook in PATCHING_HOOK_SUFFIXES
        ]

    def get_hook_filter(self) -> List[str]:
        """All hooks to cache (probing + patching). Kept as one list so
        run_with_cache does a single forward pass for both."""
        return self.get_probing_hooks() + self.get_patching_hooks()

    def _group_hook_names(self) -> Dict[str, List[str]]:
        """Map file group name -> full hook names (single for hook_embed,
        one per layer otherwise)."""
        n_layers = self.model.cfg.n_layers
        groups: Dict[str, List[str]] = {}
        for group, suffix in ACTIVATION_GROUPS:
            if not suffix:
                groups[group] = ["hook_embed"]
            else:
                groups[group] = [f"blocks.{i}.{suffix}" for i in range(n_layers)]
        return groups

    def _validate_hooks(self) -> None:
        """Warn about requested hooks absent from the model's hook_dict (names
        are architecture-dependent; some may not exist for a given model)."""
        hook_dict = getattr(self.model, "hook_dict", None) or {}
        if not hook_dict:
            return
        missing = [
            h for names in self._group_hook_names().values() for h in names
            if h not in hook_dict
        ]
        if missing:
            print(
                f"  ! WARNING: {len(missing)} requested hooks absent from hook_dict "
                "(architecture-specific; they will be skipped):"
            )
            for h in missing[:12]:
                print(f"      - {h}")
            if len(missing) > 12:
                print(f"      ... and {len(missing) - 12} more")

    def _checkpoint_path(self, lang_code: DatasetLanguageCode, kind: str) -> Path:
        model_slug = self.model_id.replace("/", "__")
        return (
            Path(self.config.checkpoint_dir)
            / f"{model_slug}_{lang_code.value}_{kind}.jsonl"
        )

    def _load_checkpoint(
        self, lang_code: DatasetLanguageCode, kind: str
    ) -> List[Dict[str, Any]]:
        """Read back rows already checkpointed for this model/language/kind."""
        path = self._checkpoint_path(lang_code, kind)
        if not path.exists():
            return []
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if rows:
            print(f"Resuming {kind}: {len(rows)} rows already checkpointed at {path}")
        return rows

    def _append_checkpoint(
        self, lang_code: DatasetLanguageCode, kind: str, rows: List[Dict[str, Any]]
    ) -> None:
        """Append newly-completed rows so a crash doesn't lose this batch."""
        path = self._checkpoint_path(lang_code, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def clear_checkpoint(self, lang_code: DatasetLanguageCode, kind: str) -> None:
        """Delete the local checkpoint file after its rows have been safely
        uploaded to the Hub, keeping the persistent volume free of
        already-persisted checkpoints. Only called after a successful upload
        (see run()) — if upload is skipped or fails, the checkpoint is kept."""
        path = self._checkpoint_path(lang_code, kind)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # Activation checkpointing (numpy shards + id manifest)
    # ------------------------------------------------------------------ #
    def _activation_dir(self, lang_code: DatasetLanguageCode) -> Path:
        model_slug = self.model_id.replace("/", "__")
        return (
            Path(self.config.checkpoint_dir)
            / f"{model_slug}_{lang_code.value}_activations"
        )

    def _activation_shards_dir(self, lang_code: DatasetLanguageCode) -> Path:
        return self._activation_dir(lang_code) / "shards"

    def _activation_done_path(self, lang_code: DatasetLanguageCode) -> Path:
        return self._activation_dir(lang_code) / "done.jsonl"

    def _activation_done_count(self, lang_code: DatasetLanguageCode) -> int:
        """Number of rows already checkpointed. Rows are processed in dataset
        order and written to the manifest in order, so the count is the prefix
        length already persisted (and the resume offset)."""
        p = self._activation_done_path(lang_code)
        if not p.exists():
            return 0
        return sum(1 for _ in open(p))

    def _append_activation_done(
        self, lang_code: DatasetLanguageCode, ids: List[str]
    ) -> None:
        p = self._activation_done_path(lang_code)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for i in ids:
                f.write(i + "\n")

    def _write_activation_shard(
        self, lang_code: DatasetLanguageCode, group: str, start_idx: int, arr: "np.ndarray"
    ) -> None:
        """Write one batch's array for a group. Keyed by the absolute row
        offset so a re-processed batch overwrites the same file (no duplicates)."""
        shards_dir = self._activation_shards_dir(lang_code)
        shards_dir.mkdir(parents=True, exist_ok=True)
        np.save(shards_dir / f"{group}_{start_idx:07d}.npy", arr)

    def _assemble_activations(
        self, lang_code: DatasetLanguageCode
    ) -> Dict[str, "np.ndarray"]:
        """Concatenate per-group shards in row order into the final arrays."""
        shards_dir = self._activation_shards_dir(lang_code)
        arrays: Dict[str, "np.ndarray"] = {}
        for group in self._group_hook_names():
            files = sorted(
                shards_dir.glob(f"{group}_*.npy"),
                key=lambda p: int(p.stem.rsplit("_", 1)[1]),
            )
            if files:
                arrays[group] = np.concatenate(
                    [np.load(f) for f in files], axis=0
                )
        return arrays

    def clear_activation_checkpoint(self, lang_code: DatasetLanguageCode) -> None:
        """Remove the whole activation checkpoint dir (shards + manifest)
        after its data has been uploaded to the Hub."""
        import shutil

        shutil.rmtree(self._activation_dir(lang_code), ignore_errors=True)

    def _build_activation_index(self, dataset: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ACTIVATION_INDEX_COLUMNS if c in dataset.columns]
        index = dataset[cols].reset_index(drop=True)
        index.insert(0, "row_idx", range(len(index)))
        return index

    def upload_activations(
        self,
        helper: "HFDataHelper",
        act: "ActivationOutput",
        lang_code: DatasetLanguageCode,
    ) -> None:
        """Upload assembled activations: index.parquet + one .npy per group,
        under {model_slug}/{lang}/ in the activations repo."""
        hf_dir = _activation_hf_dir(self.model_id, lang_code)
        dtype_label = "fp16" if self.config.activation_dtype == "float16" else "fp32"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            idx_path = tmp_path / "index.parquet"
            act.index.to_parquet(idx_path, index=False)
            helper.upload_file(idx_path, f"{hf_dir}/index.parquet")
            for group, arr in act.arrays.items():
                filename = _activation_filename(group, dtype_label)
                arr_path = tmp_path / filename
                np.save(arr_path, arr)
                helper.upload_file(arr_path, f"{hf_dir}/{filename}")

    @torch.no_grad
    def generate_response(
        self, dataset: pd.DataFrame, lang_code: DatasetLanguageCode
    ) -> List[Dict[str, Any]]:
        assert self.model is not None, ValueError(
            "Initialize model by calling load() first"
        )
        results: List[Dict[str, Any]] = self._load_checkpoint(lang_code, "responses")
        completed_ids = {row["id"] for row in results}
        dataset = dataset[~dataset["id"].isin(completed_ids)].reset_index(drop=True)

        rows_per_batch = max(
            1, self.config.generation_batch_size // self.config.n_samples
        )

        for start in tqdm(
            range(0, len(dataset), rows_per_batch),
            desc="Generating responses...",
        ):
            batch_rows = [
                nan_to_none(row)
                for row in dataset.iloc[start : start + rows_per_batch].to_dict(
                    orient="records"
                )
            ]
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
                        **row,
                        model_id=self.model_id,
                        response=response,
                        sample_idx=i % self.config.n_samples,
                    ).model_dump()
                )
            results.extend(batch_results)
            self._append_checkpoint(lang_code, "responses", batch_results)

            del outputs, input_prompt
            torch.cuda.empty_cache()

        return results

    @torch.no_grad
    def extract_hidden_states(
        self, dataset: pd.DataFrame, lang_code: DatasetLanguageCode
    ) -> ActivationOutput:
        """Extract last-token activations for every hook group, checkpointing
        per-batch numpy shards so a crash resumes from the last completed
        batch. Returns assembled fp16 arrays + a label index ready to upload."""
        assert self.model is not None, ValueError(
            "Initialize model by calling load() first"
        )
        dataset = dataset.reset_index(drop=True)
        dtype = (
            torch.float16
            if self.config.activation_dtype == "float16"
            else torch.float32
        )

        done_count = self._activation_done_count(lang_code)
        done_count = min(done_count, len(dataset))
        if done_count:
            print(f"Resuming activations: {done_count}/{len(dataset)} rows already done")
        remaining = dataset.iloc[done_count:]

        for start in tqdm(
            range(0, len(remaining), self.config.activation_batch_size),
            desc="Extracting hidden states...",
        ):
            global_start = done_count + start
            batch_rows = remaining.iloc[
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

            for group, hook_names in self._group_hook_names().items():
                present = [h for h in hook_names if h in cache]
                if not present:
                    continue
                if group == "hook_embed":
                    arr = cache["hook_embed"][batch_indices, last_token_indices]
                else:
                    arr = torch.stack(
                        [cache[h][batch_indices, last_token_indices] for h in present],
                        dim=1,
                    )
                arr = arr.to(dtype).cpu().numpy()
                self._write_activation_shard(lang_code, group, global_start, arr)

            self._append_activation_done(
                lang_code, [r["id"] for r in batch_rows]
            )

            del cache, input_prompt
            torch.cuda.empty_cache()

        arrays = self._assemble_activations(lang_code)
        index = self._build_activation_index(dataset)
        return ActivationOutput(arrays=arrays, index=index, n_rows=len(dataset))


def run(config: ModelGenerationConfig) -> None:
    torch.set_grad_enabled(False)
    try:
        dataset = CrossLingualRuleFollowingDataset(config.dataset_config)

        result_helper = (
            HFDataHelper(config.hf_result_repo)
            if config.push_to_hf and config.hf_result_repo
            else None
        )
        activations_helper = (
            HFDataHelper(config.hf_activations_repo)
            if config.push_to_hf and config.hf_activations_repo
            else None
        )

        for model_name in config.model_ids:
            model_runner = ModelRunner(config)
            model_runner.load(model_id=model_name)

            for lang_code in config.language_codes:
                lang_dataset = dataset.subset(language=lang_code.value)

                if config.run_inference_response:
                    if result_helper and result_helper.exists(
                        model_id=model_name, lang_code=lang_code
                    ):
                        print(
                            f"Skipping response generation for {model_name}/{lang_code}: "
                            f"already uploaded to {config.hf_result_repo}"
                        )
                    else:
                        responses = model_runner.generate_response(
                            dataset=lang_dataset.df, lang_code=lang_code
                        )
                        if result_helper:
                            result_helper.upload(
                                df=pd.DataFrame(responses),
                                model_id=model_name,
                                lang_code=lang_code,
                            )
                            model_runner.clear_checkpoint(lang_code, "responses")
                        else:
                            print(
                                f"Skipping response upload for {model_name}/{lang_code}"
                            )

                if config.run_inference_activations:
                    act_index_path = (
                        f"{_activation_hf_dir(model_name, lang_code)}/index.parquet"
                    )
                    if activations_helper and activations_helper.exists_path(
                        act_index_path
                    ):
                        print(
                            f"Skipping activation extraction for {model_name}/{lang_code}: "
                            f"already uploaded to {config.hf_activations_repo}"
                        )
                    else:
                        activations = model_runner.extract_hidden_states(
                            dataset=lang_dataset.df, lang_code=lang_code
                        )
                        if activations_helper:
                            model_runner.upload_activations(
                                activations_helper, activations, lang_code
                            )
                            model_runner.clear_activation_checkpoint(lang_code)
                        else:
                            print(
                                f"Skipping activation upload for {model_name}/{lang_code}"
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
