"""Unified translator adapter.

One `Translator` instance handles one direction (forward: en→target, or
backward: target→en) for one language. Two backends are supported:

  - azure       : Azure AI Foundry deployments (GPT-4o / GPT-4o-mini / Claude / any
                  model exposed through Foundry's OpenAI-compatible Chat Completions
                  endpoint). The `model` field in YAML is the *deployment name*
                  from your Foundry project, NOT the base model ID.
  - huggingface : local seq2seq MT models — NLLB or IndicTrans2. Use `backend_mode: local`
                  and provide `src_lang`/`tgt_lang` codes in the config.

Local HF models are moved to CUDA if available; otherwise CPU.
"""
from __future__ import annotations

import os
from typing import Any


# ============================================================================
# Prompt templates (used by the Azure path — the seq2seq HF path drives
# translation via the tokenizer's src/tgt lang codes, no prompt)
# ============================================================================

FORWARD_TEMPLATE = """You are a professional translator. Translate the following English text into {target_language}.

Rules:
- Preserve meaning exactly. Do not paraphrase or explain.
- Keep HTML/XML tags, code blocks, and numeric digits (e.g. "3", "five") in their original form; do not translate tag names or convert numerals.
- Keep proper nouns and quoted words in their original form unless a natural {target_language} equivalent is standard.
- Output ONLY the translation. No preamble, no quotes, no notes.

English text:
{text}"""


BACKWARD_TEMPLATE = """You are a professional translator. Translate the following {source_language} text into English.

Rules:
- Preserve meaning exactly. Do not paraphrase or explain.
- Keep HTML/XML tags, code blocks, and numeric digits in their original form.
- Output ONLY the translation. No preamble, no quotes, no notes.

{source_language} text:
{text}"""


def forward_prompt(text: str, target_language: str) -> str:
    return FORWARD_TEMPLATE.format(text=text, target_language=target_language)


def backward_prompt(text: str, source_language: str) -> str:
    return BACKWARD_TEMPLATE.format(text=text, source_language=source_language)


# ============================================================================
# Module-level singletons (heavy objects — created once per process)
# ============================================================================

_LOCAL_MODEL_CACHE: dict[str, Any] = {}
_AZURE_CLIENT = None
_DEVICE: str | None = None


def _pick_device() -> str:
    """Return best available torch device: mps > cuda > cpu. Cached per process."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    import torch
    if torch.backends.mps.is_available():
        _DEVICE = "mps"
    elif torch.cuda.is_available():
        _DEVICE = "cuda"
    else:
        _DEVICE = "cpu"
    return _DEVICE


def _azure_client():
    """Lazy-init a single OpenAI client pointed at the Azure Foundry v1 endpoint.

    Azure AI Foundry exposes an OpenAI-compatible surface at
        https://<resource>.services.ai.azure.com/openai/v1
    which speaks the standard Chat Completions API.
    """
    global _AZURE_CLIENT
    if _AZURE_CLIENT is not None:
        return _AZURE_CLIENT
    from openai import OpenAI
    api_key = os.environ.get("AZURE_FOUNDRY_API_KEY")
    endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT")
    if not api_key or not endpoint:
        raise RuntimeError(
            "AZURE_FOUNDRY_API_KEY and AZURE_FOUNDRY_ENDPOINT must be set in .env "
            "for any language using `backend: azure` in config/translation.yaml."
        )
    _AZURE_CLIENT = OpenAI(base_url=endpoint, api_key=api_key)
    return _AZURE_CLIENT


# ============================================================================
# Translator
# ============================================================================


class Translator:
    def __init__(self, spec: dict, direction: str, target_language_name: str,
        source_language_name: str, cache):
        """
        spec: the `translator` or `backtranslator` sub-block from the language config.
        direction: "forward" (source_language → target_language) or "backward" (reverse).
        target_language_name / source_language_name: human-readable names used in LLM prompts.
        """
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be forward|backward, got {direction!r}")
        self.spec = spec
        self.direction = direction
        self.target_language_name = target_language_name
        self.source_language_name = source_language_name
        self.cache = cache

        self.model: str = spec["model"]
        self.backend: str = spec["backend"]
        self.backend_mode: str = spec.get("backend_mode", "local")
        self.src_lang: str | None = spec.get("src_lang")
        self.tgt_lang: str | None = spec.get("tgt_lang")

    def translate(self, text: str) -> str:
        """Translate a chunk of text. Uses cache first."""
        text = text.strip()
        if not text:
            return text

        key = self.cache.key_from(
            (
                self.backend,
                self.model,
                self.src_lang,
                self.tgt_lang,
                self.direction,
                self.target_language_name,
                self.source_language_name,
                text,
            )
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        if self.backend == "azure":
            result = self._azure(text)
        elif self.backend == "huggingface":
            result = self._huggingface(text)
        else:
            raise ValueError(
                f"unknown backend {self.backend!r} — supported: azure | huggingface"
            )

        result = result.strip()
        self.cache.set(key, result)
        return result

    # ------------------------------------------------------------------
    # Azure AI Foundry
    # ------------------------------------------------------------------

    def _azure(self, text: str) -> str:
        client = _azure_client()

        if self.direction == "forward":
            prompt = forward_prompt(text, self.target_language_name)
        else:
            prompt = backward_prompt(text, self.source_language_name)

        # `model` is the Foundry deployment name, not the base model ID.
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # HuggingFace — local seq2seq only (NLLB or IndicTrans2)
    # ------------------------------------------------------------------

    def _huggingface(self, text: str) -> str:
        if self.backend_mode != "local":
            raise ValueError(
                f"backend_mode {self.backend_mode!r} not supported. The Inference API "
                f"path was removed after HF deprecated api-inference.huggingface.co; "
                f"use backend_mode: local."
            )
        if not self.src_lang or not self.tgt_lang:
            raise ValueError(
                f"HF seq2seq model {self.model!r} requires src_lang and tgt_lang in config"
            )
        if self.model.startswith("ai4bharat/indictrans2-"):
            return self._hf_local_indictrans2(text)
        return self._hf_local_nllb(text)

    def _hf_local_nllb(self, text: str) -> str:
        """NLLB-200 (and any tokenizer that supports `tok.src_lang` + forced_bos)."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.model not in _LOCAL_MODEL_CACHE:
            device = _pick_device()
            tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            dtype = torch.float32 if device == "cpu" else torch.float16
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model, trust_remote_code=True, dtype=dtype
            ).to(device)
            model.eval()
            _LOCAL_MODEL_CACHE[self.model] = (tok, model, device)
        tok, model, device = _LOCAL_MODEL_CACHE[self.model]

        tok.src_lang = self.src_lang
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        forced_bos = tok.convert_tokens_to_ids(self.tgt_lang)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=512,
                num_beams=4,
            )
        return tok.batch_decode(generated, skip_special_tokens=True)[0]

    def _hf_local_indictrans2(self, text: str) -> str:
        """IndicTrans2 path — requires IndicTransToolkit's IndicProcessor.

        Install with:
          pip install --ignore-requires-python \
              "IndicTransToolkit @ git+https://github.com/VarunGumma/IndicTransToolkit.git"

        Pinned to CPU: MPS SIGTRAPs on IndicTrans2's remote modeling code.
        use_cache=False: the model's older KV-cache implementation is incompatible
        with newer transformers versions (raises 'NoneType has no shape').
        """
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError as e:
            raise RuntimeError(
                "IndicTrans2 requires IndicTransToolkit — see the installation note "
                "in this method's docstring."
            ) from e

        cache_key = f"{self.model}::indictrans"
        if cache_key not in _LOCAL_MODEL_CACHE:
            device = "cpu"  # MPS crashes; ~1B params, CPU is tolerable
            tok = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model, trust_remote_code=True, dtype=torch.float32
            ).to(device)
            model.eval()
            processor = IndicProcessor(inference=True)
            _LOCAL_MODEL_CACHE[cache_key] = (tok, model, processor, device)
        tok, model, processor, device = _LOCAL_MODEL_CACHE[cache_key]

        preprocessed = processor.preprocess_batch(
            [text], src_lang=self.src_lang, tgt_lang=self.tgt_lang
        )
        inputs = tok(
            preprocessed,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                use_cache=False,
                min_length=0,
                max_length=512,
                num_beams=5,
                num_return_sequences=1,
            )
        decoded = tok.batch_decode(
            generated.detach(), skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return processor.postprocess_batch(decoded, lang=self.tgt_lang)[0]
