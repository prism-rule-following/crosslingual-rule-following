"""Translator: one OpenRouter call per source string."""

from __future__ import annotations

import os

DEFAULT_MODEL = "anthropic/claude-opus-5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


PROMPT = """You are a professional translator. Translate the English text into {target_language}.

These texts are rules and user messages from an AI assistant evaluation benchmark.
Translation accuracy directly affects measurement, so follow these rules exactly:

- Preserve meaning exactly. Do not paraphrase, soften, explain, or add anything.
- PRESERVE GRAMMATICAL FORCE. Bare imperative, modal obligation and polite request are
  three deliberately distinct forms and a measured variable. Render each distinctly; if
  the language cannot mark all three, use the closest distinct forms, never collapse two.
- Do not introduce a grammatical gender that the English does not have.
{register_rule}- Keep numerals, units, and drug names in their original form.
- Do not add a trailing period unless the English has one.
- Output ONLY the translation. No preamble, no quotes, no notes."""

# Rendered into PROMPT from the language config. See config/translation.yaml for
# why each language has the value it does 
_REGISTER_CLAUSES = {
    "informal": "- ADDRESS FORM: address the assistant with the informal second person.\n",
    "formal": "- ADDRESS FORM: address the assistant with the formal second person.\n",
    "none": "",  # the language draws no such distinction; say nothing
}


def register_rule(lang_config: dict) -> str:
    """The address-form line for a language, or empty if it has no such distinction.
    """
    register = (lang_config or {}).get("register", "none")
    if register not in _REGISTER_CLAUSES:
        raise ValueError(
            f"unknown register {register!r} — expected one of "
            f"{sorted(_REGISTER_CLAUSES)} in config/translation.yaml"
        )
    return _REGISTER_CLAUSES[register]

_CLIENT = None


def _client():
    """OpenRouter client, created once per process."""
    global _CLIENT
    if _CLIENT is None:
        from openai import OpenAI

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env or export it."
            )
        _CLIENT = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return _CLIENT


class Translator:
    """Translates English into one target language"""

    PROMPT_VERSION = "prompt-v4"

    def __init__(self, target_language_name: str, cache=None, model: str = DEFAULT_MODEL,
                 lang_config: dict | None = None):
        self.target_language_name = target_language_name
        self.cache = cache
        self.model = model
        self.register_rule = register_rule(lang_config)

    def translate(self, text: str) -> str:
        if not text.strip():
            return text

        key = None
        if self.cache is not None:
            key = self.cache.key_from(
                ["openrouter", self.model, self.PROMPT_VERSION,
                 self.target_language_name, self.register_rule, text]
            )
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        result = self._call(text)

        if self.cache is not None and key is not None:
            self.cache.set(key, result)
        return result

    def _call(self, text: str) -> str:
        # No temperature/top_p/top_k: claude-opus-5 rejects them with a 400.
        response = _client().chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": PROMPT.format(
                        target_language=self.target_language_name,
                        register_rule=self.register_rule,
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(f"empty response translating {text!r}")
        return content.strip()
