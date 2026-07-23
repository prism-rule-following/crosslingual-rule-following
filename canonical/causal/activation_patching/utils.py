"""Utility function for activation patching on the EAP circuit."""

from typing import Any, Dict
import torch

def build_chat_tokenizer(model, system_field: str = "system_rule",
                         user_field: str = "user_query",
                         add_generation_prompt: bool = True):
    """Builds a chat tokeniser with system_prompt and user_prompt."""
    tok = model.tokenizer

    def tokenize_fn(row: Dict[str, Any]) -> torch.Tensor:
        system_text = row.get(system_field, "")
        user_text = row.get(user_field, "")
        if getattr(tok, "chat_template", None):
            messages = []
            if system_text:
                messages.append({"role": "system", "content": system_text})
            messages.append({"role": "user", "content": user_text})
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt)
            return model.to_tokens(text, prepend_bos=False)
        # Plain concatenation if a chat template isn't available like for Gemmas
        text = f"{system_text}\n\n{user_text}".strip()
        return model.to_tokens(text)

    return tokenize_fn


def build_generator(max_new_tokens: int = 64,
                    temperature: float = 0.0,
                    stop_at_eos: bool = True):
    """Build a `generate_fn(model, tokens) -> str` for the verification runs."""
    def generate_fn(model, tokens: torch.Tensor) -> str:
        n_prompt = tokens.shape[-1]

        out = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            stop_at_eos=stop_at_eos,
            verbose=False,          # suppress the per-call progress bar
        )

        # model.generate returns prompt + completion; keep only the completion.
        completion_tokens = out[0, n_prompt:]
        return model.to_string(completion_tokens)

    return generate_fn
