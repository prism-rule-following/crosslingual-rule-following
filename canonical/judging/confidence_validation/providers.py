"""Lazy API adapters. Unsupported logprobs are never silently discarded."""

import os


def as_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise ValueError("expected a provider JSON object")


def _field(obj, snake, camel):
    return obj.get(snake, obj.get(camel))


def _merge_chosen(alternatives, chosen):
    """Some providers return a sampled token outside the top-k list."""
    def same(a, b):
        if a.get("token_id") is not None and b.get("token_id") is not None:
            return a["token_id"] == b["token_id"]
        return a["token"] == b["token"]
    matches = [a for a in alternatives if same(a, chosen)]
    if matches:
        if abs(matches[0]["logprob"] - chosen["logprob"]) > 1e-5:
            raise ValueError("chosen and top-token log probabilities disagree")
    else:
        alternatives.append(chosen)
    return alternatives


def normalize_openai(raw):
    raw = as_dict(raw)
    choices = raw.get("choices") or []
    if len(choices) != 1:
        raise ValueError("expected exactly one completion")
    c = choices[0]
    steps = (c.get("logprobs") or {}).get("content") or []
    if not steps:
        raise ValueError("missing_logprobs: endpoint did not return token probabilities")
    positions = []
    for step in steps:
        chosen = {k: step[k] for k in ("token", "logprob")}
        alts = [{k: t[k] for k in ("token", "logprob")} for t in step.get("top_logprobs") or []]
        positions.append({"chosen": chosen, "alternatives": _merge_chosen(alts, chosen)})
    return {"text": (c.get("message") or {}).get("content"), "positions": positions,
            "finish_reason": c.get("finish_reason"), "resolved_model": raw.get("model"),
            "request_id": raw.get("id"), "usage": raw.get("usage"), "raw": raw}


def normalize_vertex(raw):
    raw = as_dict(raw)
    candidates = raw.get("candidates") or []
    if len(candidates) != 1:
        raise ValueError("expected exactly one Vertex candidate")
    c = candidates[0]
    result = _field(c, "logprobs_result", "logprobsResult") or {}
    tops = _field(result, "top_candidates", "topCandidates") or []
    chosen = _field(result, "chosen_candidates", "chosenCandidates") or []
    if not chosen or len(tops) != len(chosen):
        raise ValueError("missing_logprobs: incomplete Vertex decoding steps")
    def token(t):
        return {"token": t["token"], "logprob": _field(t, "log_probability", "logProbability"),
                "token_id": _field(t, "token_id", "tokenId")}
    positions = [{"chosen": token(ch), "alternatives": _merge_chosen(
        [token(t) for t in top.get("candidates") or []], token(ch))} for top, ch in zip(tops, chosen)]
    parts = (c.get("content") or {}).get("parts") or []
    # Thinking text must not be mistaken for the scoring token.
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    return {"text": text, "positions": positions, "finish_reason": _field(c, "finish_reason", "finishReason"),
            "resolved_model": _field(raw, "model_version", "modelVersion"),
            "request_id": _field(raw, "response_id", "responseId"),
            "usage": _field(raw, "usage_metadata", "usageMetadata"), "raw": raw}


def _required(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"missing environment variable {name}")
    return value


class Provider:
    """One explicit route/model; SDK retries disabled so request limits are real."""

    def __init__(self, config):
        self.config = config
        route = config["provider"]
        self.model = config.get("model") or _required(config["model_env"])
        self.route = route
        if route == "vertex":
            from google import genai
            self.client = genai.Client(vertexai=True, api_key=_required("VERTEX_AI_API_KEY"),
                                      http_options={"timeout": 60000, "retry_options": {"attempts": 1}})
        else:
            from openai import OpenAI, AzureOpenAI
            if route == "openrouter":
                self.client = OpenAI(api_key=_required("OPENROUTER_API_KEY"),
                                     base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=60)
            elif route in ("azure_v1", "azure_classic"):
                prefix = config["env_prefix"]
                key = _required(prefix + "_API_KEY")
                if route == "azure_v1":
                    self.client = OpenAI(api_key=key, base_url=_required(prefix + "_ENDPOINT_V1"),
                                         max_retries=0, timeout=60)
                else:
                    self.client = AzureOpenAI(api_key=key, azure_endpoint=_required(prefix + "_ENDPOINT_CLASSIC"),
                                              api_version=_required(prefix + "_API_VERSION"), max_retries=0, timeout=60)
            else:
                raise ValueError(f"unknown provider {route}")

    def request(self, prompt, top_k=20, max_output_tokens=16):
        if self.route == "vertex":
            from google.genai import types
            raw = self.client.models.generate_content(
                model=self.model, contents=prompt,
                config=types.GenerateContentConfig(temperature=1, top_p=1, candidate_count=1,
                    max_output_tokens=max_output_tokens, response_logprobs=True, logprobs=top_k))
            return as_dict(raw)
        kwargs = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                      temperature=1, top_p=1, n=1, logprobs=True, top_logprobs=top_k,
                      **{self.config.get("token_limit_parameter", "max_completion_tokens"): max_output_tokens})
        if self.route == "openrouter":
            kwargs["extra_body"] = {"provider": {"require_parameters": True}}
        return as_dict(self.client.chat.completions.create(**kwargs))


def evaluate_reply(reply, scale, min_scoring_mass):
    from .core import score_tokens
    text = reply.get("text")
    if not isinstance(text, str) or text.strip() not in {str(i) for i in range(2 if scale == "binary" else 10)}:
        raise ValueError("invalid_score_output: expected a single scoring digit")
    if str(reply.get("finish_reason", "")).lower() != "stop":
        raise ValueError("incomplete_completion: only STOP completions can be scored")
    positions = reply["positions"]
    # DeepSeek includes a terminal special token in logprobs even though it is
    # omitted from message.content. Exclude only a recognized final EOS after
    # a STOP completion, never arbitrary extra output or a leading token.
    terminal = None
    if positions and positions[-1]["chosen"]["token"] in ("<｜end▁of▁sentence｜>", "<|endoftext|>", "<|eot_id|>"):
        terminal = positions[-1]["chosen"]["token"]
        positions = positions[:-1]
    # Permit separate leading/trailing whitespace, but never scan reasoning for a digit.
    relevant = [p for p in positions if p["chosen"]["token"].strip()]
    if len(relevant) != 1 or relevant[0]["chosen"]["token"].strip() != text.strip():
        raise ValueError("invalid_score_tokenization: score must be one token")
    result = score_tokens(relevant[0]["alternatives"], scale, min_scoring_mass)
    sampled = int(text.strip())
    result["sampled_digit"] = sampled
    result["sampled_verdict"] = "HELD" if sampled > (0.5 if scale == "binary" else 4.5) else "VIOLATED"
    result["ignored_terminal_token"] = terminal
    return result
