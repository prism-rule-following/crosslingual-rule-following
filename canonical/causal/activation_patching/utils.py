"""Utility function for activation patching on the EAP circuit."""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Tuple,
    Union,
    cast,
)
import torch

from canonical.causal.activation_patching.evaluation_metrics import (
    MetricFn,
    logit_diff,
    make_adherence_metric,
    make_internal_state_metric,
)

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

InterventionType = Literal["mean", "patching", "zero"]
VALID_INTERVENTIONS: Tuple[InterventionType, ...] = ("mean", "patching", "zero")

METRIC_REGISTRY: Dict[str, Union[MetricFn, Callable[..., MetricFn]]] = {
    "logit_diff": logit_diff,
    "adherence": make_adherence_metric,
    # "cosine": make_internal_state_metric,
    "fidelity": make_internal_state_metric,
}


def resolve_metrics(metric_names: List[str], **metric_kwargs: Any) -> Dict[str, MetricFn]:
    """Translate metric name strings into a {name: Callable} dict, so a metric's
    function can always be looked up by its own name (e.g. resolved["logit_diff"]).

    'logit_diff' is used as-is.

    'adherence' is built via
    make_adherence_metric(checker, tokenizer) using the `checker`/`tokenizer` kwargs.

    'fidelity' loads the trained probes and evaluates them during activation patching.

    Any name not in METRIC_REGISTRY is looked up directly in metric_kwargs under its
    own name, e.g. resolve_metrics(["my_metric"], my_metric=some_callable). Raises if
    it's neither a known metric nor provided that way.
    """
    resolved: Dict[str, MetricFn] = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            custom_fn = metric_kwargs.get(name)
            if not callable(custom_fn):
                raise ValueError(
                    f"Metric {name!r} not found in METRIC_REGISTRY and no callable "
                    f"provided for it in metric_kwargs (pass {name}=<callable>)."
                )
            resolved[name] = custom_fn
            continue

        factory_or_fn = METRIC_REGISTRY[name]
        if name == "adherence":
            resolved[name] = factory_or_fn(
                metric_kwargs["checker"], metric_kwargs["tokenizer"]
            )
        elif name == "fidelity":
            pass
        else:
            resolved[name] = factory_or_fn
    return resolved


def validate_intervention(intervention: str) -> InterventionType:
    """Checks if intervention is valid."""
    if intervention not in VALID_INTERVENTIONS:
        raise ValueError(
            f"intervention must be one of {VALID_INTERVENTIONS}, got {intervention!r}"
        )
    return cast(InterventionType, intervention)


def to_serialisable(obj: Any) -> Any:
    """Recursively convert tensors and tuple keys into JSON-safe structures."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): to_serialisable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serialisable(value) for value in obj]
    return obj


def build_chat_tokenizer(
    model: "HookedTransformer",
    system_field: str = "system_rule",
    user_field: str = "user_query",
    add_generation_prompt: bool = True,
) -> Callable[[Dict[str, Any]], torch.Tensor]:
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
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
            return model.to_tokens(text, prepend_bos=False)
        # Plain concatenation if a chat template isn't available like for Gemmas
        text = f"{system_text}\n\n{user_text}".strip()
        return model.to_tokens(text)

    return tokenize_fn


def build_generator(
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    stop_at_eos: bool = True,
) -> Callable[["HookedTransformer", torch.Tensor], str]:
    """Build a `generate_fn(model, tokens) -> str` for the verification runs."""

    def generate_fn(model: "HookedTransformer", tokens: torch.Tensor) -> str:
        n_prompt = tokens.shape[-1]

        out = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            stop_at_eos=stop_at_eos,
            verbose=False,  # suppress the per-call progress bar
        )

        # model.generate returns prompt + completion; keep only the completion.
        completion_tokens = out[0, n_prompt:]
        return model.to_string(completion_tokens)

    return generate_fn
