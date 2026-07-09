# canonical

Validated, stable implementation of the rule-following pipeline. Code here is what confirmatory analyses (C1–C4) and reported results are drawn from — changes should be deliberate and backwards-compatible with existing results where possible.

New methods start in [`../experimental/`](../experimental/) and move here once validated on held-out data.

## Sub-folders

- `data/` — rule generators, ACTIVE/REVOKED pair construction, per-language datasets, translation/HITL pipeline.
- `models/` — model + activation-hook wrappers for Gemma-2-2B-IT and Qwen3-8B.
- `evaluation/` — checker functions, calibrated LLM judge, behavioural adherence scoring (Stage One §3.1).
- `probing/` — presence/decodability classifiers, cross-lingual transfer (Stage One §3.2).
- `causal/` — AtrP* localization, activation patching, ablation, engagement scoring (Stage One §3.3).
- `transport/` — Procrustes / affine / Gromov-Wasserstein OT transformations and routing (Stage Two §4).
- `analysis/` — confirmatory (C1–C4) and exploratory analyses, FDR correction (§5).
- `configs/` — model/language/experiment configuration files.
- `notebooks/` — finalized, reproducible analysis notebooks.
- `results/` — figures, tables, cached outputs backing reported findings.
