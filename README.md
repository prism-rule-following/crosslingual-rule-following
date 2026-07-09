# Crosslingual Rule-Following

Research codebase for **"Failure Modes and Enforcement of Rule-Following in Multilingual Language Models"** (working draft, July 2026).

## Research questions

- **RQ1 — failure localization.** When a model violates a rule in a low-resource language, is it a *fidelity* failure (never encoded), an *acceptance* failure (encoded but causally inert), or an *execution* failure (encoded, engaged, but insufficient)?
- **RQ2 — the transformation spectrum.** What is the minimal transformation `T` (identity / orthogonal / affine / nonlinear / none) that maps one language's rule representation onto another's, and can that transformation be used as an enforcement mechanism?

## Pipeline

**Stage One — Rule-Following Representation Identification:** behavioural adherence → presence (decodability probing) → engagement (AtrP*-localized causal patching/ablation) → failure taxonomy.

**Stage Two — Rule-Following Enforcement:** representation transport (Procrustes / affine / Gromov-Wasserstein OT) → routing → inference-time intervention.

Models: Gemma-2-2B-IT, Qwen3-8B. Languages: German, Italian, Russian, Mandarin, Korean/Japanese, Turkish, Hindi, Tamil, Swahili, Yoruba, Igbo, Quechua, Amharic.

## Repository layout

- [`canonical/`](canonical/) — validated, stable implementation of each pipeline stage; what confirmatory results are drawn from.
- [`experimental/`](experimental/) — prototypes and in-progress ideas, not yet validated; graduates into `canonical/` once stable.

Both share the same internal structure — see each folder's README for the per-stage sub-folder breakdown.

License: TBD.
