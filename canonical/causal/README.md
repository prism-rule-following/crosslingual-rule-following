# Activation Patching to Confirm Discovered EAP Circuits

A circuit is judged by three community-standard criteria: faithful (reproduce the behavior), complete (contain every component the model uses), and minimal (every component is necessary) (arxiv 2607.01940, from Wang et al. 2022). The experiments map to these, in this order:


**Step 1** SUFFICIENCY (denoising). Keep ONLY the circuit, patch everything else to corrupt, check the behavior survives. Faithfulness (Sufficiency): Can the model reproduce the specific behavior relying solely on the isolated sub-circuit with top-k edges retained? (arxiv 2604.01457). This is denoising: restoring a corrupt component to its clean value to test sufficiency (arxiv 2606.06267). Computed in `activation_patching/circuit_verifier.py::CircuitVerifier.verify_sufficiency`, which evaluates the graph with only the circuit's edges left in-graph and normalises against the clean/corrupt baselines.

**Step 2** NECESSITY (noising). Ablate the circuit, check the behaviour drops. Completeness (Necessity): To what extent does severing the sub-circuit remove the signal, rather than allowing it to persist through alternative pathways? (arxiv 2604.01457). This is noising: replacing a clean component with its corrupt value to test necessity (arxiv 2606.06267). Computed in `activation_patching/circuit_verifier.py::CircuitVerifier.verify_necessity`, which flips `in_graph` to the circuit's complement (ablate only the circuit, keep everything else) and normalises the same way.

**Step 3** COMPLETENESS / self-repair check. The gap between sufficiency and necessity is the self-repair signal. Self-repair is precisely a completeness failure (arxiv 2607.01940). If necessity is weak (ablating the circuit doesn't kill the behavior) despite high sufficiency, backups are compensating — you must ablate the circuit plus its backups together (conditional co-ablation). This is exactly your Hydra thread, and it's a named method now (arxiv 2607.01940). Computed in `circuit_verifier.py::CircuitVerifier.verify_completeness`/`incompleteness_score`, which greedily grows a worst-case backup edge set (via `node_sampling.py::sample_greedy`) that maximises the gap between the circuit-minus-edge_set and model-minus-edge_set behaviour.

**Step 4** MINIMALITY. Confirm each component matters: drop components one at a time; if removing one doesn't hurt, it wasn't necessary (arxiv 2607.01940). Computed in `circuit_verifier.py::CircuitVerifier.verify_minimality`, which knocks out each circuit edge individually and records the drop from the clean baseline per edge.

# Structure

- `activation_patching/`
  - `circuit_verifier.py` — the `CircuitVerifier` class: sufficiency/necessity/completeness/minimality, all built on top of `evaluate_with_edge_mask` (a thin wrapper around `eap.evaluate.evaluate_graph`).
  - `node_sampling.py` — edge-sampling strategies used as ablation controls: `sample_random_nodes`, `sample_functional_nodes`, and the greedy backup-search `sample_greedy` used by completeness.
  - `evaluation_metrics.py` — the metric callables passed to `evaluate_graph`/`evaluate_baseline`: `logit_difference` (`logit_diff`), `make_adherence_metric`, `make_internal_state_metric`.
  - `dataloaders.py` — `(clean, corrupted, label)` CSV dataloader for evaluation, and a small neutral-sentence dataloader for the `intervention="mean"` baseline.
  - `utils.py` — shared helpers: intervention validation, metric-name resolution (`resolve_metrics`), JSON serialisation of results, chat-tokenizer/generator builders.
  - `run_verification.py` — runs all four checks in order into one JSON report; also the CLI entry point (`python -m canonical.causal.activation_patching.run_verification ...`).
- `tests/`
  - `unit/` — pure-logic tests (masks, metrics, resolution helpers), no real model.
  - `integration/` — tests against the real `attn-only-1l` toy model; some require CUDA (skipped automatically without a GPU).
  - `conftest.py` — shared fixtures.
  - `colab_test_runner.ipynb` — clones the repo and runs the full suite on a Colab GPU runtime.

# How to Run

Install (from `crosslingual-rule-following/`): `pip install -e ".[test]"`.

Run the tests: `pytest` (CUDA-only integration tests skip automatically without a GPU; use `colab_test_runner.ipynb` for those).

Run verification from the command line, given a model name, a saved circuit (`Graph.to_json` output), and a CSV of `clean`/`corrupted`/`label` rows:

```bash
python -m canonical.causal.activation_patching.run_verification \
    attn-only-1l circuit.json eval_data.csv \
    --metrics logit_diff --intervention patching --out results.json
```