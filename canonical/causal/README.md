# causal

AtrP* localization, multi-layer activation patching, ablation, and engagement (proportion mediated) scoring — Stage One §3.3.

# Activation Patching to Confirm Discovered EAP Circuits

A circuit is judged by three community-standard criteria: faithful (reproduce the behavior), complete (contain every component the model uses), and minimal (every component is necessary) (arxiv 2607.01940, from Wang et al. 2022). The experiments map to these, in this order:

**Step 0** pick the circuit from EAP scores. Greedy top-k by absolute score, binary-search the size: search for the minimal circuit that achieves at least 80% of the whole model's performance ... using binary search over circuit sizes (arxiv 2407.10827). Absolute score (not just positive) so you catch negative/auxiliary edges (arxiv 2403.17806).

**Step 1** SUFFICIENCY (denoising). Keep ONLY the circuit, patch everything else to corrupt, check the behavior survives. Formally: Faithfulness (Sufficiency): Can the model reproduce the specific behavior relying solely on the isolated sub-circuit with top-k edges retained? — (Δ_circuit_only − Δ_corrupt)/(Δ_clean − Δ_corrupt) × 100% (arxiv 2604.01457). This is denoising: restoring a corrupt component to its clean value to test sufficiency (arxiv 2606.06267).

**Step 2** NECESSITY (noising). Ablate the circuit, check the behavior drops. Completeness (Necessity): To what extent does severing the sub-circuit remove the signal, rather than allowing it to persist through alternative pathways? — (Δ_clean − Δ_ablate_circuit)/Δ_clean (arxiv 2604.01457). This is noising: replacing a clean component with its corrupt value to test necessity (arxiv 2606.06267).

**Step 3** COMPLETENESS / self-repair check. The gap between sufficiency and necessity is the self-repair signal. Self-repair is precisely a completeness failure (arxiv 2607.01940). If necessity is weak (ablating the circuit doesn't kill the behavior) despite high sufficiency, backups are compensating — you must ablate the circuit plus its backups together (conditional co-ablation). This is exactly your Hydra thread, and it's a named method now (arxiv 2607.01940).

**Step 4** MINIMALITY. Confirm each component matters: drop components one at a time; if removing one doesn't hurt, it wasn't necessary. every component is necessary (arxiv 2607.01940).