# No-rule baseline (canonical/baseline)

Answers: does the rule in the system prompt cause rule-following? Same queries, **persona-only system prompt** (rule + deontic wording removed), judged with the same pipeline, compared against the rule run at L0.

Results: open `REPORT.pdf` (or `REPORT.html` / `REPORT.md`). Tables in `data/`, figures in `figures/`.
Generations are **not** in git (they live in the HF baseline results repo). `data/*_verdicts_*.jsonl` hold judge verdicts only.

## Layout
| file | purpose |
|---|---|
| `run_baseline_norule.py`, `run_baseline_core.sh` | generate baseline responses (en, yo, sw; Llama + Qwen) |
| `check_baseline.py` | sanity-check generations (counts, empties, duplicates, random Q&A) |
| `prep_baseline_for_judging.py` | restore the original rule text as the judge's RULE_TEXT |
| `judge_configs/` | configs for `canonical.judging.judge_{gpt_mini,deepseek,gemini}` |
| `build_consensus.py` | 3-judge consensus per response |
| `baseline_stats.py` | paired baseline-vs-rule tests per cell |
| `run_all.sh`, `make_judge_configs.py` | one-command end-to-end driver (all 13 languages, both models) |
| `analyze.py`, `make_figures.py`, `make_html.py` | judge agreement, per-judge/model/category tables, figures, report |

## Run order (from the repo root, `source /workspace/.runpod_env`, `export PYTHONPATH=.`)
```bash
# 1. generate (dry -> smoke -> full)
bash canonical/baseline/run_baseline_core.sh dry
bash canonical/baseline/run_baseline_core.sh smoke
RESULT_REPO=crosslingual-rule-following/<new-baseline-repo> bash canonical/baseline/run_baseline_core.sh full
python canonical/baseline/check_baseline.py /workspace/baseline_out 5 26

# 2. judge (one tmux window each)
python canonical/baseline/prep_baseline_for_judging.py --in /workspace/baseline_out --out /workspace/baseline_judge_in
python -m canonical.judging.judge_gpt_mini --config canonical/baseline/judge_configs/judge_gpt_mini_baseline.json
python -m canonical.judging.judge_deepseek --config canonical/baseline/judge_configs/judge_deepseek_baseline.json
python -m canonical.judging.judge_gemini   --config canonical/baseline/judge_configs/judge_gemini_baseline.json

# 3. analysis (judge outputs land in results/baseline/<judge>/final.jsonl)
python canonical/baseline/build_consensus.py --out canonical/baseline/data/baseline_scored.jsonl \
   results/baseline/gemini/final.jsonl results/baseline/deepseek/final.jsonl results/baseline/gpt_mini/final.jsonl
python canonical/baseline/analyze.py \
   --baseline gemini=results/baseline/gemini/final.jsonl gpt_mini=results/baseline/gpt_mini/final.jsonl deepseek=results/baseline/deepseek/final.jsonl \
   --rule gemini=canonical/baseline/data/rule_verdicts_gemini_L0.jsonl gpt_mini=canonical/baseline/data/rule_verdicts_gpt_mini_L0.jsonl deepseek=canonical/baseline/data/rule_verdicts_deepseek_L0.jsonl \
   --out canonical/baseline/data
python canonical/baseline/make_figures.py && python canonical/baseline/make_html.py
```
Unit of analysis is the 26 base-query cells per language; samples and wordings are repeats inside a cell.

## How the 13-language report was built
The baseline ran in two batches (en, yo, sw first; then am, de, hi, ig, it, ko, ru, ta, tr, ur). Their judge outputs were concatenated per judge and analysed together:
```bash
# per judge: cat <batch1>/results/baseline/<judge>/final.jsonl <batch2>/results/baseline/<judge>/final.jsonl > merged/<judge>.jsonl
python canonical/baseline/build_consensus.py --out canonical/baseline/data/baseline_scored.jsonl merged/gemini.jsonl merged/deepseek.jsonl merged/gpt_mini.jsonl
python canonical/baseline/analyze.py --baseline gemini=merged/gemini.jsonl gpt_mini=merged/gpt_mini.jsonl deepseek=merged/deepseek.jsonl \
   --rule gemini=canonical/baseline/data/rule_verdicts_gemini_L0.jsonl gpt_mini=canonical/baseline/data/rule_verdicts_gpt_mini_L0.jsonl deepseek=canonical/baseline/data/rule_verdicts_deepseek_L0.jsonl \
   --out canonical/baseline/data
python canonical/baseline/make_figures.py && python canonical/baseline/make_html.py   # make_html needs `pip install markdown`
```
The verdict-only files in `data/` (`baseline_verdicts_*.jsonl` + `*_batch2.jsonl`) contain the same verdicts without generations and can be used instead of `final.jsonl`.

## Capability baseline (can the models perform the act when asked directly?)
Reviewer ask: ask directly for a referral / reasoned refusal in Yoruba and Igbo (no rule), then judge whether the act is performed.
- `run_capability_act.py` generates (persona-only system prompt; the in-language rule instruction is appended to the user turn).
  Variant **A** = the rule's own instruction text (native wording, third-person rule phrasing); variant **B** = short English meta-instruction ("Reply in <Language>. ... consult <professional> / decline and explain why").
  Only `mandatory_referral` and `refuse_with_reason`, L0, 4 topics each = 8 base queries per language; default all 13 languages (yo, ig requested by reviewer; en = control; rest for breadth), 10 samples each; variant-A labels for languages beyond en/yo/ig are inferred from the data (check the dry run).
- `run_capability.sh dry|gen|judge|retry|finish|all` runs generation -> 3 judges (original rule rubric, original question restored) -> consensus/analysis.
- `analyze_capability.py` compares no-rule baseline, rule run, direct ask A and B (bootstrap CI over the 8 queries) -> `data/capability/`.
Reading: capability high but rule adherence ~0 = portability gap; both low = capacity limit. Caveats: 8 queries/language, translations not native-verified, response language is only proxied (English stop-word share), descriptive only.

### Capability baseline results and how it relates to the no-rule baseline
Report: `CAPABILITY_REPORT.pdf` / `.html` / `.md`. The two baselines are complementary and form a ladder per (model, language):
no-rule baseline (default behaviour) -> direct ask (capability ceiling, this experiment) -> rule in system prompt (rule run).
`rule - baseline` = rule effect, `direct - baseline` = what instruction can achieve at all, `direct - rule` = portability gap.
Extra scripts: `analyze_capability_gap.py` (paired gap contrasts, degeneracy and echo diagnostics, figures) and `make_capability_html.py`.
Files in `data/capability/`: `capability_vs_rule.csv`, `capability_gap.csv`, `capability_diagnostics.csv`, `capability_degeneracy.csv`, `capability_A_echo.csv`, `cap_fig_*.png`, `responses/`.
