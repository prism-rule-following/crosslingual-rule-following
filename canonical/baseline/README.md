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
