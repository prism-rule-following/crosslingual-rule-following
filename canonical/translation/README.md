# Translation pipeline

Translates the English `rb_attrpatch_dataset.json` (480 rule-following pairs across 8 categories) into any configured target language. Every translated pair carries provenance and passes a five-stage quality check before landing in the main output.

## File layout

```
pipeline/client.py               Translator adapter (Azure Foundry + HuggingFace) + prompts
pipeline/translate.py            CLI orchestrator + cache + per-field policy
pipeline/categories.py           Per-category translation + checker-rewriting handlers
pipeline/validators.py           COMET/AfriCOMET + back-translation + LID + structural + checker-sanity
pipeline/test_categories.py      Unit tests for category handlers (no API calls)
config/translation.yaml          Per-language: translator/backtranslator/quality-metric/active-cancelled words
```


## How the Pipeline works
1) Take the English source pair.
2) Translate the field into the target language (rule-aware and preserves the checker's semantics).
3) Run 5 automated quality checks on the result.
4) If the checks pass → the correct ones
5) Any check that fails → the wrong ones with a diagnostic explaining which check failed and why 

The 5 checks include:

1) COMET/AfriCOMET semantic quality (source vs translation). Threshold: ≥ per-language calibrated threshold

2) Back-translation similarity: back-translate to English, compare to original. 
Threshold: chrF ≥ 0.30 AND embedding cosine ≥ 0.75

3) Language ID (fastText) on every translated field. 
Threshold: Detected language = target, OR confidence < 0.5

4) Structural invariants (category-specific). 
Threshold: e.g. active_cancelled pair must differ by exactly 1 word; bold_html rule text can't contain literal tags

5) Checker sanity:  the rewritten regex must correctly score synthetic outputs. 
Threshold: Accepts an obviously-correct sample; rejects an obviously-wrong one


## Run

```bash
# Smoke test on one language:
python -m pipeline.translate --lang de --limit 5

# Full run on one language:
python -m pipeline.translate --lang de

# Run all languages:
python -m pipeline.translate --all
```
