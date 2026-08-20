# Translation pipeline

Translates `data/source/judgment_rules_expanded.verified.json` (2,340 judgment-tier rule-following
items) into any configured target language, and produces a review set for native-speaker
verification.

## How it works

The 2,340 items are a cross-product built from only **214 distinct strings**. The pipeline
translates that distinct set and **composes** the items from it, rather than translating
each item.

That makes two properties true by construction rather than checked after the fact:

- a rule clause renders identically in every item that contains it
- a pressure prefix is byte-identical across all items at that level

It also means the translation cost is ~214 calls per language instead of 2,340, and that
a reviewer correction to one string propagates to every item built from it.

**The string table is the source of truth.** The item file is derived and regenerated on
every run, so merging corrections is: overlay, rebuild, write — at zero API cost.

## File layout

```
data/source/                the English source dataset
pipeline/client.py          Translator — one OpenRouter call per string + the prompt
pipeline/compose.py         decompose / build_rows 
pipeline/validators.py      checks a reviewer structurally cannot do
pipeline/translate.py       CLI and per-language orchestration
config/translation.yaml     per-language status words and yes/no labels
```

## Run

```bash
# Smoke test — translates 20 strings
python -m pipeline.translate --lang ig --limit-strings 20

# Full run for one language
python -m pipeline.translate --lang ig

# All 12 languages
python -m pipeline.translate --all

# Rebuild items from saved strings and corrections
python -m pipeline.translate --lang ig --compose-only
```


Corrections come back keyed by string hash:

```json
{
  "language": "de",
  "strings": {
    "<sha256-of-english>": { "en": "...", "corrected": "...", "reviewer": "..." }
  },
  "templates": { "rule_text_tmpl": "..." }
}
```

Drop that in `data/authored/`, rerun with `--compose-only`, and every affected item is
regenerated. 
