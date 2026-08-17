# Translation review

Native speakers check machine-translated datasets in Label Studio, English beside the translation.

## Two commands

```bash
# every language in data/, straight into Label Studio
python3 tools/make_tasks.py --push

# pull the finished work back out
python3 tools/collect.py
```

Both take `--lang ig` to do a single language. Without `--push`, `make_tasks.py`
just writes `out/tasks_<lang>.json` for you to import by hand.

`collect.py` writes to `data/reviewed/`, using the same filename as the source so
a finished file drops straight in over the original. The originals are never
written to.

## Files

```
tools/make_tasks.py          datasets -> task files, optionally straight into Label Studio
tools/collect.py             reviewed work -> full_dataset_<lang>_reviewed.json
tools/prism_review.py        shared conversion + a small Label Studio client
tools/labeling_config.xml    the reviewer screen
deploy/                      Dockerfile + nginx gatekeeper, what Cloud Run builds
docs/VALIDATOR_GUIDE.md      for reviewers
docs/SETUP_HOSTING.md        putting it online (Google Cloud Run)
data/                        the source datasets
data/reviewed/               finished datasets, same filenames
out/                         generated task files and reviewer notes
```

Running it locally: `docs/GETTING_STARTED.md`. 

Hosting it: `docs/SETUP_HOSTING.md`.
