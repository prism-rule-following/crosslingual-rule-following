# Getting the dataset into Label Studio

Run everything from the project folder:

```bash
cd validation
```

---

## 1. Check the conversion works, before involving Label Studio

```bash
python tools/make_tasks.py              # every language
python tools/make_tasks.py --lang ig    # just one
```

This needs nothing running. It reads `data/full_dataset.json` and every
`data/full_dataset_<lang>.json` beside it, and writes one task file per language.

Expect:

```
english : data/full_dataset.json  (2250 records)

ig (Igbo)
      2250 records, 9 editable fields each
      wrote out/tasks_ig.json  (10.1 MB)

yo (Yoruba)
      2250 records, 9 editable fields each
      wrote out/tasks_yo.json  (10.9 MB)
```

If you only want the files — to import by hand, or to check them — you can stop
here.

---

## 2. Start Label Studio

```bash
LABEL_STUDIO_BASE_DATA_DIR=$PWD/.labelstudio \
  .venv/bin/label-studio start --port 8080 --no-browser
```

Leave it running and open a **second terminal** for everything below.

First run only: open http://localhost:8080 and create an account. Locally the
email and password are stored on your machine and can be anything.

---

## 3. Get an access token

In Label Studio: your initials (top right) → **Account & Settings** →
**Personal Access Token** → Copy.

---

## 4. Put the settings in `.env`

Create an `.env` and fill in the token:

```
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=paste-your-token-here
PRISM_ADMIN_KEY=
```

Leave `PRISM_ADMIN_KEY` empty locally. It is only for the hosted setup, where
Label Studio sits behind the gatekeeper.

---

## 5. Load the languages

```bash
python tools/make_tasks.py --push
```

For each language this creates a project, applies the review screen, and loads
all 2,250 records:

```
ig (Igbo)
      2250 records, 9 editable fields each
      wrote out/tasks_ig.json  (10.1 MB)
      loaded 2250/2250 records
      project 1: http://localhost:8080/projects/1/data
```

---

## 6. Look at a record before anyone else does

Open the project link and click **Label All Tasks**.

Check three things:

1. English on the left, the translation on the right, one row per field.
2. **The right-hand boxes already contain the translation.** They should never
   be empty — reviewers are correcting text, not typing it from scratch.
3. The grey box at the top shows the record's id, category and checker, and
   cannot be edited.

Edit a row, press **Submit**, and the next record loads.

---

## 7. Collect the reviewed work

```bash
python tools/collect.py            # every language
python tools/collect.py --lang ig  # just one
```

Finished datasets go in `data/reviewed/`, keeping the same filename as the
source so a finished file drops straight in over the original when you are
ready. The originals in `data/` are never written to.

```
ig (Igbo)
      project 5: Igbo translation review
      2 records reviewed, 2 fields edited
      wrote data/reviewed/full_dataset_ig.json
      1 notes from reviewers -> out/notes_ig.json
```

Reviewer notes go to `out/notes_<lang>.json`.

Rows nobody edited keep their original text, and the fields reviewers never see
(`checker`, `category`, and the rest) are carried across untouched.

Run it as often as you like; each run rebuilds from scratch off the current
state of the project.

---

## Adding another language

Drop `full_dataset_ha.json` into `data/` and re-run step 5. The scripts find
languages by looking at the folder, so there is nothing to configure. Languages
already loaded are skipped, so only the new one is created.

## Reloading a language

`make_tasks.py --push` will not create a second project with the same name — two
projects called "Igbo translation review" would leave `collect.py` unable to
tell which one holds the real work. To rebuild one from scratch:

```bash
python tools/make_tasks.py --lang ig --push --replace
```

That deletes the existing project **and any reviewing done in it**, then loads
the records again. Collect first if there is work you want to keep.

---

## Troubleshooting

**`No API token.`** — `.env` is missing or `LABEL_STUDIO_API_KEY` is blank. The
message tells you which file it looked at.

**`HTTP 401`** — the token is wrong or was revoked. Copy a fresh one (step 3).
Label Studio 1.20+ issues JWT personal access tokens; older ones issue a short
key. The scripts accept either, so just paste whatever the page shows you.

**`was refused as an admin action`** — only happens with the gatekeeper in
front. Set `PRISM_ADMIN_KEY` in `.env` to the same value as `ADMIN_KEY` on the
server.

**`Cannot reach http://localhost:8080`** — Label Studio isn't running, or it is
on a different port. Check the terminal from step 2.

**The edit boxes are empty.** Open the project's **Settings → Annotation** and
turn on *Show predictions to annotators*. `make_tasks.py` sets this when it
creates the project, so this should not happen — tell whoever set this up if it
does.

**Start completely over.** Stop Label Studio, then `rm -rf .labelstudio out` and
go back to step 2. Your files in `data/` are never written to by anything except
`collect.py`, which only ever creates new `_reviewed` files.
