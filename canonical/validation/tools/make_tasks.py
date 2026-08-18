#!/usr/bin/env python3
"""Turn translated datasets into Label Studio task files.

Every language at once (the default):
    python3 tools/make_tasks.py

Just one:
    python3 tools/make_tasks.py --lang ig

Straight into Label Studio, creating the project and loading the records so
there is nothing to upload by hand:

    export LABEL_STUDIO_URL=https://prism-review-xxxxx.run.app
    export LABEL_STUDIO_API_KEY=...      # Account & Settings -> Personal Access Token
    export PRISM_ADMIN_KEY=...           # only if the gatekeeper is in front
    python3 tools/make_tasks.py --push

After changing tools/labeling_config.xml, push just the new screen to the projects
that already exist, keeping every task and every annotation already submitted:
    python3 tools/make_tasks.py --update-config

Writes out/tasks_<lang>.json for each language it processes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prism_review as P  # noqa: E402

BATCH = 250


def language_name(code: str, override: str | None) -> str:
    if override:
        return override
    known = {"ig": "Igbo", "yo": "Yoruba", "de": "German", "sw": "Swahili",
             "hi": "Hindi", "it": "Italian", "ko": "Korean", "ru": "Russian",
             "am": "Amharic", "ta": "Tamil", "tr": "Turkish", "ur": "Urdu"}
    return known.get(code, code)


def push(ls, lang_name: str, tasks: list, title: str | None,
         existing: list[dict], replace: bool) -> int | None:
    """Create the project, apply the review screen, load every record.
    cannot make a second project with the same name. 
    """
    title = title or f"{lang_name} translation review"
    clashes = [p for p in existing if (p.get("title") or "") == title]

    if clashes and not replace:
        ids = ", ".join(str(p["id"]) for p in clashes)
        print(f"      already in Label Studio as project {ids} — not creating another.")
        print(f"      re-run with --replace to delete and rebuild it, or use --title "
              f"to make a separate one.")
        return None

    for project in clashes:
        print(f"      deleting existing project {project['id']}")
        ls.request("DELETE", f"/api/projects/{project['id']}/")

    project = ls.request("POST", "/api/projects/", {
        "title": title,
        "description": f"Check each record's {lang_name} translation against the "
                       f"English and fix what is wrong.",
        "label_config": P.LAYOUT.read_text(encoding="utf-8"),
        "maximum_annotations": 1,
        "sampling": "Sequential sampling",
        "show_collab_predictions": True,
        "show_instruction": True,
        "expert_instruction": P.REVIEWER_INSTRUCTIONS,
    })
    pid = project["id"]

    done = 0
    for i in range(0, len(tasks), BATCH):
        batch = [P.with_prefill(t) for t in tasks[i:i + BATCH]]
        ls.request("POST", f"/api/projects/{pid}/import?return_task_ids=false", batch)
        done += len(batch)
        print(f"      loading {done}/{len(tasks)}", end="\r", flush=True)
    print(f"      loaded {done}/{len(tasks)} records ")
    return pid


def update_config(ls, lang_name: str, title: str | None, existing: list[dict]) -> int | None:
    """Put the current review screen on a project that already exists.

    Only `label_config` is sent, so tasks and annotations are left alone. The
    alternative is `--replace`, which deletes the project and rebuilds it --
    that also throws away everything reviewers have submitted so far, which is
    far too high a price for a change to the layout or its CSS.
    """
    title = title or f"{lang_name} translation review"
    matches = [p for p in existing if (p.get("title") or "") == title]

    if not matches:
        print(f"      no project called {title!r} in Label Studio — nothing to update.")
        print(f"      re-run with --push to create it.")
        return None
    if len(matches) > 1:
        ids = ", ".join(str(p["id"]) for p in matches)
        print(f"      {len(matches)} projects share that name ({ids}) — refusing to guess "
              f"which one. Use --title to name the right one.")
        return None

    pid = matches[0]["id"]
    ls.request("PATCH", f"/api/projects/{pid}/",
               {"label_config": P.LAYOUT.read_text(encoding="utf-8")})
    print(f"      project {pid}: review screen updated, tasks and annotations untouched")
    return pid


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--en", default="data/full_dataset.json",
                    help="English source (default: %(default)s)")
    ap.add_argument("--lang", help="one language code; default is every language found")
    ap.add_argument("--lang-name", help="display name, only meaningful with --lang")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--push", action="store_true",
                    help="also create the Label Studio project and load the records")
    ap.add_argument("--replace", action="store_true",
                    help="delete and rebuild a project that already has the same name")
    ap.add_argument("--update-config", action="store_true",
                    help="only put the current labeling_config.xml on projects that "
                         "already exist; leaves tasks and annotations alone")
    ap.add_argument("--title", help="project title, only meaningful with --lang --push")
    ap.add_argument("--url", help="Label Studio address (or LABEL_STUDIO_URL)")
    ap.add_argument("--token", help="access token (or LABEL_STUDIO_API_KEY)")
    args = ap.parse_args()

    if args.update_config and args.replace:
        raise SystemExit(
            "--update-config and --replace contradict each other: one keeps the project "
            "and its annotations, the other deletes them. Pick one.")

    if args.lang:
        path = Path(args.data_dir) / f"full_dataset_{args.lang}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found.")
        languages = [(args.lang, path)]
    else:
        languages = P.find_languages(args.data_dir, args.en)
        if not languages:
            raise SystemExit(
                f"No full_dataset_<lang>.json files in {args.data_dir}/ besides the English one.")

    ls = P.LabelStudio(args.url, args.token) if (args.push or args.update_config) else None
    existing = ls.projects() if ls else []

    # Nothing here needs the datasets themselves, so they are never read: this
    # path only swaps one field on a project that already exists.
    if args.update_config:
        print(f"screen  : {P.LAYOUT}")
        for lang, _ in languages:
            name = language_name(lang, args.lang_name if args.lang else None)
            print(f"\n{lang} ({name})")
            update_config(ls, name, args.title if args.lang else None, existing)
        return 0

    en_data = P.load_dataset(args.en)
    en_pairs = en_data["pairs"]

    print(f"english : {args.en}  ({len(en_pairs)} records)")
    for lang, path in languages:
        name = language_name(lang, args.lang_name if args.lang else None)
        tr_data = P.load_dataset(path)
        tasks, missing = P.build_tasks(en_pairs, tr_data["pairs"], name)

        out_path = Path(args.out_dir) / f"tasks_{lang}.json"
        P.save_json(out_path, [P.with_prefill(t) for t in tasks])

        avg = round(sum(len(t["data"]["fields"]) for t in tasks) / len(tasks)) if tasks else 0
        size = out_path.stat().st_size / 1e6
        print(f"\n{lang} ({name})")
        print(f"      {len(tasks)} records, {avg} editable fields each")
        print(f"      wrote {out_path}  ({size:.1f} MB)")
        if missing:
            print(f"      note: {len(missing)} records had no English record with the "
                  f"same id and were left out (first: {missing[0]})")

        if ls:
            pid = push(ls, name, tasks, args.title if args.lang else None,
                       existing, args.replace)
            if pid is not None:
                print(f"      project {pid}: {ls.url}/projects/{pid}/data")

    if not ls:
        print("\nImport each file in Label Studio: Create Project -> Labeling Setup ->")
        print("Custom template -> paste tools/labeling_config.xml -> Data Import -> drop")
        print("the file -> Save. Or re-run with --push to skip all of that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
