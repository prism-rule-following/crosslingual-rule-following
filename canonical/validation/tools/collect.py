#!/usr/bin/env python3
"""Turn reviewed work back into a dataset.

Every language at once:
    python3 tools/collect.py

Just one:
    python3 tools/collect.py --lang ig

From a file you exported yourself (Export -> JSON in Label Studio):
    python3 tools/collect.py --lang ig --export out/export_ig.json

Writes data/reviewed/full_dataset_<lang>.json — the same filename as the source,
so a finished file drops straight in over the original when you are ready. The
originals in data/ are never written to. Reviewer notes go to out/notes_<lang>.json.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prism_review as P  # noqa: E402


def project_for(projects: list[dict], lang: str, lang_name: str):
    """Find the one project for a language. Returns (project, ambiguous_matches).

    Matching is by whole word, never bare substring: a two-letter code like `ig`
    appears inside ordinary words such as "Original", and matching that way once
    picked a scratch project instead of the real one. Tried most specific first,
    and an ambiguous result is reported rather than guessed at.
    """
    name, code = lang_name.casefold(), lang.casefold()
    exact, by_name, by_code = [], [], []

    for project in projects:
        title = (project.get("title") or "").casefold()
        if title == f"{name} translation review":
            exact.append(project)
        elif re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", title):
            by_name.append(project)
        elif re.search(rf"(?<![a-z]){re.escape(code)}(?![a-z])", title):
            by_code.append(project)

    for group in (exact, by_name, by_code):
        if len(group) == 1:
            return group[0], None
        if len(group) > 1:
            return None, group
    return None, None


def describe(projects: list[dict]) -> str:
    return "\n".join(
        f"        {p['id']}  {p.get('title', '')}" for p in projects) or "        (none)"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", help="one language code; default is every language found")
    ap.add_argument("--export", help="a Label Studio JSON export; otherwise it is fetched")
    ap.add_argument("--project", type=int,
                    help="project id, when the title doesn't identify the language")
    ap.add_argument("--en", default="data/full_dataset.json")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--reviewed-dir",
                    help="where finished datasets go (default: <data-dir>/reviewed)")
    ap.add_argument("--out-dir", default="out", help="where reviewer notes go")
    ap.add_argument("--suffix", default="",
                    help="added to the output filename, e.g. _reviewed (default: none)")
    ap.add_argument("--url", help="Label Studio address (or LABEL_STUDIO_URL)")
    ap.add_argument("--token", help="access token (or LABEL_STUDIO_API_KEY)")
    args = ap.parse_args()

    if args.project and not args.lang:
        raise SystemExit("--project applies to one language, so use it with --lang.")
    if args.export and not args.lang:
        raise SystemExit("--export is one language's export, so use it with --lang.")

    if args.lang:
        path = Path(args.data_dir) / f"full_dataset_{args.lang}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found.")
        languages = [(args.lang, path)]
    else:
        languages = P.find_languages(args.data_dir, args.en)
        if not languages:
            raise SystemExit(f"No full_dataset_<lang>.json files in {args.data_dir}/.")

    reviewed_dir = Path(args.reviewed_dir) if args.reviewed_dir \
        else Path(args.data_dir) / "reviewed"

    ls = None
    projects: list[dict] = []
    if not args.export:
        ls = P.LabelStudio(args.url, args.token)
        projects = ls.projects()
        if not projects:
            raise SystemExit("Label Studio has no projects to collect from.")

    from make_tasks import language_name  # same naming in both scripts

    written = 0
    for lang, path in languages:
        name = language_name(lang, None)
        print(f"\n{lang} ({name})")

        if args.export:
            export = P.load_json_list(args.export)
        elif args.project:
            print(f"      project {args.project}")
            export = ls.request(
                "GET", f"/api/projects/{args.project}/export"
                       f"?exportType=JSON&download_all_tasks=false") or []
        else:
            project, ambiguous = project_for(projects, lang, name)
            if ambiguous:
                print(f"      more than one project could be {name}:")
                print(describe(ambiguous))
                print("      pick one with --project <id> — skipped")
                continue
            if project is None:
                print(f"      no Label Studio project is named after {name}. "
                      f"Projects available:")
                print(describe(projects))
                print("      use --project <id> if one of these is it — skipped")
                continue
            print(f"      project {project['id']}: {project['title']}")
            export = ls.request(
                "GET", f"/api/projects/{project['id']}/export"
                       f"?exportType=JSON&download_all_tasks=false") or []

        original = P.load_dataset(path)
        out, stats = P.apply_review(export, original)

        if not stats["reviewed"]:
            print("      nobody has submitted a record yet — nothing written")
            continue

        out.setdefault("metadata", {})["review"] = {
            "records_reviewed": stats["reviewed"],
            "fields_edited": stats["changed"],
        }
        out_path = reviewed_dir / f"full_dataset_{lang}{args.suffix}.json"
        # The reviewed file has the same name as its source, so a mis-set
        # --reviewed-dir would quietly overwrite the original. Never do that.
        if out_path.resolve() == path.resolve():
            raise SystemExit(
                f"That would overwrite the source file {path}.\n"
                "Choose a different --reviewed-dir, or add --suffix _reviewed."
            )
        P.save_json(out_path, out, indent=2)
        written += 1

        print(f"      {stats['reviewed']} records reviewed, "
              f"{stats['changed']} fields edited")
        print(f"      wrote {out_path}")
        if stats["unmatched"]:
            print(f"      note: {stats['unmatched']} reviewed records had no match in "
                  f"{path.name} and were skipped")
        if stats["notes"]:
            notes_path = Path(args.out_dir) / f"notes_{lang}.json"
            P.save_json(notes_path, stats["notes"], indent=2)
            print(f"      {len(stats['notes'])} notes from reviewers -> {notes_path}")

    if not written:
        print("\nNothing was written.")
    else:
        print(f"\n{written} language(s) in {reviewed_dir}/ — the originals in "
              f"{args.data_dir}/ are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
