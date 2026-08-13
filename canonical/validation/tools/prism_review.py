"""Shared pieces for the review scripts.
Two things live here: the record -> Label Studio task conversion, and a small
Label Studio API client. Both `make_tasks.py` and `collect.py` use them.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from html import escape
from pathlib import Path

#: Carried through untouched and shown to reviewers as read-only context.
READONLY = [
    "id", "category", "topic", "grammar_type", "language", "pair_type",
    "check_native", "checker", "target_lang", "target_count",
]

#: Friendly names for the editable rows. Anything not listed keeps its key.
LABELS = {
    "context": "Context",
    "rule_text": "Rule text",
    "non_rule_text": "Non-rule text",
    "system_rule": "System rule",
    "system_non_rule": "System non-rule",
    "user_query": "User query",
    "rule_clause": "Rule clause",
    "correct_answer": "Correct answer",
    "correct_keywords": "Correct keywords (one per line)",
    "target_word": "Target word",
    "opener": "Opener",
    "anchor_token": "Anchor token",
    "truth": "Truth",
    "expected_answer": "Expected answer",
    "ack_token": "Acknowledgement token",
    "expected_full": "Expected full reply",
    "banned_word": "Banned word",
}

REVIEWER_INSTRUCTIONS = (
    "<p>Read the English on the left and the translation on the right, row by row.</p>"
    "<ul><li>If a translation is fine, leave it alone.</li>"
    "<li>If it is wrong, fix the text in the box.</li>"
    "<li>Then press Submit and the next record loads.</li></ul>"
    "<p>Keep quoted words inside their quotes, keep HTML tags such as &lt;b&gt; exactly "
    "where they are, and keep one keyword per line. If you are unsure, leave the text "
    "as it is and write a note at the bottom.</p>"
)

LAYOUT = Path(__file__).with_name("labeling_config.xml")
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


def load_env() -> Path | None:
    """Read KEY=VALUE lines from a .env file into the environment.

    Looked for in the current directory first, then beside the scripts. Real
    environment variables win, so `LABEL_STUDIO_URL=... python3 tools/…` still
    overrides the file for a one-off.

    Deliberately tiny: no dependency, and no inline `# comment` stripping, so a
    value containing a `#` survives intact.
    """
    for directory in (Path.cwd(), PROJECT_ROOT):
        path = directory / ".env"
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)
        return path
    return None


ENV_FILE = load_env()


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def load_dataset(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("pairs"), list):
        raise ValueError(f"{path}: expected a top-level 'pairs' list")
    return data


def load_json_list(path: str | Path) -> list:
    """Read a Label Studio export, which is a list of tasks."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(
            f"{path} is not a Label Studio export — expected a list of tasks. "
            "Export with Export -> JSON."
        )
    return data


def save_json(path: str | Path, data, indent=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)
        if indent:
            fh.write("\n")


def find_languages(data_dir: str | Path, english: str | Path) -> list[tuple[str, Path]]:
    """Every full_dataset_<lang>.json next to the English file, as (lang, path)."""
    english = Path(english).resolve()
    found = []
    for path in sorted(Path(data_dir).glob("full_dataset_*.json")):
        if path.resolve() == english or path.stem.endswith("_reviewed"):
            continue
        found.append((path.stem[len("full_dataset_"):], path))
    return found


# --------------------------------------------------------------------------
# Records -> tasks
# --------------------------------------------------------------------------


def esc(value) -> str:
    """Escape &, <, > and " -- but not apostrophes.

    Checkers are full of single quotes (`re.search(r'<(b|strong)...')`) and
    leaving them alone keeps them readable on screen. Attributes here are always
    double-quoted, so this is still safe.
    """
    if isinstance(value, bool):
        value = "true" if value else "false"    # as it reads in the file, not "True"
    elif value is None:
        value = "null"
    return escape(str(value), quote=False).replace('"', "&quot;")


def meta_html(en: dict, tr: dict) -> str:
    rows = []
    for key in READONLY:
        if key not in tr and key not in en:
            continue
        value = tr[key] if key in tr else en[key]
        cell = f"<code>{esc(value)}</code>" if key == "checker" else esc(value)
        rows.append(f'<tr><td class="k">{esc(key)}</td><td>{cell}</td></tr>')
    return f'<table>{"".join(rows)}</table>'


def build_tasks(en_pairs: list, tr_pairs: list, lang_name: str):
    """One task per record: every editable field as an English/translation row.

    Records are matched by id, not position, so a reordered file still works.
    Returns (tasks, ids_with_no_english_record).

    Every key put in `data` here is stored in Label Studio and shown as a column,
    so only keys something actually reads belong in it: `fields` and `position`
    are rendered by the review screen, `lang_name` titles the right-hand column,
    and `record_id` is what collect.py matches work back on.
    """
    by_id = {r.get("id"): r for r in en_pairs}
    tasks, missing = [], []

    for i, tr in enumerate(tr_pairs):
        en = by_id.get(tr.get("id"))
        if en is None:
            missing.append(tr.get("id"))
            continue

        fields = []
        for key in tr:                       # the record's own key order
            if key in READONLY:
                continue
            tv, ev = tr[key], en.get(key, "")
            is_list = isinstance(tv, list)
            fields.append({
                "key": key,
                "label": LABELS.get(key, key.replace("_", " ")),
                "list": is_list,
                "en": "\n".join(map(str, ev)) if isinstance(ev, list) else str(ev),
                "tr": "\n".join(map(str, tv)) if is_list else str(tv),
            })

        tasks.append({"data": {
            "record_id": tr.get("id"),
            "lang_name": lang_name,
            "position": f"Record {i + 1} of {len(tr_pairs)}  ·  {tr.get('id')}",
            "meta_html": meta_html(en, tr),
            "fields": fields,
        }})

    return tasks, missing


def with_prefill(task: dict) -> dict:
    """Ship each row's current translation as a prediction.

    This is what puts the existing text in the reviewer's edit box; Label Studio
    copies a prediction into the annotation when the record is opened.
    """
    return {
        "data": task["data"],
        "predictions": [{
            "model_version": "machine-translation",
            "result": [
                {
                    "from_name": f"tr_{i}",
                    "to_name": f"en_{i}",
                    "type": "textarea",
                    "value": {"text": [f["tr"]]},
                }
                for i, f in enumerate(task["data"]["fields"])
            ],
        }],
    }


# --------------------------------------------------------------------------
# Tasks -> dataset
# --------------------------------------------------------------------------


def apply_review(export_tasks: list, original: dict):
    """Fold reviewed records back into a copy of the original dataset.

    A row the reviewer never touched submits nothing, so it keeps its original
    value. Nothing is recomputed or corrected -- whatever was typed is stored.
    """
    out = json.loads(json.dumps(original))
    by_id = {r.get("id"): r for r in out["pairs"]}

    reviewed = changed = unmatched = 0
    notes = []

    for task in export_tasks:
        data = task.get("data") or {}
        record = by_id.get(data.get("record_id"))
        if record is None:
            unmatched += 1
            continue

        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        reviewed += 1

        fields = data.get("fields") or []
        for result in annotations[-1].get("result", []):     # last one wins
            text = (result.get("value") or {}).get("text") or []
            if not text:
                continue
            text = text[0]

            if result.get("from_name") == "note":
                if str(text).strip():
                    notes.append({"id": data.get("record_id"), "note": str(text).strip()})
                continue

            name = str(result.get("from_name", ""))
            if not name.startswith("tr_") or not name[3:].isdigit():
                continue
            index = int(name[3:])
            if index >= len(fields):
                continue
            field = fields[index]

            if field.get("list"):
                value = [s.strip() for s in str(text).split("\n") if s.strip()]
            else:
                value = str(text)

            if value != record.get(field["key"]):
                changed += 1
            record[field["key"]] = value

    return out, {"reviewed": reviewed, "changed": changed,
                 "unmatched": unmatched, "notes": notes}


# --------------------------------------------------------------------------
# Label Studio
# --------------------------------------------------------------------------


class LabelStudio:
    """Minimal API client, so no SDK version has to be pinned."""

    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")).rstrip("/")
        self.token = (token or os.environ.get("LABEL_STUDIO_API_KEY", "")).strip()
        self.admin_key = os.environ.get("PRISM_ADMIN_KEY", "").strip()
        if not self.token:
            where = f"in {ENV_FILE}" if ENV_FILE else "in a .env file next to this project"
            raise SystemExit(
                f"No API token. Set LABEL_STUDIO_API_KEY {where}\n"
                "(Label Studio -> your initials -> Account & Settings -> Personal "
                "Access Token), or pass --token."
            )
        self.auth = self._negotiate()

    def _negotiate(self) -> str:
        """Personal access tokens are JWT refresh tokens and must be exchanged.

        Older instances issue a plain key instead; the shape tells them apart.
        """
        if self.token.count(".") != 2:
            return f"Token {self.token}"
        payload = json.dumps({"refresh": self.token}).encode()
        req = urllib.request.Request(
            f"{self.url}/api/token/refresh", data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return "Bearer " + json.loads(resp.read().decode())["access"]
        except urllib.error.HTTPError:
            raise SystemExit(
                "That access token was rejected. Copy a fresh one from Label Studio "
                "-> Account & Settings -> Personal Access Token."
            ) from None
        except urllib.error.URLError as e:
            raise SystemExit(f"Cannot reach {self.url} ({e.reason}).") from None

    def request(self, method: str, path: str, payload=None, _retried=False):
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Authorization": self.auth}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.admin_key:
            headers["Cookie"] = f"prism_admin={self.admin_key}"

        req = urllib.request.Request(f"{self.url}{path}", data=body, method=method,
                                     headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                text = resp.read().decode()
                return json.loads(text) if text else None
        except urllib.error.HTTPError as e:
            if e.code == 401:
                detail = e.read().decode("utf-8", "replace")
                # Label Studio 1.20+ turns off the old-style tokens by default,
                # which is what `label-studio start --user-token` hands you.
                if "legacy token" in detail.lower():
                    raise SystemExit(
                        "Label Studio has disabled old-style API tokens.\n"
                        "Use a Personal Access Token instead: open "
                        f"{self.url} -> your initials (top right) -> Account & Settings\n"
                        "-> Personal Access Token -> Copy, and put that in "
                        "LABEL_STUDIO_API_KEY."
                    ) from None
                # Access tokens last five minutes; a long import outlives one.
                if not _retried:
                    self.auth = self._negotiate()
                    return self.request(method, path, payload, True)
            if e.code == 403:
                raise SystemExit(
                    f"{method} {path} was refused as an admin action.\n"
                    "If Label Studio is behind the gatekeeper, set PRISM_ADMIN_KEY to "
                    "the same value as ADMIN_KEY on the server."
                ) from None
            raise SystemExit(
                f"{method} {path} -> HTTP {e.code}\n{e.read().decode('utf-8', 'replace')[:600]}"
            ) from None
        except urllib.error.URLError as e:
            raise SystemExit(f"Cannot reach {self.url} ({e.reason}).") from None

    def projects(self) -> list[dict]:
        out, page = [], 1
        while True:
            try:
                data = self.request("GET", f"/api/projects?page={page}&page_size=100")
            except SystemExit:
                break                      # past the last page Label Studio 404s
            rows = data.get("results", data) if isinstance(data, dict) else data
            out.extend(rows or [])
            if not isinstance(data, dict) or not data.get("next"):
                break
            page += 1
        return out
