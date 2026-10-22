"""Durability, budget and resume tests, with no network access."""

import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import full_run
from .core import content_hash
from .data import file_hash, write_jsonl, write_json
from .test_validation import row, openai_reply


def cfg():
    return {"scale": "binary", "max_output_tokens": 512, "min_scoring_mass": .9,
            "max_attempts_per_job": 3, "max_total_attempts": 20,
            "judges": [{"id": "mock", "provider": "azure_v1", "model": "test-judge", "env_prefix": "TEST",
                        "top_k": 5, "workers": 2, "rpm": 600, "allowed_resolved_models": ["test-judge"]}]}


def init(tmp_path, count=3):
    src = tmp_path / "source"
    src.mkdir()
    rows = [{**row(i), "confidence_judge_eligible": True, "source_revision": "test-snapshot",
             "content_sha256": content_hash(row(i))} for i in range(count)]
    write_jsonl(src / "responses.jsonl", rows)
    write_json(src / "manifest.json", {"revision": "test-snapshot", "confidence_eligible_n": count,
                "responses_sha256": file_hash(src / "responses.jsonl")})
    out = tmp_path / "run"
    full_run.initialize(src / "responses.jsonl", cfg(), out)
    return out


def job(db):
    r = db.execute("SELECT j.id,i.payload,i.input_hash FROM jobs j JOIN inputs i ON i.id=j.input_id WHERE j.state='pending' ORDER BY j.id LIMIT 1").fetchone()
    db.execute("UPDATE jobs SET state='running' WHERE id=?", (r["id"],))
    db.commit()
    return dict(r)


class Fake:
    def request(self, *_):
        return openai_reply()


class NoWait:
    def take(self):
        pass
    def cooldown(self, _):
        pass


def process(out, client=None):
    db = full_run.connect(out / "checkpoint.sqlite3")
    manifest = full_run.get_meta(db, "manifest")
    claimed = job(db)
    db.close()
    stop = threading.Event()
    full_run.process_job(out / "checkpoint.sqlite3", claimed, cfg()["judges"][0], client or Fake(), NoWait(), manifest, stop)
    return stop


def test_every_success_is_durable_and_unique(tmp_path):
    out = init(tmp_path)
    process(out)
    db = full_run.connect(out / "checkpoint.sqlite3")
    record = db.execute("SELECT * FROM jobs WHERE state='done'").fetchone()
    assert record["status"] == "ok"
    assert full_run.unpack(record["result"])["raw_provider_response"]["model"] == "test-judge"
    assert db.execute("SELECT outcome FROM attempts").fetchone()[0] == "response_saved"
    assert full_run.get_meta(db, "attempts_reserved") == 1
    assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO jobs(input_id,judge_id) VALUES (1,'mock')")
    db.close()


def test_429_is_bounded_retry_and_recorded(tmp_path):
    class Limited:
        def request(self, *_):
            exc = RuntimeError("rate limit")
            exc.status_code = 429
            raise exc
    out = init(tmp_path)
    process(out, Limited())
    db = full_run.connect(out / "checkpoint.sqlite3")
    record = db.execute("SELECT * FROM jobs WHERE attempts=1").fetchone()
    assert record["state"] == "retry" and record["next_try"] > 0
    assert record["status"] == "transient_http"
    assert db.execute("SELECT http_status FROM attempts").fetchone()[0] == 429
    db.close()


def test_uncertain_network_result_is_not_automatically_replayed(tmp_path):
    class Lost:
        def request(self, *_):
            raise TimeoutError("request may have reached the provider")
    out = init(tmp_path)
    process(out, Lost())
    db = full_run.connect(out / "checkpoint.sqlite3")
    record = db.execute("SELECT * FROM jobs WHERE attempts=1").fetchone()
    assert record["state"] == "uncertain"
    assert record["status"] == "uncertain_request_outcome"
    assert full_run.unpack(record["result"])["human_review_required"] is True
    db.close()


def test_auth_failure_stops_new_dispatch_without_exposing_secrets(tmp_path):
    class BadKey:
        def request(self, *_):
            exc = RuntimeError("API_KEY=never-save-this")
            exc.status_code = 401
            raise exc
    out = init(tmp_path)
    assert process(out, BadKey()).is_set()
    db = full_run.connect(out / "checkpoint.sqlite3")
    payload = full_run.unpack(db.execute("SELECT result FROM jobs WHERE result IS NOT NULL").fetchone()[0])
    assert "never-save-this" not in json.dumps(payload)
    db.close()


def test_exact_tie_kept_as_abstention(tmp_path):
    class Tie:
        def request(self, *_):
            return openai_reply(.5)
    out = init(tmp_path)
    process(out, Tie())
    db = full_run.connect(out / "checkpoint.sqlite3")
    r = db.execute("SELECT status,verdict,confidence FROM jobs WHERE state='done'").fetchone()
    assert r["status"] == "ambiguous_tie" and r["verdict"] is None
    assert r["confidence"] == pytest.approx(0)
    db.close()


def test_model_drift_saved_but_stops_dispatch(tmp_path):
    class Drift:
        def request(self, *_):
            return openai_reply(model="unexpected-version")
    out = init(tmp_path)
    assert process(out, Drift()).is_set()
    db = full_run.connect(out / "checkpoint.sqlite3")
    assert db.execute("SELECT status FROM jobs WHERE state='done'").fetchone()[0] == "unexpected_model_version"
    db.close()


def test_attempt_ceiling_is_durable(tmp_path):
    out = init(tmp_path)
    db = full_run.connect(out / "checkpoint.sqlite3")
    full_run.set_meta(db, "attempts_reserved", 20)
    db.commit()
    with pytest.raises(RuntimeError, match="budget"):
        full_run.reserve_attempt(db, 1, 20)
    assert full_run.get_meta(db, "attempts_reserved") == 20
    db.close()


def test_resume_does_not_repeat_successes_and_exports_blank_reviews(tmp_path, monkeypatch):
    out = init(tmp_path, 4)
    process(out)
    calls = []
    class Capture(Fake):
        def __init__(self, _):
            pass
        def request(self, *_):
            calls.append(1)
            return openai_reply()
    monkeypatch.setattr(full_run, "Provider", Capture)
    monkeypatch.setattr(full_run.Pace, "take", lambda _: None)
    full_run.run(out, tmp_path / "progress.md")
    assert len(calls) == 3
    db = full_run.connect(out / "checkpoint.sqlite3")
    assert full_run.get_meta(db, "phase") == "completed"
    assert db.execute("SELECT COUNT(*) FROM jobs WHERE state='done'").fetchone()[0] == 4
    assert full_run.get_meta(db, "attempts_reserved") == 4
    db.close()
    import csv, gzip
    with open(out / "human_review/review_blinded.csv") as f:
        labels = list(csv.DictReader(f))
    assert len(labels) == 4 and all(not r["human_verdict"] for r in labels)
    with gzip.open(out / "exports/mock_scores.jsonl.gz", "rt") as f:
        assert len(list(f)) == 4
    full_run.run(out, tmp_path / "progress.md")
    assert len(calls) == 3


def test_crash_inflight_is_uncertain_not_reissued(tmp_path, monkeypatch):
    out = init(tmp_path, 2)
    db = full_run.connect(out / "checkpoint.sqlite3")
    first = job(db)
    full_run.reserve_attempt(db, first["id"], 20)
    db.close()
    calls = []
    class Capture(Fake):
        def __init__(self, _):
            pass
        def request(self, *_):
            calls.append(1)
            return openai_reply()
    monkeypatch.setattr(full_run, "Provider", Capture)
    monkeypatch.setattr(full_run.Pace, "take", lambda _: None)
    full_run.run(out, tmp_path / "progress.md")
    assert len(calls) == 1
    db = full_run.connect(out / "checkpoint.sqlite3")
    assert db.execute("SELECT state FROM jobs WHERE id=?", (first["id"],)).fetchone()[0] == "uncertain"
    db.close()
    import gzip
    with gzip.open(out / "exports/mock_scores.jsonl.gz", "rt") as f:
        exported = [json.loads(line) for line in f]
    assert len(exported) == 2
    assert any(r["scoring"]["status"] == "crash_interrupted_request" for r in exported)
