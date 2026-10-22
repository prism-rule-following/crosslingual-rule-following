"""Durable, bounded full-dataset scoring with explicit abstentions and audit.

SQLite WAL + synchronous FULL checkpoints each attempt/result. A crash never
silently replays a request whose provider outcome is unknown.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import gzip
import heapq
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading
import time
import zlib

from .core import PROTOCOL_VERSION, build_prompt, content_hash, digest, response_key, bucket, normalize_row
from .data import file_hash, write_json
from .providers import Provider, normalize_openai, evaluate_reply
from .review import REVIEW_FIELDS, rubric_for

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE inputs (id INTEGER PRIMARY KEY, key_json TEXT UNIQUE NOT NULL,
 input_hash TEXT NOT NULL, payload BLOB NOT NULL);
CREATE TABLE jobs (id INTEGER PRIMARY KEY, input_id INTEGER NOT NULL, judge_id TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
 next_try REAL NOT NULL DEFAULT 0, result BLOB, status TEXT, verdict TEXT,
 confidence REAL, adherence REAL, resolved_model TEXT, prompt_tokens INTEGER DEFAULT 0,
 completion_tokens INTEGER DEFAULT 0, updated TEXT,
 UNIQUE(input_id,judge_id), FOREIGN KEY(input_id) REFERENCES inputs(id));
CREATE INDEX jobs_queue ON jobs(judge_id,state,next_try,id);
CREATE INDEX jobs_dispatch ON jobs(judge_id,state,id);
CREATE TABLE attempts (id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL,
 attempt_n INTEGER NOT NULL, started TEXT NOT NULL, finished TEXT,
 outcome TEXT NOT NULL, http_status INTEGER, error_kind TEXT,
 UNIQUE(job_id,attempt_n));
"""


def utc():
    return datetime.now(timezone.utc).isoformat()


def pack(obj):
    return zlib.compress(json.dumps(obj, ensure_ascii=False, allow_nan=False).encode(), level=6)


def unpack(blob):
    return json.loads(zlib.decompress(blob))


def connect(path):
    db = sqlite3.connect(path, timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def atomic_json(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def code_hashes():
    base = Path(__file__).parent
    return {name: file_hash(base / name) for name in
            ("core.py", "providers.py", "data.py", "review.py", "full_run.py")}


def get_meta(db, key):
    return json.loads(db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])


def set_meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(value, ensure_ascii=False)))


def initialize(source, config, output):
    """Validate all inputs/config before any request; exclusive new run."""
    source, output = Path(source), Path(output)
    source_manifest = json.loads((source.parent / "manifest.json").read_text())
    if file_hash(source) != source_manifest["responses_sha256"]:
        raise ValueError("source JSONL no longer matches audited source hash")
    for j in config["judges"]:
        if j["provider"] != "azure_v1" or not j.get("model") or not j.get("allowed_resolved_models"):
            raise ValueError("full run requires explicit verified Azure models and returned versions")
        if not 1 <= j["workers"] <= 12 or not 1 <= j["rpm"] <= 600:
            raise ValueError("invalid worker/rate limit")
    output.mkdir(parents=True, exist_ok=False)
    db = connect(output / "checkpoint.sqlite3")
    db.executescript(SCHEMA)
    count, excluded = 0, Counter()
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            row = normalize_row(json.loads(line))
            if not row.get("confidence_judge_eligible"):
                excluded[row.get("category", "unknown")] += 1
                continue
            if row.get("source_revision") != source_manifest["revision"]:
                raise ValueError("mixed input snapshot revisions")
            key = response_key(row)
            build_prompt(row, "binary")
            row_hash = content_hash(row)
            if row_hash != row.get("content_sha256"):
                raise ValueError("source content hash mismatch")
            cursor = db.execute("INSERT INTO inputs(key_json,input_hash,payload) VALUES (?,?,?)",
                                (json.dumps(key, ensure_ascii=False), row_hash, pack(row)))
            for judge in config["judges"]:
                db.execute("INSERT INTO jobs(input_id,judge_id) VALUES (?,?)", (cursor.lastrowid, judge["id"]))
            count += 1
            if count % 1000 == 0:
                db.commit()
                print(f"prepared {count} inputs", flush=True)
    if count != source_manifest["confidence_eligible_n"]:
        raise ValueError("eligible source count differs from audited manifest")
    manifest = {"config": config, "protocol": PROTOCOL_VERSION, "source_manifest": source_manifest,
                "implementation": code_hashes(), "input_count": count,
                "planned_judgments": count * len(config["judges"]), "excluded_categories": dict(excluded),
                "created_utc": utc(), "ties": "abstain_and_route_to_human", "score_errors": "retain_unscored",
                "review_plan": {"seed": 42, "per_judge_metric_band": 50,
                                "metrics": ["confidence_score", "adherence_score"],
                                "primary": "confidence_score", "human_labels": "blank"}}
    manifest["fingerprint"] = digest(manifest)
    set_meta(db, "manifest", manifest)
    set_meta(db, "attempts_reserved", 0)
    set_meta(db, "phase", "ready")
    db.commit()
    write_json(output / "manifest.json", manifest)
    db.close()
    return manifest


class Pace:
    def __init__(self, rpm):
        self.interval = 60 / rpm
        self.next = 0.0
        self.lock = threading.Lock()

    def take(self):
        with self.lock:
            start = max(time.monotonic(), self.next)
            self.next = start + self.interval
        time.sleep(max(0, start - time.monotonic()))

    def cooldown(self, delay):
        with self.lock:
            self.next = max(self.next, time.monotonic() + delay)


def reserve_attempt(db, job_id, total_cap):
    db.execute("BEGIN IMMEDIATE")
    count = get_meta(db, "attempts_reserved")
    if count >= total_cap:
        db.rollback()
        raise RuntimeError("attempt budget reached")
    set_meta(db, "attempts_reserved", count + 1)
    n = db.execute("SELECT attempts FROM jobs WHERE id=?", (job_id,)).fetchone()[0] + 1
    db.execute("UPDATE jobs SET attempts=?,updated=? WHERE id=?", (n, utc(), job_id))
    cur = db.execute("INSERT INTO attempts(job_id,attempt_n,started,outcome) VALUES (?,?,?,?)",
                     (job_id, n, utc(), "sent_or_uncertain"))
    db.commit()
    return n, cur.lastrowid


def error_info(exc):
    code = getattr(exc, "status_code", None)
    text = str(exc).lower()  # inspect only for classification; never persist secrets/headers
    if any(s in text for s in ("content_filter", "jailbreak", "contentfilter")):
        kind = "content_filter"
    elif code in (401, 403):
        kind = "authentication_or_permission"
    elif code in (400, 404, 422):
        kind = "permanent_request_or_deployment"
    elif code in (429, 500, 502, 503, 504):
        kind = "transient_http"
    else:
        kind = "uncertain_request_outcome"
    delay = 0
    headers = getattr(getattr(exc, "response", None), "headers", {})
    try:
        delay = float(headers.get("retry-after", 0))
        if not delay:
            delay = float(headers.get("retry-after-ms", 0)) / 1000
    except (ValueError, TypeError):
        pass
    return {"type": type(exc).__name__, "http_status": code, "kind": kind}, delay


def process_job(database, job, judge, client, pace, manifest, stop):
    db = connect(database)
    row = unpack(job["payload"])
    prompt = build_prompt(row, "binary")
    result = {**row, "judge_id": judge["id"], "judge_model": judge["model"], "provider": judge["provider"],
              "scale": "binary", "protocol": PROTOCOL_VERSION, "run_fingerprint": manifest["fingerprint"],
              "input_sha256": job["input_hash"], "prompt_sha256": digest(prompt), "created_utc": utc(),
              "scoring_request": {"top_k": judge["top_k"], "max_output_tokens": manifest["config"]["max_output_tokens"]}}
    try:
        pace.take()
        if stop.is_set():
            db.execute("UPDATE jobs SET state='pending' WHERE id=?", (job["id"],))
            db.commit()
            return
        n, attempt_id = reserve_attempt(db, job["id"], manifest["config"]["max_total_attempts"])
        try:
            raw = client.request(prompt, judge["top_k"], manifest["config"]["max_output_tokens"])
        except Exception as exc:
            error, retry_after = error_info(exc)
            result.update(error=error, verdict=None, scoring={"status": error["kind"]}, human_review_required=True)
            retry = error["kind"] == "transient_http" and n < manifest["config"]["max_attempts_per_job"]
            if retry:
                delay = max(retry_after, min(60, 2 ** n * 5))
                pace.cooldown(delay)
                state, next_try = "retry", time.time() + delay
            else:
                state, next_try = "uncertain" if error["kind"] == "uncertain_request_outcome" else "done", 0
            if error["kind"] in ("authentication_or_permission", "permanent_request_or_deployment"):
                stop.set()
            db.execute("UPDATE attempts SET finished=?,outcome=?,http_status=?,error_kind=? WHERE id=?",
                       (utc(), state, error["http_status"], error["kind"], attempt_id))
            db.execute("UPDATE jobs SET state=?,next_try=?,result=?,status=?,updated=? WHERE id=?",
                       (state, next_try, pack(result), error["kind"], utc(), job["id"]))
            db.commit()
            return
        result["raw_provider_response"] = raw
        try:
            reply = normalize_openai(raw)
            score = evaluate_reply(reply, "binary", manifest["config"]["min_scoring_mass"])
            result.update(resolved_model=reply["resolved_model"] or "UNREPORTED", scoring=score,
                          verdict=score["weighted_verdict"], error=None,
                          human_review_required=score["status"] != "ok")
            if result["resolved_model"] not in judge["allowed_resolved_models"]:
                result["score_before_model_gate"] = score
                result["scoring"] = {"status": "unexpected_model_version"}
                result["human_review_required"] = True
                stop.set()
        except (ValueError, KeyError, TypeError) as exc:
            result.update(scoring={"status": "invalid_provider_response"}, verdict=None,
                          error=str(exc), human_review_required=True)
        usage = raw.get("usage") or {}
        score = result["scoring"]
        db.execute("UPDATE attempts SET finished=?,outcome='response_saved' WHERE id=?", (utc(), attempt_id))
        db.execute("""UPDATE jobs SET state='done',result=?,status=?,verdict=?,confidence=?,adherence=?,
                   resolved_model=?,prompt_tokens=?,completion_tokens=?,updated=? WHERE id=?""",
                   (pack(result), score["status"], result.get("verdict"), score.get("confidence_score"),
                    score.get("adherence_score"), result.get("resolved_model"), usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0), utc(), job["id"]))
        db.commit()  # durable intermediate result before returning the worker slot
    finally:
        db.close()


def status_report(db, phase):
    grouped = [dict(r) for r in db.execute("SELECT judge_id,state,status,COUNT(*) AS n FROM jobs GROUP BY judge_id,state,status")]
    usage = [dict(r) for r in db.execute("SELECT judge_id,SUM(prompt_tokens) AS prompt_tokens,SUM(completion_tokens) AS completion_tokens FROM jobs GROUP BY judge_id")]
    return {"updated_utc": utc(), "phase": phase, "counts": grouped, "reported_usage": usage,
            "attempts_reserved": get_meta(db, "attempts_reserved"), "cost": "not returned by this Azure API; token usage recorded"}


def progress(output, db, phase, log):
    report = status_report(db, phase)
    atomic_json(output / "status.json", report)
    by_judge = defaultdict(Counter)
    for group in report["counts"]:
        by_judge[group["judge_id"]][group["state"]] += group["n"]
    line = f"{report['updated_utc']} phase={phase} " + "; ".join(f"{j}: {dict(v)}" for j, v in sorted(by_judge.items()))
    print(line, flush=True)
    with open(output / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
        f.flush()
        os.fsync(f.fileno())
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n- Full-run checkpoint: " + line + f"; attempts={report['attempts_reserved']}.\n")
        f.flush()
    return report


def run(output, log, max_new_attempts=None):
    output, log = Path(output), Path(log)
    import fcntl
    process_lock = open(output / "runner.lock", "a")
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        process_lock.close()
        raise RuntimeError("another coordinator already holds this run; refusing duplicate requests") from None
    db = connect(output / "checkpoint.sqlite3")
    manifest = get_meta(db, "manifest")
    if code_hashes() != manifest["implementation"]:
        raise ValueError("code changed since experiment freeze; reconcile explicitly before resume")
    if get_meta(db, "phase") == "completed":
        print("Already complete; no requests repeated.")
        return
    # Only requests that really began might have reached the provider. A queued
    # worker without an attempt journal can safely return to pending.
    db.execute("UPDATE jobs SET state='pending' WHERE state='running' AND id NOT IN (SELECT job_id FROM attempts WHERE outcome='sent_or_uncertain')")
    db.execute("UPDATE jobs SET state='uncertain',status='crash_interrupted_request',updated=? WHERE state='running'", (utc(),))
    set_meta(db, "phase", "running")
    db.commit()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    judges = {j["id"]: j for j in manifest["config"]["judges"]}
    clients = {j: Provider(cfg) for j, cfg in judges.items()}
    paces = {j: Pace(cfg["rpm"]) for j,cfg in judges.items()}
    executors = {j: ThreadPoolExecutor(max_workers=cfg["workers"]) for j,cfg in judges.items()}
    active = {j: set() for j in judges}
    initial_attempts = get_meta(db, "attempts_reserved")
    progress(output, db, "running", log)
    last_progress = time.monotonic()
    try:
        while True:
            for judge_id, cfg in judges.items():
                for future in list(active[judge_id]):
                    if future.done():
                        future.result()
                        active[judge_id].remove(future)
                if stop.is_set():
                    continue
                if max_new_attempts is not None and get_meta(db, "attempts_reserved") - initial_attempts >= max_new_attempts:
                    stop.set()
                    continue
                while len(active[judge_id]) < cfg["workers"]:
                    db.execute("BEGIN IMMEDIATE")
                    candidate = db.execute("SELECT id FROM jobs WHERE judge_id=? AND state='retry' AND next_try<=? ORDER BY next_try,id LIMIT 1",
                                           (judge_id, time.time())).fetchone()
                    if candidate is None:
                        candidate = db.execute("SELECT id FROM jobs INDEXED BY jobs_dispatch WHERE judge_id=? AND state='pending' ORDER BY id LIMIT 1",
                                               (judge_id,)).fetchone()
                    if candidate is None:
                        db.rollback()
                        break
                    job = db.execute("SELECT j.id,j.attempts,i.payload,i.input_hash FROM jobs j JOIN inputs i ON i.id=j.input_id WHERE j.id=?",
                                     (candidate["id"],)).fetchone()
                    db.execute("UPDATE jobs SET state='running',updated=? WHERE id=?", (utc(), job["id"]))
                    db.commit()
                    future = executors[judge_id].submit(process_job, output / "checkpoint.sqlite3", dict(job), cfg,
                                                       clients[judge_id], paces[judge_id], manifest, stop)
                    active[judge_id].add(future)
            pending_n = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('pending','retry')").fetchone()[0]
            futures = set().union(*active.values())
            if not futures and (stop.is_set() or not pending_n):
                break
            if futures:
                wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            else:
                time.sleep(1)
            if time.monotonic() - last_progress >= 60:
                if code_hashes() != manifest["implementation"]:
                    stop.set()
                if __import__("shutil").disk_usage(output).free < 5 * 1024**3:
                    stop.set()
                progress(output, db, "draining" if stop.is_set() else "running", log)
                last_progress = time.monotonic()
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)
        remaining = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('pending','retry','running')").fetchone()[0]
        phase = "attention_required" if stop.is_set() or remaining else "scoring_finished"
        set_meta(db, "phase", phase)
        db.commit()
        progress(output, db, phase, log)
        db.close()
    if phase == "scoring_finished":
        finalize(output, log)
    process_lock.close()


def stream_rows(db):
    manifest = get_meta(db, "manifest")
    judges = {j["id"]: j for j in manifest["config"]["judges"]}
    for record in db.execute("SELECT j.id,j.judge_id,j.state,j.status,j.result,i.payload,i.input_hash FROM jobs j JOIN inputs i ON i.id=j.input_id ORDER BY j.id"):
        if record["result"] is not None:
            result = unpack(record["result"])
        else:
            result = {**unpack(record["payload"]), "judge_id": record["judge_id"],
                      "judge_model": judges[record["judge_id"]]["model"], "provider": judges[record["judge_id"]]["provider"],
                      "protocol": PROTOCOL_VERSION, "scale": "binary", "run_fingerprint": manifest["fingerprint"],
                      "input_sha256": record["input_hash"], "resolved_model": "UNREPORTED", "verdict": None,
                      "scoring": {"status": record["status"] or record["state"]},
                      "human_review_required": True, "error": "No durable provider response; see attempt journal"}
        yield record["id"], result


def finalize(output, log):
    """Bounded-memory exports, counts, paired agreement and blinded samples."""
    import csv
    output = Path(output)
    db = connect(output / "checkpoint.sqlite3")
    manifest = get_meta(db, "manifest")
    if db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('pending','retry','running')").fetchone()[0]:
        raise ValueError("cannot finalize while judgments are pending or in flight")
    exports = output / "exports"
    exports.mkdir(exist_ok=True)
    counts = Counter()
    strata = defaultdict(int)
    selected = defaultdict(list)
    seed, sample_n = 42, 50
    writers = {}
    try:
        for j in manifest["config"]["judges"]:
            writers[j["id"]] = gzip.open(exports / (j["id"] + "_scores.jsonl.gz.tmp"), "wt", encoding="utf-8")
        for job_id, row in stream_rows(db):
            slim = {k:v for k,v in row.items() if k not in ("raw_provider_response", "judge_prompt")}
            slim["raw_checkpoint_job_id"] = job_id
            writers[row["judge_id"]].write(json.dumps(slim, ensure_ascii=False, allow_nan=False) + "\n")
            status = row["scoring"]["status"]
            counts[(row["judge_id"], row["language"], row.get("category"), status)] += 1
            if status != "ok":
                continue
            stream = (row["judge_id"], row["judge_model"], row.get("resolved_model", "UNREPORTED"))
            for metric in ("confidence_score", "adherence_score"):
                band = bucket(row["scoring"][metric])
                key = (*stream, metric, band)
                strata[key] += 1
                rank = int(digest([seed, list(key), list(response_key(row))]), 16)
                heap = selected[key]
                item = (-rank, job_id)
                if len(heap) < sample_n:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    finally:
        for writer in writers.values():
            writer.close()
    for j in writers:
        temporary = exports / (j + "_scores.jsonl.gz.tmp")
        os.replace(temporary, exports / (j + "_scores.jsonl.gz"))
    packet = output / "human_review"
    packet.mkdir(exist_ok=True)
    exposures, reviews = [], {}
    for key in sorted(selected):
        for _, job_id in sorted(selected[key]):
            row = unpack(db.execute("SELECT result FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
            rid = digest([response_key(row), row["input_sha256"]])[:24]
            rubric, _ = rubric_for(row)
            reviews[rid] = dict(zip(REVIEW_FIELDS, (rid, row["language"], row["rule_clause"], row["user_query"], row["response"],
                row["checker"].get("violation_event"), rubric.get("held_if"), rubric.get("violated_if"), rubric.get("ignore") or "",
                "", "", "", "", "")))
            exposures.append({"review_id": rid, "job_id": job_id, "stream": list(key[:3]), "metric": key[3],
                              "bucket": key[4], "population_n": strata[key], "sample_n": len(selected[key]),
                              "inclusion_probability": len(selected[key])/strata[key]})
    # Hash order is reproducible and does not expose scores or judge identity.
    with open(packet / "review_blinded.csv.tmp", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, REVIEW_FIELDS)
        writer.writeheader()
        for rid in sorted(reviews, key=lambda x: digest([seed, x])):
            writer.writerow(reviews[rid])
    os.replace(packet / "review_blinded.csv.tmp", packet / "review_blinded.csv")
    atomic_json(packet / "PRIVATE_sampling_manifest.json", {"seed": seed, "primary": "confidence_score",
                "method": "50 smallest seeded SHA256 ranks per judge/version/metric/band; equivalent uniform sampling under hash ranks",
                "exposures": exposures, "human_labels": "blank", "review_n": len(reviews)})
    # Also emit standard metric-specific packets accepted by analyze.py. The
    # union sheet above avoids asking humans to label a response twice.
    from .review import stream_key
    for metric in ("confidence_score", "adherence_score"):
        metric_dir = packet / metric
        metric_dir.mkdir(exist_ok=True)
        metric_ids, private, metric_strata = set(), [], []
        for key in sorted(k for k in selected if k[3] == metric):
            chosen = selected[key]
            first_job = chosen[0][1]
            first = unpack(db.execute("SELECT result FROM jobs WHERE id=?", (first_job,)).fetchone()[0])
            full_stream = list(stream_key(first))
            stratum_id = digest([full_stream, metric, key[4]])
            metric_strata.append({"stratum_id": stratum_id, "stream": full_stream, "bucket": key[4],
                                  "population_n": strata[key], "selected_n": len(chosen),
                                  "inclusion_probability": len(chosen)/strata[key]})
            for _, job_id in sorted(chosen):
                r = unpack(db.execute("SELECT result FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
                rid = digest([response_key(r), r["input_sha256"]])[:24]
                metric_ids.add(rid)
                slim = {k:v for k,v in r.items() if k not in ("raw_provider_response", "judge_prompt")}
                private.append({**slim, "review_id": rid, "stratum_id": stratum_id,
                                "sampling_metric": metric, "sampling_bucket": key[4],
                                "population_n": strata[key], "selected_n": len(chosen),
                                "inclusion_probability": len(chosen)/strata[key]})
        with open(metric_dir / "review_blinded.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(reviews[rid] for rid in sorted(metric_ids, key=lambda x: digest([seed,x])))
        with open(metric_dir / "PRIVATE_sample_key.jsonl", "w", encoding="utf-8") as f:
            for r in private:
                f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
        atomic_json(metric_dir / "manifest.json", {"metric": metric, "seed": seed,
                    "bounds": {"A": "0 <= x < 40", "B": "40 <= x < 80", "C": "80 <= x <= 100"},
                    "strata": metric_strata, "review_ids": sorted(metric_ids),
                    "sampling_method": "smallest seeded SHA256 ranks per fixed population stratum"})
    with open(packet / "REVIEWER_INSTRUCTIONS.txt", "w", encoding="utf-8") as f:
        f.write("Only review_blinded.csv is for annotation. Keep PRIVATE files hidden until labels are final.\n"
                "Label HELD / VIOLATED / CANNOT_ASSESS from the supplied rubric, with evidence and reviewer name.\n"
                "Leave unfamiliar languages as CANNOT_ASSESS; record coherence and actual output language separately.\n"
                "The union sheet deduplicates responses across judges and two planned sampling metrics.\n"
                "Metric-specific subfolders are standard analysis packets; split completed union labels by review_id before analyze.py.\n"
                "No human labels are prefilled. Exact ties and unscored cases are separate abstentions in checkpoint/coverage.\n")
    # Exact ties and other unscored cases remain in the checkpoint and coverage,
    # never removed from denominators or given an invented binary verdict.
    paired = [dict(r) for r in db.execute("""SELECT a.judge_id AS judge_a,b.judge_id AS judge_b,
             COUNT(*) AS comparable_n,SUM(a.verdict=b.verdict) AS agreement_n
             FROM jobs a JOIN jobs b ON a.input_id=b.input_id AND a.judge_id<b.judge_id
             WHERE a.status='ok' AND b.status='ok' GROUP BY a.judge_id,b.judge_id""")]
    summary = {"manifest_fingerprint": manifest["fingerprint"], "input_count": manifest["input_count"],
               "coverage": [{"judge": k[0], "language": k[1], "category": k[2], "status": k[3], "n": v} for k,v in sorted(counts.items())],
               "buckets": [{"stream": list(k[:3]), "metric": k[3], "bucket": k[4], "n": v} for k,v in sorted(strata.items())],
               "paired_agreement": paired, "human_review_n": len(reviews), "human_labels_collected": 0,
               "usage_and_status": status_report(db, "completed"),
               "interpretation": "New judge stream; inter-judge agreement is not human-validated accuracy. Missing input cells and excluded deterministic rows remain in manifest."}
    atomic_json(output / "summary.json", summary)
    set_meta(db, "phase", "completed")
    db.commit()
    progress(output, db, "completed", log)
    db.close()


def main():
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--input", required=True)
    prep.add_argument("--config", required=True)
    prep.add_argument("--output-dir", required=True)
    for command in ("run", "finalize"):
        p = sub.add_parser(command)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--log", required=True)
        if command == "run":
            p.add_argument("--max-new-attempts", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = initialize(args.input, json.loads(Path(args.config).read_text()), args.output_dir)
        print(json.dumps({k: manifest[k] for k in ("input_count", "planned_judgments", "fingerprint")}), flush=True)
    elif args.command == "run":
        run(args.output_dir, args.log, args.max_new_attempts)
    else:
        finalize(args.output_dir, args.log)


if __name__ == "__main__":
    main()
