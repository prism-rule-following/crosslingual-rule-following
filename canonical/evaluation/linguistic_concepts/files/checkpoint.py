#!/usr/bin/env python3
"""
Drive-backed, row-level checkpointing for extraction.

Layout (under checkpoint.drive_dir):
  <drive_dir>/
    <model>/
      row_activations/
        <row_id>.pt         # dict: {member: {position: tensor[nl+1, d]}}
      manifest.json         # {"done_ids":[...], "model":..., "n_layers":..., "d_model":...}

resume():  returns the set of already-cached row ids so extraction skips them.
save_row(): writes one row shard + updates manifest (flush_every controls cadence).
load_all(): reconstructs the in-memory store for the DIM math from shards.
"""
import os, json, torch

class RowCheckpoint:
    def __init__(self, cfg, model_key):
        cc = cfg["checkpoint"]
        self.enabled = cc.get("enabled", True)
        self.base = os.path.join(cc["drive_dir"], model_key)
        self.rows_dir = os.path.join(self.base, cc.get("row_cache_subdir", "row_activations"))
        self.manifest_path = os.path.join(self.base, cc.get("manifest_name", "manifest.json"))
        self.flush_every = cc.get("flush_every", 1)
        self.resume_on = cc.get("resume", True)
        self._pending = 0
        self.manifest = {"model": model_key, "done_ids": []}
        if self.enabled:
            os.makedirs(self.rows_dir, exist_ok=True)
            if self.resume_on and os.path.exists(self.manifest_path):
                try:
                    self.manifest = json.load(open(self.manifest_path))
                except Exception:
                    pass

    def done_ids(self):
        if not self.enabled: return set()
        # trust files on disk over manifest in case of a mid-write crash
        ids = set(os.path.splitext(f)[0] for f in os.listdir(self.rows_dir)) if os.path.isdir(self.rows_dir) else set()
        return ids

    def save_row(self, row_id, payload, meta=None):
        if not self.enabled: return
        fp = os.path.join(self.rows_dir, f"{row_id}.pt")
        tmp = fp + ".tmp"
        torch.save({"payload": payload, "meta": meta or {}}, tmp)
        os.replace(tmp, fp)  # atomic
        if row_id not in self.manifest["done_ids"]:
            self.manifest["done_ids"].append(row_id)
        self._pending += 1
        if self._pending >= self.flush_every:
            self._flush()

    def _flush(self):
        tmp = self.manifest_path + ".tmp"
        json.dump(self.manifest, open(tmp, "w"))
        os.replace(tmp, self.manifest_path)
        self._pending = 0

    def finalize(self):
        if self.enabled: self._flush()

    def load_all(self):
        """Return list of (row_id, payload, meta) for every cached shard."""
        out = []
        if not self.enabled or not os.path.isdir(self.rows_dir): return out
        for f in sorted(os.listdir(self.rows_dir)):
            if not f.endswith(".pt"): continue
            d = torch.load(os.path.join(self.rows_dir, f), map_location="cpu")
            out.append((os.path.splitext(f)[0], d["payload"], d.get("meta", {})))
        return out
