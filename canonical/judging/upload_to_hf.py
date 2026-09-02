"""
Uploads final.jsonl for ONE judge to a specific path in your HF dataset
repo, preserving the folder-per-judge structure (gpt_mini/, deepseek/,
gemini/) -- same pattern as the original dataset's model_id folders.

SETUP: add to .env:
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

RUN:
    python upload_to_hf.py gpt_mini
    python upload_to_hf.py deepseek   (once that judge is done)
    python upload_to_hf.py gemini     (once that judge is done)
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from huggingface_hub import HfApi, login

if len(sys.argv) != 2 or sys.argv[1] not in ("gpt_mini", "deepseek", "gemini"):
    raise SystemExit("Usage: python upload_to_hf.py [gpt_mini|deepseek|gemini]")

JUDGE_NAME = sys.argv[1]
LOCAL_FILE = os.path.join("results", JUDGE_NAME, "final.jsonl")

# <-- set your actual target repo
HF_RESULTS_REPO = "crosslingual-rule-following/judge-results-active-only"
PRIVATE = True

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("HF_TOKEN not set -- add it to .env (needs write access).")

if not os.path.exists(LOCAL_FILE):
    raise SystemExit(f"{LOCAL_FILE} not found -- run consolidate.py for this judge first.")

login(token=HF_TOKEN)
api = HfApi()

# Creates the repo if it doesn't exist yet (harmless no-op if it does).
api.create_repo(repo_id=HF_RESULTS_REPO, repo_type="dataset", private=PRIVATE, exist_ok=True)

remote_path = f"{JUDGE_NAME}/results.jsonl"
print(f"Uploading {LOCAL_FILE} -> {HF_RESULTS_REPO}/{remote_path} ...")

api.upload_file(
    path_or_fileobj=LOCAL_FILE,
    path_in_repo=remote_path,
    repo_id=HF_RESULTS_REPO,
    repo_type="dataset",
)

print(f"Done. https://huggingface.co/datasets/{HF_RESULTS_REPO}/tree/main/{JUDGE_NAME}")
