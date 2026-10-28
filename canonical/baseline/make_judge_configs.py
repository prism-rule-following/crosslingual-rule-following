"""Write the 3 judge configs for the current MODELS/LANGS so judging always matches what was generated.
  python canonical/baseline/make_judge_configs.py --models Qwen/Qwen3-8B,meta-llama/Llama-3.1-8B-Instruct --langs en,yo,sw \
      --out-dir /workspace/baseline_judge_cfg
Uses canonical/baseline/judge_configs/*.json as templates (judge model, workers, HF results repo)."""
import argparse, json, os
H = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--langs", required=True)
ap.add_argument("--out-dir", default="/workspace/baseline_judge_cfg")
ap.add_argument("--data", default="/workspace/baseline_judge_in"); ap.add_argument("--results-dir", default="results/baseline")
ap.add_argument("--hf-results-repo", default=None)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
for j in ["gpt_mini", "deepseek", "gemini"]:
    c = json.load(open(f"{H}/judge_configs/judge_{j}_baseline.json"))
    c.update(model_ids=a.models.split(","), language_codes=a.langs.split(","), dataset_repo=a.data, results_dir=a.results_dir)
    if a.hf_results_repo: c["hf_results_repo"] = a.hf_results_repo
    p = f"{a.out_dir}/judge_{j}.json"; json.dump(c, open(p, "w"), indent=2); print("wrote", p)
