"""
Smoke test for the RunPod environment: loads Llama-3.1-8B-Instruct and
Qwen3-8B, runs one short generation on each, and reports pass/fail + GPU
memory used. Run after `source /workspace/.runpod_env`.

    python infra/runpod/smoke_test.py
    python infra/runpod/smoke_test.py --models meta-llama/Llama-3.1-8B-Instruct

Both models are gated/require the HF hub license accepted on huggingface.co
under the account whose HF_TOKEN is exported in this shell.
"""

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]

PROMPT = "In one sentence, what is the capital of France?"


def run_one(model_id: str) -> bool:
    print(f"\n{'=' * 60}\n{model_id}\n{'=' * 60}")
    t0 = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    except Exception as e:
        print(f"FAIL (load): {e}")
        return False

    load_s = time.time() - t0
    print(f"Loaded in {load_s:.1f}s")

    try:
        messages = [{"role": "user", "content": PROMPT}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        t1 = time.time()
        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=32, do_sample=False)
        gen_s = time.time() - t1

        text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        print(f"Generated in {gen_s:.1f}s: {text!r}")
    except Exception as e:
        print(f"FAIL (generate): {e}")
        return False
    finally:
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"Peak GPU memory: {mem_gb:.1f} GB")
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Did you `source /workspace/.runpod_env`?")
        sys.exit(1)

    results = {m: run_one(m) for m in args.models}

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for m, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {m}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
