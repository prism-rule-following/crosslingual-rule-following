"""
run_pipeline.py — generate grammar variants, then verify the result, in one call.

Step 1: generate_grammar_variants.main() expands v2/judgment_rules.json into
        v2/judgment_rules_expanded.json.
Step 2: verify.main() runs uniqueness/schema/deterministic (and optionally
        model) checks over that expanded file and writes the clean, final
        verified file.

Both steps keep writing their own side files (rejects.jsonl,
generation_report.json, verify_rejects.jsonl, verify_report.json) exactly as
they do when run standalone — this script only chains the two calls and
passes the generator's output straight into the verifier's input.
"""

from __future__ import annotations

import argparse

import generate_grammar_variants as gen
import verify as ver
from dotenv import load_dotenv

load_dotenv()


def main(argv: list | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Passed to the generator: no API calls, mock the generator output.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Passed to the generator: process only N scenario cells (0 = all).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Passed to the generator: scenario cells per batched model call.",
    )
    ap.add_argument(
        "--with-verify-model",
        action="store_true",
        help="Passed to verify: run Stage 3 (second-model equivalence check).",
    )
    ap.add_argument(
        "--verified-out",
        default=None,
        help="Output path for the final verified file "
        "(default: <expanded file>.verified.json).",
    )
    args = ap.parse_args(argv)

    print("=== Step 1/2: generate_grammar_variants ===")
    gen_argv = ["--limit", str(args.limit), "--batch-size", str(args.batch_size)]
    if args.dry_run:
        gen_argv.append("--dry-run")
    gen.main(gen_argv)

    print("\n=== Step 2/2: verify ===")
    verify_argv = [str(gen.OUT)]
    if args.with_verify_model:
        verify_argv.append("--with-model")
    verify_out = args.verified_out or str(gen.OUT.with_suffix(".verified.json"))
    verify_argv += ["--out", verify_out]
    ver.main(verify_argv)

    print(f"\n=== Pipeline done: {verify_out} ===")


if __name__ == "__main__":
    main()
