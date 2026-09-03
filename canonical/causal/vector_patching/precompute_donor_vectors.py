"""Runs once, sequentially, before both Stage B recipient processes start.
Downloads each donor language's activations exactly once, builds dom vectors +
per-id held activations, pickles to disk so the recipient processes never
touch HF for this data at all.
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from canonical.causal.vector_patching import pair_selection as ps
from canonical.causal.vector_patching.run_stage_b import (
    DONOR_LANGS,
    MODEL_ID,
    PRESSURE,
    load_donor_activations,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/exp2_out")
    ap.add_argument(
        "--donors",
        default=",".join(DONOR_LANGS),
        help="comma-separated donor languages, default all 8",
    )
    args = ap.parse_args()
    donors = [lang.strip() for lang in args.donors.split(",") if lang.strip()]

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading judge verdicts...", flush=True)
    verdicts = ps.load_judge_verdicts()
    collapsed = ps.collapse_verdicts(verdicts)
    subset = collapsed[(collapsed["model_id"] == MODEL_ID) & (collapsed["pressure_level"] == PRESSURE)]

    print(f"[{time.strftime('%H:%M:%S')}] downloading {len(donors)} donor languages sequentially...", flush=True)
    donor_acts, dom_vectors = None, None
    for attempt in range(5):
        try:
            donor_acts, dom_vectors = load_donor_activations(donors, subset)
            break
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt+1} failed ({e}); "
                  f"retrying in {wait}s", flush=True)
            time.sleep(wait)
    if donor_acts is None:
        raise RuntimeError("failed to load donor activations after 5 attempts")

    out_path = Path(args.out_dir) / "donor_cache.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"donor_acts": donor_acts, "dom_vectors": dom_vectors}, f)
    print(f"[{time.strftime('%H:%M:%S')}] cached {len(donors)} donors to {out_path}, "
          f"took {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()