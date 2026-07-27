from causal.attribution_patching import run
from argparse import ArgumentParser

from dotenv import load_dotenv

load_dotenv()


branch = "dataset-design-gen-pipeline"

parser = ArgumentParser()
parser.add_argument("--mode", type=str, required=False, default="normal")
parser.add_argument("--model_id", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=False, default=50)
parser.add_argument("--metrics", type=str, required=False, default="logit_diff")
parser.add_argument("--steps", type=int, required=False, default=10)
parser.add_argument("--n_edges", type=int, required=False, default=20)
parser.add_argument("--method", type=str, required=False, default="EAP-IG-activations")
parser.add_argument("--data_url", type=str, required=True)
parser.add_argument("--data_source", type=str, required=True)
parser.add_argument("--data_lang", type=str, required=True)
parser.add_argument("--data_category", type=str, required=True)

args = parser.parse_args()

config = {
    "mode": args.mode,
    "model_id": args.model_id,
    "dataset_config": {
        "url": args.data_url,
        "source": args.data_source,
        "language": [args.data_lang],
        "category": [args.data_category],
    },
    "batch_size": args.batch_size,
    "metrics": args.metrics,
    "method": args.method,
    "steps": args.steps,
    "n_edges": args.n_edges,
}

result = run(config)
print(result)
