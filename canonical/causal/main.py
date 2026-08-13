from argparse import ArgumentParser, Namespace

from dotenv import load_dotenv
from huggingface_hub import HfApi

from causal.attribution_patching import run
from causal.model import EAPConfig, EAPResults

load_dotenv()


def upload_results_to_hf(repo_id: str, prefix: str, files: dict[str, str]):
    api = HfApi()
    for remote_name, local_path in files.items():
        if local_path is None:
            continue  # e.g. pygraphviz/matplotlib weren't available, so this artifact was never generated
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{prefix}/{remote_name}",
            repo_id=repo_id,
            repo_type="dataset",
        )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default="normal",
        help="'normal' scores the first response token; 'teacher_forced' scores a forced multi-token response",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model id of the model to attribute",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=50,
        help="number of examples per dataloader batch",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=False,
        default="logit_diff",
        help="which metric to attribute/evaluate with respect to",
    )
    parser.add_argument(
        "--ig_steps",
        type=int,
        required=False,
        default=10,
        help="number of integrated-gradients interpolation steps",
    )
    parser.add_argument(
        "--n_edge_start",
        type=int,
        required=False,
        default=50,
        help="smallest n_edges tested in the sweep",
    )
    parser.add_argument(
        "--n_edge_steps",
        type=int,
        required=False,
        default=20,
        help="number of points to sample in the n_edges sweep",
    )
    parser.add_argument(
        "--n_edge_end_proportion",
        type=float,
        required=False,
        default=0.05,
        help="largest n_edges tested in the sweep, as a proportion of the graph's total real edges",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=False,
        default="EAP-IG-activations",
        help="attribution method: EAP, EAP-IG-inputs, EAP-IG-activations, or clean-corrupted",
    )
    parser.add_argument(
        "--data_url",
        type=str,
        required=True,
        help="URL of the dataset JSON file to attribute over",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        required=True,
        help="dataset source, e.g. 'gh'",
    )
    parser.add_argument(
        "--data_lang",
        type=str,
        required=True,
        help="language of the rows to attribute over, e.g. 'en'",
    )
    parser.add_argument(
        "--data_category",
        type=str,
        required=True,
        help="rule category to attribute over, e.g. 'bold_html'",
    )
    parser.add_argument(
        "--upload_target",
        type=str,
        required=False,
        default=None,
        choices=["huggingface"],
        help="where to upload results and artifacts; omit to skip uploading entirely",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=False,
        help="HF dataset repo id, e.g. 'org/dataset-name'; required if --upload_target=huggingface",
    )
    return parser


def build_config(args: Namespace) -> EAPConfig:
    return EAPConfig(
        mode=args.mode,
        model_id=args.model_id,
        dataset_config={
            "url": args.data_url,
            "source": args.data_source,
            "language": [args.data_lang],
            "category": [args.data_category],
        },
        batch_size=args.batch_size,
        metrics=args.metrics,
        method=args.method,
        ig_steps=args.ig_steps,
        n_edge_steps=args.n_edge_steps,
        n_edge_start=args.n_edge_start,
        n_edge_end_proportion=args.n_edge_end_proportion,
    )


def build_upload_prefix(config: EAPConfig) -> str:
    model_id = config.model_id.split("/")[1]
    return (
        f"{model_id}_{config.dataset_config.language[0].value}_"
        f"{config.method.value}_{config.dataset_config.category[0].value}"
    )


def maybe_upload(args: Namespace, config: EAPConfig, result: EAPResults) -> None:
    if not args.upload_target:
        return

    prefix = build_upload_prefix(config)
    results_json_path = f"{prefix}_result.json"
    with open(results_json_path, "w") as f:
        f.write(result.model_dump_json(indent=2))

    files = {
        "results.json": results_json_path,
        "best_circuit.png": result.circuit_image_path,
        "knee_plot.png": result.knee_image_path,
    }

    if args.upload_target == "huggingface":
        upload_results_to_hf(repo_id=args.repo_id, prefix=prefix, files=files)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.upload_target == "huggingface" and not args.repo_id:
        parser.error("--repo_id is required when --upload_target=huggingface")

    config = build_config(args)
    result = run(config)
    print(result)

    maybe_upload(args, config, result)


if __name__ == "__main__":
    main()
