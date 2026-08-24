"""
1. Train on shuffled labels DONE
2. p-value, subsampling DONE
3. cross-validation
4. Run on all the confound datasets DONE
"""

from typing import Any, Callable, Dict, List, Optional

import json
import numpy as np
from canonical.probing.config import RunConfig
from canonical.probing.evaluate_probes import evaluate
from canonical.probing.probes_dataset_creation_script import (
    create_canonical_dataset,
    distractor_word_data,
    neutral_filler_data,
    no_rule_keyword,
    opposite_statuses_rules,
)
from canonical.probing.pydantic_models import ShuffledLabelsResults
from canonical.probing.train_probes import training
from canonical.probing.utils import (
    check_clf_match,
    load_trained_clfs,
    save_clf_with_skops,
    split_by_layer,
)
from sklearn.metrics import classification_report, roc_auc_score


def _safe_roc_auc(y_true, y_score) -> float:
    """roc_auc_score needs both classes present; a permutation can produce only one by chance."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)


METRIC_EXTRACTORS = {
    "roc_auc": lambda report: report["roc_auc"],
    "accuracy": lambda report: report["accuracy"],
    "macro_precision": lambda report: report["macro avg"]["precision"],
    "macro_recall": lambda report: report["macro avg"]["recall"],
    "macro_f1": lambda report: report["macro avg"]["f1-score"],
}


def train_on_shuffled_labels(
    cfg: RunConfig,
    jsonl_in_hf: str,
    hf_repo_ix: str,
    classifiers: List[Callable],
    hf_repo_type: str = "dataset",
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    model: Any = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
) -> ShuffledLabelsResults:
    """Control training on shuffled labels."""
    # loading and shuffling the data
    dataset = create_canonical_dataset(
        jsonl_in_hf,
        hf_repo_ix,
        hf_repo_type=hf_repo_type,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
    )
    np.random.shuffle(dataset.train_y)

    # training on shuffled labels
    shuffled_path_prefix = "ShuffledLabels"
    shuffled_clfs = training(
        cfg,
        classifiers,
        split_by_layer(dataset.train_x),
        dataset.train_y,
    )
    shuffled_train_path = save_clf_with_skops(
        cfg, shuffled_clfs, save_path_prefix=shuffled_path_prefix
    )

    # evaluating shuffled labels
    shuffled_eval_results, shuffled_eval_path = evaluate(
        cfg,
        shuffled_clfs,
        split_by_layer(dataset.test_x),
        dataset.test_y,
        save_path_prefix=shuffled_path_prefix,
    )
    return ShuffledLabelsResults(
        shuffled_eval_results=shuffled_eval_results,
        shuffled_eval_path=shuffled_eval_path,
        shuffled_train_path=shuffled_train_path,
    )


def p_value_control(
    cfg: RunConfig,
    jsonl_in_hf: str,
    hf_repo_ix: str,
    classifiers: List[Callable],
    hf_repo_type: str = "dataset",
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    model: Any = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    n_perm: int = 1000,
    save_path_prefix: str = "PValue",
    load_normal_eval_scores: Optional[str] = None,
) -> Dict[str, Dict[str, Dict]]:
    """Permutation test: max-statistic across layers per classifier type, corrected
    p-value comparing real train-label performance against permuted-label performance."""
    dataset = create_canonical_dataset(
        jsonl_in_hf,
        hf_repo_ix,
        hf_repo_type=hf_repo_type,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
    )
    train_X = split_by_layer(dataset.train_x)
    test_X = split_by_layer(dataset.test_x)

    def best_per_layer(train_y):
        trained = training(cfg, classifiers, train_X, train_y)
        scores = {}
        for name, layer_clfs in trained.items():
            layer_reports = {}
            for layer, clf in layer_clfs.items():
                predictions = clf.predict(test_X[layer])
                report = classification_report(dataset.test_y, predictions, output_dict=True)
                report["roc_auc"] = _safe_roc_auc(
                    dataset.test_y, clf.predict_proba(test_X[layer])[:, 1]
                )
                layer_reports[layer] = report
            scores[name] = {
                m: max(f(r) for r in layer_reports.values()) for m, f in METRIC_EXTRACTORS.items()
            }
        return scores

    if load_normal_eval_scores:
        with open(load_normal_eval_scores) as f:
            loaded = json.load(f)
        check_clf_match(loaded.keys(), classifiers)
        observed_best = {
            name: {
                m: max(f(r) for r in layer_dict.values()) for m, f in METRIC_EXTRACTORS.items()
            }
            for name, layer_dict in loaded.items()
        }
    else:
        observed_best = best_per_layer(dataset.train_y)

    rng = np.random.default_rng(42)
    null_max = {
        name: {m: np.empty(n_perm) for m in METRIC_EXTRACTORS} for name in observed_best
    }
    for i in range(n_perm):
        perm_best = best_per_layer(rng.permutation(dataset.train_y))
        for name in null_max:
            for m in METRIC_EXTRACTORS:
                null_max[name][m][i] = perm_best[name][m]

    p_value_results = {
        name: {
            m: {
                "observed": observed_best[name][m],
                "p_value": float(
                    (1 + np.sum(null_max[name][m] >= observed_best[name][m])) / (n_perm + 1)
                ),
            }
            for m in METRIC_EXTRACTORS
        }
        for name in null_max
    }
    with open(f"{cfg.eval_path}/{save_path_prefix}_{cfg.language}.json", "w") as f:
        json.dump(p_value_results, f)
    return p_value_results


def _confound_p_value(
    trained_classifiers: Dict[str, Dict[int, object]],
    confound_X: Dict[int, np.ndarray],
    confound_y: np.ndarray,
    n_perm: int = 1000,
) -> Dict[str, Dict[str, Dict]]:
    """Keeps an already-trained probe's predictions fixed and permutes the confound
    dataset's labels, to test whether its performance on this confound set is above chance."""
    rng = np.random.default_rng(42)
    results = {}
    for name, layer_clfs in trained_classifiers.items():
        predictions = {layer: clf.predict(confound_X[layer]) for layer, clf in layer_clfs.items()}
        probabilities = {
            layer: clf.predict_proba(confound_X[layer])[:, 1] for layer, clf in layer_clfs.items()
        }

        def best_across_layers(y):
            layer_reports = {}
            for layer in predictions:
                report = classification_report(y, predictions[layer], output_dict=True)
                report["roc_auc"] = _safe_roc_auc(y, probabilities[layer])
                layer_reports[layer] = report
            return {
                m: max(f(r) for r in layer_reports.values()) for m, f in METRIC_EXTRACTORS.items()
            }

        observed = best_across_layers(confound_y)
        null_max = {m: np.empty(n_perm) for m in METRIC_EXTRACTORS}
        for i in range(n_perm):
            perm_best = best_across_layers(rng.permutation(confound_y))
            for m in METRIC_EXTRACTORS:
                null_max[m][i] = perm_best[m]

        results[name] = {
            m: {
                "observed": observed[m],
                "p_value": float((1 + np.sum(null_max[m] >= observed[m])) / (n_perm + 1)),
            }
            for m in METRIC_EXTRACTORS
        }
    return results


def neutral_filler_control(
    cfg: RunConfig,
    trained_classifiers: Dict[str, Dict[int, object]],
    neutral_fillers_path: str,
    model: Any,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    n_perm: int = 1000,
) -> Dict[str, Dict[str, Dict]]:
    """Checks whether the probe still performs above chance with the rule replaced by neutral text."""
    dataset = neutral_filler_data(
        neutral_fillers_path, model, hook_name=hook_name, pos_slice=pos_slice
    )
    results = _confound_p_value(
        trained_classifiers,
        split_by_layer(dataset.neutral_x),
        np.array(dataset.neutral_y),
        n_perm=n_perm,
    )
    with open(f"{cfg.eval_path}/NeutralFiller_{cfg.language}.json", "w") as f:
        json.dump(results, f)
    return results


def distractor_control(
    cfg: RunConfig,
    trained_classifiers: Dict[str, Dict[int, object]],
    original_text_hf: str,
    hf_repo_ix: str,
    model: Any,
    hf_repo_type: str = "dataset",
    n_perm: int = 1000,
) -> Dict[str, Dict[str, Dict]]:
    """Checks whether the probe still performs above chance with 'Rule status:' replaced by a distractor word."""
    dataset = distractor_word_data(original_text_hf, hf_repo_ix, model, hf_repo_type=hf_repo_type)
    results = _confound_p_value(
        trained_classifiers,
        split_by_layer(dataset.distractor_x),
        np.array(dataset.distractor_y),
        n_perm=n_perm,
    )
    with open(f"{cfg.eval_path}/Distractor_{cfg.language}.json", "w") as f:
        json.dump(results, f)
    return results


def no_keyword_control(
    cfg: RunConfig,
    trained_classifiers: Dict[str, Dict[int, object]],
    jsonl_in_hf: str,
    hf_repo_ix: str,
    model: Any,
    hf_repo_type: str = "dataset",
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    n_perm: int = 1000,
) -> Dict[str, Dict[str, Dict]]:
    """Checks whether the probe still performs above chance with no explicit 'Rule' keyword present."""
    dataset = no_rule_keyword(
        model,
        jsonl_in_hf,
        hf_repo_ix,
        repo_type=hf_repo_type,
        hook_name=hook_name,
        pos_slice=pos_slice,
    )
    results = _confound_p_value(
        trained_classifiers,
        split_by_layer(dataset.nokrule_x),
        np.array(dataset.nokrule_y),
        n_perm=n_perm,
    )
    with open(f"{cfg.eval_path}/NoKeyword_{cfg.language}.json", "w") as f:
        json.dump(results, f)
    return results


def double_rule_control(
    cfg: RunConfig,
    trained_classifiers: Dict[str, Dict[int, object]],
    filepath: str,
    model: Any,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    n_perm: int = 1000,
) -> Dict[str, Dict[str, Dict]]:
    """Checks whether the probe survives two rules with opposite statuses in the same system prompt."""
    dataset = opposite_statuses_rules(filepath, model, hook_name=hook_name, pos_slice=pos_slice)
    results = _confound_p_value(
        trained_classifiers,
        split_by_layer(dataset.doublerule_x),
        np.array(dataset.doublerule_y),
        n_perm=n_perm,
    )
    with open(f"{cfg.eval_path}/DoubleRule_{cfg.language}.json", "w") as f:
        json.dump(results, f)
    return results


def confound_datasets_control(
    cfg: RunConfig,
    jsonl_in_hf: str,
    hf_repo_ix: str,
    classifiers: List[Callable],
    model: Any,
    neutral_fillers_path: str,
    original_text_hf: str,
    no_keyword_jsonl_in_hf: str,
    double_rule_filepath: str,
    hf_repo_type: str = "dataset",
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    trained_clfs_folder: Optional[str] = None,
    n_perm: int = 1000,
) -> Dict[str, Dict[str, Dict]]:
    """Checks how the probe performs on the confound datasets:
    1) neutral filler
    2) distractor word dataset
    3) no rule keyword dataset
    4) double rule dataset

    Each evaluation for each dataset is in its own function so it can be run separately.
    This function just runs them all at once and co-ordinates.
    """
    if trained_clfs_folder:
        trained_classifiers = load_trained_clfs(trained_clfs_folder, cfg.language)
        check_clf_match(trained_classifiers.keys(), classifiers)
    else:
        dataset = create_canonical_dataset(
            jsonl_in_hf,
            hf_repo_ix,
            hf_repo_type=hf_repo_type,
            activations_in_hf=activations_in_hf,
            y_in_hf=y_in_hf,
            model=model,
            hook_name=hook_name,
            pos_slice=pos_slice,
        )
        trained_classifiers = training(
            cfg, classifiers, split_by_layer(dataset.train_x), dataset.train_y
        )

    return {
        "neutral_filler": neutral_filler_control(
            cfg, trained_classifiers, neutral_fillers_path, model, n_perm=n_perm
        ),
        "distractor": distractor_control(
            cfg,
            trained_classifiers,
            original_text_hf,
            hf_repo_ix,
            model,
            hf_repo_type=hf_repo_type,
            n_perm=n_perm,
        ),
        "no_keyword": no_keyword_control(
            cfg,
            trained_classifiers,
            no_keyword_jsonl_in_hf,
            hf_repo_ix,
            model,
            hf_repo_type=hf_repo_type,
            n_perm=n_perm,
        ),
        "double_rule": double_rule_control(
            cfg, trained_classifiers, double_rule_filepath, model, n_perm=n_perm
        ),
    }


def weights_vs_diff_of_means(
    cfg: RunConfig,
    jsonl_in_hf: str,
    hf_repo_ix: str,
    classifiers: List[Callable],
    hf_repo_type: str = "dataset",
    activations_in_hf: Optional[str] = None,
    y_in_hf: Optional[str] = None,
    model: Any = None,
    hook_name: str = "hook_resid_post",
    pos_slice: int = -1,
    trained_clfs_folder: Optional[str] = None,
) -> Dict[str, Dict[int, float]]:
    """Checks how similar the probe's learnt weights are to the diff of means.
    diff_of_means = X[where X=1] - X[where X=0]. Then cosine similarity on clf.coef_"""
    dataset = create_canonical_dataset(
        jsonl_in_hf,
        hf_repo_ix,
        hf_repo_type=hf_repo_type,
        activations_in_hf=activations_in_hf,
        y_in_hf=y_in_hf,
        model=model,
        hook_name=hook_name,
        pos_slice=pos_slice,
    )
    train_X = split_by_layer(dataset.train_x)
    if trained_clfs_folder:
        trained_classifiers = load_trained_clfs(trained_clfs_folder, cfg.language)
        check_clf_match(trained_classifiers.keys(), classifiers)
    else:
        trained_classifiers = training(cfg, classifiers, train_X, dataset.train_y)

    results = {}
    for name, layer_clfs in trained_classifiers.items():
        results[name] = {}
        for layer, clf in layer_clfs.items():
            if not hasattr(clf, "coef_"):
                continue
            diff_of_means = (
                train_X[layer][dataset.train_y == 1].mean(axis=0)
                - train_X[layer][dataset.train_y == 0].mean(axis=0)
            )
            weights = clf.coef_.flatten()
            cosine_sim = np.dot(weights, diff_of_means) / (
                np.linalg.norm(weights) * np.linalg.norm(diff_of_means)
            )
            results[name][layer] = float(cosine_sim)

    with open(f"{cfg.eval_path}/WeightsVsDiffMeans_{cfg.language}.json", "w") as f:
        json.dump(results, f)
    return results


def cross_validation_training():
    pass


def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Control experiments for probing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_args(sp, required=True):
        sp.add_argument("--jsonl-in-hf", required=required)
        sp.add_argument("--hf-repo-ix", required=required)
        sp.add_argument("--hf-repo-type", default="dataset")
        sp.add_argument(
            "--activations-in-hf",
            default=None,
            help="If given, download precomputed activations instead of extracting them.",
        )
        sp.add_argument("--y-in-hf", default=None)
        sp.add_argument(
            "--model-name",
            default=None,
            help="Required when activations aren't precomputed, to extract them on the fly.",
        )
        sp.add_argument("--hook-name", default="hook_resid_post")
        sp.add_argument("--pos-slice", type=int, default=-1)
        sp.add_argument(
            "--classifiers",
            required=True,
            help="JSON object mapping classifier name to kwargs.",
        )
        sp.add_argument("--language", required=True)
        sp.add_argument("--n-layers", type=int, required=True)
        sp.add_argument("--dataset-name", required=True)
        sp.add_argument("--results-folder", default=None)

    train_shuffled = subparsers.add_parser("train-shuffled")
    add_data_args(train_shuffled)

    p_value = subparsers.add_parser("p-value")
    add_data_args(p_value)
    p_value.add_argument("--n-perm", type=int, default=1000)
    p_value.add_argument("--save-path-prefix", default="PValue")
    p_value.add_argument("--load-normal-eval-scores", default=None)

    weights_vs_means = subparsers.add_parser("weights-vs-means")
    add_data_args(weights_vs_means)
    weights_vs_means.add_argument("--trained-clfs-folder", default=None)

    confound = subparsers.add_parser("confound-datasets")
    add_data_args(confound, required=False)
    confound.add_argument("--llm", required=True)
    confound.add_argument("--neutral-fillers-path", required=True)
    confound.add_argument("--original-text-hf", required=True)
    confound.add_argument("--no-keyword-jsonl-in-hf", required=True)
    confound.add_argument("--double-rule-filepath", required=True)
    confound.add_argument("--trained-clfs-folder", default=None)
    confound.add_argument("--n-perm", type=int, default=1000)

    return parser


def main():
    from canonical.probing.utils import build_classifiers, create_results_path

    args = _build_arg_parser().parse_args()

    cfg = RunConfig(
        language=args.language,
        n_layers=args.n_layers,
        dataset_name=args.dataset_name,
        results_folder=args.results_folder,
    )
    create_results_path(cfg)
    classifiers = build_classifiers(json.loads(args.classifiers))

    if args.command in ("train-shuffled", "p-value", "weights-vs-means"):
        model = None
        if args.model_name:
            from transformer_lens import HookedTransformer

            model = HookedTransformer.from_pretrained(args.model_name)

    if args.command == "train-shuffled":
        results = train_on_shuffled_labels(
            cfg,
            args.jsonl_in_hf,
            args.hf_repo_ix,
            classifiers,
            hf_repo_type=args.hf_repo_type,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            model=model,
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
        )
    elif args.command == "p-value":
        results = p_value_control(
            cfg,
            args.jsonl_in_hf,
            args.hf_repo_ix,
            classifiers,
            hf_repo_type=args.hf_repo_type,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            model=model,
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
            n_perm=args.n_perm,
            save_path_prefix=args.save_path_prefix,
            load_normal_eval_scores=args.load_normal_eval_scores,
        )
    elif args.command == "weights-vs-means":
        results = weights_vs_diff_of_means(
            cfg,
            args.jsonl_in_hf,
            args.hf_repo_ix,
            classifiers,
            hf_repo_type=args.hf_repo_type,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            model=model,
            hook_name=args.hook_name,
            pos_slice=args.pos_slice,
            trained_clfs_folder=args.trained_clfs_folder,
        )
    elif args.command == "confound-datasets":
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(args.llm)
        results = confound_datasets_control(
            cfg,
            args.jsonl_in_hf,
            args.hf_repo_ix,
            classifiers,
            model,
            args.neutral_fillers_path,
            args.original_text_hf,
            args.no_keyword_jsonl_in_hf,
            args.double_rule_filepath,
            hf_repo_type=args.hf_repo_type,
            activations_in_hf=args.activations_in_hf,
            y_in_hf=args.y_in_hf,
            trained_clfs_folder=args.trained_clfs_folder,
            n_perm=args.n_perm,
        )
    print(results)


if __name__ == "__main__":
    main()
