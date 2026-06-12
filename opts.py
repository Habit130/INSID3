"""Command-line arguments for TFRIS inference."""

import argparse

SUPPORTED_DATASETS = ["refcoco", "refcoco+", "refcocog"]


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("TFRIS inference", add_help=False)

    # Model
    parser.add_argument(
        "--image-size",
        default=1024,
        type=int,
        help="Input image resolution",
    )
    parser.add_argument(
        "--crf-mask-refinement",
        action="store_true",
        help="Enable CRF-based mask refinement.",
    )

    # Hyperparameters
    parser.add_argument(
        "--tau",
        default=0.6,
        type=float,
        help="Clustering distance threshold",
    )
    parser.add_argument(
        "--merge-thresh",
        default=0.32,
        type=float,
        help="Cluster aggregation threshold on the soft score "
             "intra_sim * cross_sim_norm (a cluster joins the output when its "
             "score exceeds this)",
    )
    parser.add_argument(
        "--cand-quantile",
        default=0.9,
        type=float,
        help="Candidate Localization quantile: patches with "
             "sim > quantile(sim, q) become candidates",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="refcoco",
        choices=SUPPORTED_DATASETS,
        help="Dataset for evaluation",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory of datasets",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Evaluation split for RefCOCO-family datasets "
             "(val/testA/testB for refcoco and refcoco+; val/test_U/test_G for refcocog)",
    )

    # Runtime
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Cap inference to the first N expressions; the official evaluator "
             "is skipped (it requires complete splits) and the prediction "
             "contract is validated instead",
    )
    parser.add_argument(
        "--sample",
        default=None,
        type=int,
        help="Evaluate a random subset of N expressions (drawn with --seed). "
             "The official evaluator is skipped (it requires complete splits); "
             "sampled metrics are computed by the script instead and written "
             "to sampled_metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for logs and results",
    )
    parser.add_argument(
        "--exp-name",
        default="tfris-refcoco",
        help="Run name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed (project-wide evaluation protocol fixes seed=42)",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="Number of data loading workers",
    )

    return parser