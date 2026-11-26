"""
Paper experiments for uncertainty-aware CSI localization, run on the
trustworthy-uq engine.

This is a thin wrapper over ``trustworthy_uq.LocalizationUQ.run_experiment``.
It reproduces the workflows of ``Trustworthy_Localization_Experiments_full_modular.py``
(kept unchanged as the reference implementation), but the models (ADN backbone,
one-sided multi-SLA CQR head, Adaptive SCP scale model), the data splits,
conformal calibration, evaluation and the CSV / JSON / plot artifacts all come
from the engine. This script only:

- maps the paper configuration onto ``run_experiment``,
- seeds the run the same way as the original script,
- logs parameters, metrics and the produced artifacts to MLflow.

Supported workflows (same as the original script):
- Methods: ``"ASCP"`` (Adaptive SCP with a learned scale model) and ``"CQR"``
  (one-sided multi-SLA CQR on a frozen ADN backbone).
- Evaluation modes: ``"pooled"`` (dynamic scenarios 0-5) or ``"single"``.
- ADN backbone: load a checkpoint, or train a new one on the static scenario.
- CQR head / ASCP scale model: load a checkpoint, or train from scratch.
- Optional single-sample evaluation with a prediction-disk plot.

Install the engine first:

    pip install git+https://github.com/6G-DALI/trustworthy-uq.git

Then edit the configuration block below, or override it from the command line:

    python Trustworthy_Localization_Experiments_uq_engine.py --method CQR --eval-mode pooled
    python Trustworthy_Localization_Experiments_uq_engine.py --method ASCP --no-mlflow
    python Trustworthy_Localization_Experiments_uq_engine.py --method CQR --eval-mode single --scenario 3 --train-cqr-head
"""

import argparse
import time
from pathlib import Path
from typing import List, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import seed_everything

from trustworthy_uq import LocalizationUQ

BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# User-configurable section (same names and defaults as the original script)
# ============================================================================

# Method: "ASCP" (adaptive SCP with scale model) or "CQR" (two-stage, one-sided CQR)
METHOD: str = "CQR"

# Evaluation mode: "pooled" (all dynamic scenarios 0-5) or "single" (one scenario)
EVAL_MODE: str = "pooled"

# If EVAL_MODE == "single", specify the scenario ID (0-5)
SINGLE_SCENARIO_ID: int = 0

# Backbone checkpoint handling
USE_EXISTING_BACKBONE_CHECKPOINT: bool = True
# random attenuation augmentation checkpoint model (best)
BACKBONE_CHECKPOINT_PATH: Path = BASE_DIR / "mlartifacts/851097306110734214/c5ce26cc459c457fb37c42b721f203d6/artifacts/checkpoints/best_model.ckpt"
# vanilla augmentation checkpoint model
# BACKBONE_CHECKPOINT_PATH = BASE_DIR / "mlartifacts/851097306110734214/8a31fb1083504eb580e14437b0002d5c/artifacts/checkpoints/best_model.ckpt"
# no augmentation checkpoint model
# BACKBONE_CHECKPOINT_PATH = BASE_DIR / "mlartifacts/851097306110734214/4b7c1e2582a04239ab3fbfe0957605c3/artifacts/checkpoints/best_model.ckpt"

# CQR head checkpoint handling (only used when METHOD == "CQR")
USE_EXISTING_CQR_HEAD: bool = True
CQR_HEAD_CHECKPOINT_PATH: Optional[Path] = BASE_DIR / "483259487935076753/fa4d15510f974de586300cc61026b010/checkpoints/best_cqr_head.ckpt"

# Adaptive SCP scale-model checkpoint (only used when METHOD == "ASCP").
# The original script always trains the scale model; leave this as None to do the same.
ASCP_CHECKPOINT_PATH: Optional[Path] = None

# SLA levels
SLA_LEVELS: List[float] = [0.90, 0.95, 0.99]

# Data and MLflow config
SEED: int = 42
DATA_DIR: Path = BASE_DIR / "data/nomadic_dataset/ULA_lab_LoS"
ENABLE_MLFLOW: bool = True
TRACKING_URI: str = "http://127.0.0.1:8080"  # mlflow server --host 127.0.0.1 --port 8080
EXPERIMENT_NAME: str = "ADN-Conformal-Unified"
RUN_NAME: str = "uq_engine_experiments"

# Backbone training config (static scenario, ADN training)
TRAIN_ADN_IF_NEEDED: bool = False
ADN_MAX_EPOCHS: int = 200
ADN_AUGMENT_METHOD: str = "random_attenuation"  # "random_attenuation", "vanilla", "no_augmentation"

SCENARIO_IDS: List[int] = [0, 1, 2, 3, 4, 5]
NUM_USERS: int = 4
NUM_SAMPLES: int = 240
BATCH_SIZE: int = 32
NUM_WORKERS: int = 0

# Split fractions for head/scale training, conformal calibration, and testing
HEAD_TRAIN_FRACTION: float = 0.25
CALIBRATION_FRACTION: float = 0.45
TEST_FRACTION: float = 0.30

# Training hyperparameters
MAX_EPOCHS_SCALE: int = 60
SCALE_LR: float = 1e-3
SCALE_WEIGHT_DECAY: float = 1e-5

MAX_EPOCHS_CQR_HEAD: int = 100
CQR_HEAD_LR: float = 1e-3
CQR_HEAD_WEIGHT_DECAY: float = 1e-5

# Output directory. Kept separate from the original script's
# output/unified_conformal_experiments so the committed paper outputs are not overwritten.
OUTPUT_DIR: Path = BASE_DIR / "output/uq_engine_experiments"

# Optional single-sample evaluation
ENABLE_SINGLE_SAMPLE_EVAL: bool = True
SINGLE_SAMPLE_INDEX: int = 0  # index within the test split
SINGLE_SAMPLE_SLA: float = 0.95  # SLA level for which to draw the uncertainty circle

# The dataset loader expects "vanilla" or "random attenuation" (with a space), and
# no augmentation is expressed as None. Map the config names used by the paper script.
_AUGMENT_METHODS = {
    "random_attenuation": "random attenuation",
    "vanilla": "vanilla",
    "no_augmentation": None,
}

# Metric-name prefixes used by the original script in MLflow.
_METRIC_PREFIX = {"CQR": "cqr", "ASCP": "adaptivescp_scale"}
_METHOD_NAME = {"CQR": "CQR-OneSided", "ASCP": "AdaptiveSCP-Scale"}


# ============================================================================
# Helpers
# ============================================================================


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=["CQR", "ASCP"], type=str.upper, default=METHOD)
    p.add_argument("--eval-mode", choices=["pooled", "single"], type=str.lower, default=EVAL_MODE)
    p.add_argument("--scenario", type=int, default=SINGLE_SCENARIO_ID,
                   help="scenario ID (0-5) when --eval-mode single")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--backbone-checkpoint", type=Path,
                   default=BACKBONE_CHECKPOINT_PATH if USE_EXISTING_BACKBONE_CHECKPOINT else None,
                   help="ADN checkpoint to load")
    p.add_argument("--train-adn", action="store_true", default=TRAIN_ADN_IF_NEEDED,
                   help="train a new ADN backbone on the static scenario instead of loading one")
    p.add_argument("--adn-augment", choices=list(_AUGMENT_METHODS), default=ADN_AUGMENT_METHOD)
    p.add_argument("--cqr-checkpoint", type=Path,
                   default=CQR_HEAD_CHECKPOINT_PATH if USE_EXISTING_CQR_HEAD else None)
    p.add_argument("--train-cqr-head", action="store_true",
                   help="train a new CQR head instead of loading --cqr-checkpoint")
    p.add_argument("--ascp-checkpoint", type=Path, default=ASCP_CHECKPOINT_PATH,
                   help="Adaptive SCP scale-model checkpoint to load (default: train a new one)")
    p.add_argument("--tracking-uri", default=TRACKING_URI)
    p.add_argument("--no-mlflow", dest="mlflow", action="store_false", default=ENABLE_MLFLOW)
    p.add_argument("--no-single-sample", dest="single_sample", action="store_false",
                   default=ENABLE_SINGLE_SAMPLE_EVAL)
    args = p.parse_args(argv)

    if args.train_adn:
        args.backbone_checkpoint = None
    elif args.backbone_checkpoint is None:
        p.error("no ADN backbone: set USE_EXISTING_BACKBONE_CHECKPOINT / --backbone-checkpoint, "
                "or TRAIN_ADN_IF_NEEDED / --train-adn")
    if args.train_cqr_head:
        args.cqr_checkpoint = None
    if args.method == "CQR" and args.cqr_checkpoint is None:
        args.train_cqr_head = True
    return args


def split_sizes(n_scenarios: int):
    """Sizes of the head/calibration/test splits, as computed by LocalizationUQ.split_dataset."""
    n = n_scenarios * NUM_USERS * NUM_SAMPLES
    n_head = int(round(n * HEAD_TRAIN_FRACTION))
    n_cal = int(round(n * CALIBRATION_FRACTION))
    return n, n_head, n_cal, n - n_head - n_cal


class _NoRun:
    """Stand-in for an MLflow run context when logging is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _start_run(args, run_name: str, nested: bool = False):
    if not args.mlflow:
        return _NoRun()
    return mlflow.start_run(run_name=run_name, nested=nested)


def log_results(args, method: str, eval_tag: str, metrics: pd.DataFrame,
                out_dir: Path, runtime_seconds: float) -> None:
    """Log params, metrics and artifacts with the metric names used by the original script."""
    n_scen = len(SCENARIO_IDS) if args.eval_mode == "pooled" else 1
    n_total, n_head, n_cal, n_test = split_sizes(n_scen)

    mlflow.log_params({
        "method": _METHOD_NAME[method],
        "engine": "trustworthy_uq.LocalizationUQ",
        "eval_mode": eval_tag,
        "sla_levels": SLA_LEVELS,
        "seed": SEED,
        "split": (HEAD_TRAIN_FRACTION, CALIBRATION_FRACTION, TEST_FRACTION),
        "backbone_checkpoint": str(args.backbone_checkpoint) if args.backbone_checkpoint else "trained",
        "cqr_checkpoint": str(args.cqr_checkpoint) if method == "CQR" and args.cqr_checkpoint else "",
        "ascp_checkpoint": str(args.ascp_checkpoint) if method == "ASCP" and args.ascp_checkpoint else "",
    })
    mlflow.log_metrics({
        "experiment_runtime_seconds": float(runtime_seconds),
        "pooled_total_size": n_total,
        "head_train_size": n_head,
        "calibration_size": n_cal,
        "test_size": n_test,
    })
    for _, row in metrics.iterrows():
        prefix = f"{_METRIC_PREFIX[method]}_{str(row['target_assurance']).replace('.', 'p')}"
        for col, value in row.items():
            if col in {"method", "target_assurance"} or value is None:
                continue
            if np.isscalar(value) and pd.notna(value):
                try:
                    mlflow.log_metric(f"{prefix}_{col}", float(value))
                except (TypeError, ValueError):
                    pass
    mlflow.log_artifacts(str(out_dir), artifact_path=f"uq_engine_{method.lower()}")


# ============================================================================
# Main entry point
# ============================================================================


def main(argv=None):
    args = parse_args(argv)
    if abs((HEAD_TRAIN_FRACTION + CALIBRATION_FRACTION + TEST_FRACTION) - 1.0) > 1e-9:
        raise ValueError("HEAD_TRAIN_FRACTION + CALIBRATION_FRACTION + TEST_FRACTION must sum to 1.0")

    seed_everything(SEED, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    if args.mlflow:
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        # Training curves of the ADN / CQR head / scale model go to the active run.
        mlflow.pytorch.autolog(log_models=False, silent=True)

    method = args.method
    eval_tag = "pooled_scenarios" if args.eval_mode == "pooled" else f"single_scenario_{args.scenario}"
    out_dir = args.output_dir / eval_tag / method

    uq = LocalizationUQ(
        sla_levels=SLA_LEVELS, seed=SEED, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        split=(HEAD_TRAIN_FRACTION, CALIBRATION_FRACTION, TEST_FRACTION),
    )

    # Backbone: trained here (not via run_experiment) so its epochs/augmentation
    # do not mix with the head's training kwargs.
    if args.backbone_checkpoint is None:
        print(f"Training AttentionDenseNet backbone with augment_method={args.adn_augment}...")
        with _start_run(args, f"{RUN_NAME}-ADN-train-{args.adn_augment}"):
            result = uq.train_adn(
                str(args.data_dir), output_dir=str(args.output_dir / "adn" / args.adn_augment),
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, num_users=NUM_USERS,
                num_samples=NUM_SAMPLES, augment_method=_AUGMENT_METHODS[args.adn_augment],
                max_epochs=ADN_MAX_EPOCHS, seed=SEED,
            )
            print(f"Best ADN backbone checkpoint: {result['checkpoint_path']}")
            if args.mlflow:
                mlflow.log_params({"adn_augment_method": args.adn_augment, "adn_max_epochs": ADN_MAX_EPOCHS})
                mlflow.log_metric("adn_train_runtime_seconds", float(result["runtime_seconds"]))
                mlflow.log_artifact(result["checkpoint_path"], artifact_path="adn_backbone_checkpoints")

    if method == "CQR":
        train_kwargs = dict(max_epochs=MAX_EPOCHS_CQR_HEAD, lr=CQR_HEAD_LR, weight_decay=CQR_HEAD_WEIGHT_DECAY)
    else:
        train_kwargs = dict(max_epochs=MAX_EPOCHS_SCALE, lr=SCALE_LR, weight_decay=SCALE_WEIGHT_DECAY)

    with _start_run(args, f"{RUN_NAME}-{method}"):
        t0 = time.time()
        metrics, preds = uq.run_experiment(
            data_dir=str(args.data_dir),
            method=method,
            eval_mode=args.eval_mode,
            scenario_id=args.scenario,
            scenario_ids=SCENARIO_IDS,
            adn_checkpoint=str(args.backbone_checkpoint) if args.backbone_checkpoint else None,
            cqr_checkpoint=str(args.cqr_checkpoint) if method == "CQR" and args.cqr_checkpoint else None,
            ascp_checkpoint=str(args.ascp_checkpoint) if method == "ASCP" and args.ascp_checkpoint else None,
            train_new_cqr=method == "CQR" and args.train_cqr_head,
            train_new_ascp=method == "ASCP" and not args.ascp_checkpoint,
            sla_levels=SLA_LEVELS,
            output_dir=str(args.output_dir),
            enable_single_sample=args.single_sample,
            single_sample_index=SINGLE_SAMPLE_INDEX,
            single_sample_sla=SINGLE_SAMPLE_SLA,
            **train_kwargs,
        )
        runtime_seconds = time.time() - t0
        if args.mlflow:
            log_results(args, method, eval_tag, metrics, out_dir, runtime_seconds)

    print(f"Evaluation mode: {eval_tag}, outputs in {out_dir}")
    print(f"{_METHOD_NAME[method]} metrics:\n", metrics.to_string(index=False))
    return metrics, preds


if __name__ == "__main__":
    main()
