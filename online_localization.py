# -*- coding: utf-8 -*-
"""
Online uncertainty-aware localization with trustworthy-uq.

This script shows how the calibrated UQ engine is used at inference time,
one CSI measurement at a time, as a localization service would:

    raw complex CSI (.npy, 64 antennas x 100 subcarriers)
        ↓
    TrustedLocalizer.locate()
        ↓
    position (mm) + conformal trust radius per SLA level
        ↓
    accept / reject against an accuracy requirement

Engine:
    If ENGINE_DIR (written by full_workflow.py, or by this script on its
    first run) exists, it is loaded directly; no dataset or recalibration is
    needed. Otherwise the engine is built once from the committed ADN and CQR
    checkpoints, calibrated on the calibration split, and saved.

Stream:
    The demo replays the held-out test split of the pooled dynamic scenarios
    (the samples the engine was not calibrated on) straight from the raw
    measurement files, and reports running coverage, how often the trust
    radius meets the requirement, and per-sample latency.

Usage:
    python online_localization.py
    python online_localization.py --engine package_functional_test/engine \
        --sla 0.95 --max-radius-mm 25 --num-samples 500
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "data" / "nomadic_dataset" / "ULA_lab_LoS"

ADN_CHECKPOINT = (
    BASE_DIR / "mlartifacts" / "851097306110734214"
    / "c5ce26cc459c457fb37c42b721f203d6" / "artifacts" / "checkpoints"
    / "best_model.ckpt"
)

CQR_CHECKPOINT = (
    BASE_DIR / "483259487935076753" / "fa4d15510f974de586300cc61026b010"
    / "checkpoints" / "best_cqr_head.ckpt"
)

ENGINE_DIR = BASE_DIR / "package_functional_test" / "engine"

SLA_LEVELS = [0.90, 0.95, 0.99]
SCENARIO_IDS = list(range(6))
NUM_USERS = 4
NUM_SAMPLES = 240
SEED = 42


# =============================================================================
# Engine
# =============================================================================

def build_engine(engine_dir):
    """Calibrate the committed ADN + CQR checkpoints once and save the engine."""
    from trustworthy_uq import LocalizationUQ, TrustedLocalizer

    print("[engine] No saved engine found; calibrating from checkpoints")
    uq = LocalizationUQ(sla_levels=SLA_LEVELS, seed=SEED)
    uq.load_adn(str(ADN_CHECKPOINT))
    uq.load_cqr(str(CQR_CHECKPOINT))

    dataset = uq.build_pooled_dataset(
        str(DATASET_PATH), SCENARIO_IDS, NUM_USERS, NUM_SAMPLES)
    _, calibration, _ = uq.split_dataset(dataset)
    uq.calibrate_cqr(calibration, SLA_LEVELS)

    localizer = TrustedLocalizer(uq, method="CQR", metadata={
        "eval_mode": "pooled",
        "scenario_ids": SCENARIO_IDS,
        "seed": SEED,
        "split": list(uq.split),
        "adn_checkpoint": ADN_CHECKPOINT.relative_to(BASE_DIR).as_posix(),
        "cqr_checkpoint": CQR_CHECKPOINT.relative_to(BASE_DIR).as_posix(),
    })
    manifest = localizer.save(engine_dir)
    print(f"[engine] Saved: {manifest}")
    return localizer


def load_or_build_engine(engine_dir):
    from trustworthy_uq import TrustedLocalizer

    if (Path(engine_dir) / "engine.json").exists():
        localizer = TrustedLocalizer.load(engine_dir)
        print(f"[engine] Loaded: {engine_dir}")
    else:
        localizer = build_engine(engine_dir)
    print(f"[engine] {localizer}")
    return localizer


# =============================================================================
# Stream of held-out measurements
# =============================================================================

def held_out_stream(localizer, scenario_ids, seed):
    """Yield (sample_id, raw complex CSI, true position) for the test split.

    The pooled dataset concatenates scenarios in order, each holding
    NUM_USERS x NUM_SAMPLES measurements, so a pooled index maps to the
    measurement id ``scenario*10000 + user*1000 + sample``. The test indices
    are the same ones LocalizationUQ.split_dataset() uses, so none of the
    streamed samples was used for calibration (for the default pooled
    engine; a custom engine should be streamed with its own scenarios/seed).
    """
    from trustworthy_uq import LocalizationUQ

    per_scenario = NUM_USERS * NUM_SAMPLES
    n_total = per_scenario * len(scenario_ids)
    splitter = LocalizationUQ(seed=seed, split=tuple(
        localizer.metadata.get("split", (0.25, 0.45, 0.30))))
    _, _, test = splitter.split_dataset(range(n_total))

    user_positions = np.load(DATASET_PATH / "user_positions.npy")[:, :2]

    for pooled_index in test.indices:
        scenario = scenario_ids[pooled_index // per_scenario]
        user, sample = divmod(pooled_index % per_scenario, NUM_SAMPLES)
        sample_id = scenario * 10000 + user * 1000 + sample
        csi = np.load(
            DATASET_PATH / "samples"
            / f"channel_measurement_{sample_id:06d}.npy")
        yield sample_id, csi, user_positions[user].astype(np.float32)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--engine", type=Path, default=ENGINE_DIR,
                        help="engine directory (built if missing)")
    parser.add_argument("--sla", type=float, default=0.95,
                        help="SLA level used for the accuracy requirement")
    parser.add_argument("--max-radius-mm", type=float, default=25.0,
                        help="required accuracy: accept a fix only if its "
                             "trust radius at --sla is at most this")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="stop after this many measurements")
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--output", type=Path, default=None,
                        help="optional JSON-lines file with every result")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. Extract data.rar first.")

    localizer = load_or_build_engine(args.engine)
    levels = localizer.sla_levels
    scenario_ids = localizer.metadata.get("scenario_ids", SCENARIO_IDS)
    seed = localizer.metadata.get("seed", SEED)

    sink = open(args.output, "w", encoding="utf-8") if args.output else None
    n = 0
    covered = {level: 0 for level in levels}
    accepted = accepted_covered = 0
    latencies = []

    print(f"\n[stream] requirement: radius <= {args.max_radius_mm} mm "
          f"at SLA {args.sla}")
    start = time.perf_counter()
    try:
        for sample_id, csi, true_xy in held_out_stream(
                localizer, scenario_ids, seed):

            result = localizer.locate(csi, y_true=true_xy)

            n += 1
            latencies.append(result.latency_ms)
            for level in levels:
                covered[level] += result.covered[level]
            ok = result.meets(args.max_radius_mm, sla=args.sla)
            accepted += ok
            accepted_covered += ok and result.error_mm <= result.radius(args.sla)

            if sink:
                sink.write(json.dumps(
                    {"sample_id": int(sample_id),
                     "accepted": bool(ok), **result.to_dict()}) + "\n")

            if n <= 3 or n % args.print_every == 0:
                x, y = result.position_mm
                print(f"  #{n:5d} id={sample_id:06d} pos=({x:9.1f}, {y:9.1f}) mm"
                      f"  r@{args.sla}={result.radius(args.sla):6.1f} mm"
                      f"  err={result.error_mm:6.1f} mm"
                      f"  {'ACCEPT' if ok else 'reject'}")

            if args.num_samples and n >= args.num_samples:
                break
    finally:
        if sink:
            sink.close()

    elapsed = time.perf_counter() - start
    print(f"\n[summary] {n} measurements in {elapsed:.1f} s "
          f"(median locate latency {np.median(latencies):.1f} ms)")
    for level in levels:
        print(f"  SLA {level:.2f}: empirical coverage {covered[level] / n:.4f}")
    print(f"  Accepted (r@{args.sla} <= {args.max_radius_mm} mm): "
          f"{accepted}/{n} = {accepted / n:.3f}")
    if accepted:
        print(f"  Coverage among accepted fixes: "
              f"{accepted_covered / accepted:.4f}")
    if sink:
        print(f"  Results written to {args.output}")


if __name__ == "__main__":
    main()
