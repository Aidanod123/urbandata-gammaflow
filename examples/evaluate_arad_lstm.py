"""
Evaluate ARAD-LSTM on preprocessed runs.

This script:
- Loads a trained ARAD-LSTM model
- Calibrates threshold on a background run (or specified run)
- Evaluates alarms/hour and score stats on evaluation runs

Requires preprocessed run*.pt files (see preprocess_radai_runs.py).
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

# Add src to path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gammaflow.core.time_series import SpectralTimeSeries
from src.detectors.arad_lstm import ARADLSTMDetector


def list_runs(preprocessed_dir: Path) -> List[Path]:
    return sorted(preprocessed_dir.glob("run*.pt"))


def load_time_series(run_file: Path) -> SpectralTimeSeries:
    data = torch.load(run_file, map_location="cpu", weights_only=False)
    spectra = data["spectra"].numpy()
    real_times = data["real_times"]
    energy_edges = data["energy_edges"]

    counts = spectra * real_times[:, None]

    return SpectralTimeSeries(
        counts=counts,
        edges=energy_edges,
        real_times=real_times,
        timestamps=data.get("timestamps"),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARAD-LSTM on preprocessed runs")
    parser.add_argument("--preprocessed-dir", type=str, required=True, help="Directory with run*.pt files")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--calibration-run-id", type=int, default=None, help="Run ID for threshold calibration")
    parser.add_argument("--alarms-per-hour", type=float, default=0.5, help="Target FAR for calibration")
    parser.add_argument("--max-eval-runs", type=int, default=None, help="Max runs to evaluate")
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    run_files = list_runs(preprocessed_dir)
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files in {preprocessed_dir}")

    detector = ARADLSTMDetector()
    detector.load(args.model_path)

    # Select calibration run
    if args.calibration_run_id is not None:
        calib_file = preprocessed_dir / f"run{args.calibration_run_id}.pt"
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration run not found: {calib_file}")
    else:
        calib_file = run_files[0]

    calib_ts = load_time_series(calib_file)
    detector.set_threshold_by_far(calib_ts, alarms_per_hour=args.alarms_per_hour)

    # Evaluation runs (exclude calibration run)
    eval_files = [p for p in run_files if p != calib_file]
    if args.max_eval_runs is not None:
        eval_files = eval_files[: args.max_eval_runs]

    far_values = []
    max_scores = []
    mean_scores = []
    total_alarms = 0
    total_hours = 0.0

    for run_file in eval_files:
        ts = load_time_series(run_file)
        scores, alarms = detector.detect(ts)
        max_scores.append(float(np.max(scores)))
        mean_scores.append(float(np.mean(scores)))

        run_hours = np.sum(ts.real_times) / 3600.0
        total_hours += run_hours
        total_alarms += len(alarms)
        far = len(alarms) / run_hours if run_hours > 0 else 0.0
        far_values.append(far)

    print("Calibration run:", calib_file.name)
    print(f"Threshold: {detector.threshold:.6f}")
    print(f"Eval runs: {len(eval_files)}")
    print(f"Total hours: {total_hours:.2f}")
    print(f"Total alarms: {total_alarms}")
    if total_hours > 0:
        print(f"Overall FAR: {total_alarms / total_hours:.2f} alarms/hour")
    print(f"Mean FAR per run: {np.mean(far_values):.2f}")
    print(f"Score mean (avg): {np.mean(mean_scores):.6f}")
    print(f"Score max (avg): {np.mean(max_scores):.6f}")


if __name__ == "__main__":
    main()
