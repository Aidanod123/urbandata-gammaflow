"""
Evaluate ARAD-LSTM on preprocessed runs.

This script:
- Loads a trained ARAD-LSTM model
- Calibrates threshold on a background run (or specified run)
- Evaluates alarms/hour and score stats on evaluation runs

Requires preprocessed run*.pt files (see preprocess_radai_runs.py).
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

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
    real_times = np.asarray(data["real_times"], dtype=np.float64)
    live_times = data.get("live_times")
    if live_times is None:
        live_times = real_times
    else:
        # Older preprocessed sets may store live_times as object arrays of None.
        # In that case, fall back to real_times to avoid NaN propagation.
        if getattr(live_times, "dtype", None) == object:
            live_times = real_times
        else:
            live_times = np.asarray(live_times, dtype=np.float64)
            if np.any(~np.isfinite(live_times)) or np.any(live_times <= 0):
                live_times = real_times

    if np.any(~np.isfinite(real_times)) or np.any(real_times <= 0):
        real_times = np.ones_like(real_times, dtype=np.float64)

    if np.any(~np.isfinite(live_times)) or np.any(live_times <= 0):
        live_times = real_times

    energy_edges = data["energy_edges"]
    count_rates = data.get("count_rates")

    if count_rates is not None:
        gross_count_rates = np.asarray(count_rates, dtype=np.float64)
        counts = spectra * gross_count_rates[:, None] * live_times[:, None]
    else:
        counts = spectra * live_times[:, None]

    return SpectralTimeSeries.from_array(
        counts=counts,
        energy_edges=energy_edges,
        timestamps=data.get("timestamps"),
        live_times=live_times,
        real_times=real_times,
    )


def get_run_id(run_file: Path) -> str:
    stem = run_file.stem
    return stem.replace("run", "") if stem.startswith("run") else stem


def get_acquisition_times(time_series: SpectralTimeSeries) -> np.ndarray:
    times = time_series.live_times
    if times is None or getattr(times, "dtype", None) == object:
        times = time_series.real_times
    else:
        times = np.asarray(times, dtype=np.float64)
        if np.any(~np.isfinite(times)) or np.any(times <= 0):
            times = time_series.real_times
    return np.asarray(times, dtype=np.float64)


def get_first_valid_index(detector: ARADLSTMDetector) -> int:
    return detector.sequence_length if detector.target_mode == "next" else (detector.sequence_length - 1)


def summarize_alarm_durations(alarms: List[Dict[str, float]]) -> Dict[str, float]:
    if not alarms:
        return {
            "total_alarm_seconds": 0.0,
            "mean_alarm_seconds": 0.0,
            "max_alarm_seconds": 0.0,
        }

    durations = np.array([
        max(0.0, float(alarm["end_time"]) - float(alarm["start_time"]))
        for alarm in alarms
    ], dtype=np.float64)
    return {
        "total_alarm_seconds": float(durations.sum()),
        "mean_alarm_seconds": float(durations.mean()),
        "max_alarm_seconds": float(durations.max()),
    }


def format_metric(value: float, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_csv_rows(output_path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_scored_window_bounds(
    timestamps: np.ndarray,
    acquisition_times: np.ndarray,
    first_valid_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(timestamps) <= first_valid_idx or len(acquisition_times) <= first_valid_idx:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    valid_timestamps = np.asarray(timestamps[first_valid_idx:], dtype=np.float64)
    valid_times = np.asarray(acquisition_times[first_valid_idx:], dtype=np.float64)
    if valid_timestamps.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    starts = valid_timestamps - 0.5 * valid_times
    ends = valid_timestamps + 0.5 * valid_times
    return starts, ends


def compute_unique_coverage_seconds(starts: np.ndarray, ends: np.ndarray) -> float:
    if starts.size == 0 or ends.size == 0:
        return 0.0

    total = 0.0
    current_start = float(starts[0])
    current_end = float(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        start = float(start)
        end = float(end)
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(0.0, current_end - current_start)
            current_start = start
            current_end = end

    total += max(0.0, current_end - current_start)
    return total


def calibrate_threshold_from_runs(
    detector: ARADLSTMDetector,
    calibration_files: List[Path],
    alarms_per_hour: float,
    batch_size: int,
    max_iterations: int = 20,
) -> tuple[float, float, int]:
    first_valid_idx = get_first_valid_index(detector)
    original_threshold = detector.threshold
    calibration_records = []
    total_hours = 0.0
    all_valid_scores = []

    detector.threshold = float("inf")
    for run_file in calibration_files:
        ts = load_time_series(run_file)
        scores, _ = detector.detect(ts, batch_size=batch_size)
        acquisition_times = get_acquisition_times(ts)
        starts, ends = make_scored_window_bounds(ts.timestamps, acquisition_times, first_valid_idx)
        run_hours = float(compute_unique_coverage_seconds(starts, ends) / 3600.0)
        valid_scores = np.asarray(scores[first_valid_idx:], dtype=np.float64)
        if valid_scores.size == 0 or run_hours <= 0:
            continue

        calibration_records.append({
            "scores": scores,
            "timestamps": np.asarray(ts.timestamps, dtype=np.float64),
        })
        all_valid_scores.append(valid_scores)
        total_hours += run_hours

    if not calibration_records or total_hours <= 0:
        detector.threshold = original_threshold
        raise ValueError("Calibration data produced no valid scored windows.")

    combined_scores = np.concatenate(all_valid_scores)
    low_threshold = float(np.min(combined_scores))
    high_threshold = float(np.max(combined_scores) * 1.5)
    initial_percentile = detector._estimate_initial_threshold_percentile(
        n_scores=len(combined_scores),
        total_time_hours=total_hours,
        alarms_per_hour=alarms_per_hour,
    )

    best_threshold = float(np.percentile(combined_scores, initial_percentile))
    best_far_diff = float("inf")
    best_observed_far = 0.0
    best_alarm_count = 0

    for _ in range(max_iterations):
        test_threshold = (low_threshold + high_threshold) / 2.0
        detector.threshold = test_threshold
        total_alarms = 0
        for record in calibration_records:
            total_alarms += len(detector._detect_alarms(record["scores"], record["timestamps"]))

        observed_far = total_alarms / total_hours
        far_diff = abs(observed_far - alarms_per_hour)

        is_better = False
        if far_diff < best_far_diff:
            is_better = True
        elif far_diff == best_far_diff:
            if observed_far > best_observed_far:
                is_better = True
            elif observed_far == best_observed_far and test_threshold < best_threshold:
                is_better = True

        if is_better:
            best_far_diff = far_diff
            best_threshold = test_threshold
            best_observed_far = observed_far
            best_alarm_count = total_alarms

        if observed_far > alarms_per_hour:
            low_threshold = test_threshold
        else:
            high_threshold = test_threshold

        if far_diff < max(0.1 * alarms_per_hour, 1e-6) or (high_threshold - low_threshold) < 1e-8:
            break

    detector.threshold = best_threshold
    detector.alarms = []
    return best_threshold, best_observed_far, best_alarm_count


def select_calibration_files(
    calibration_files: List[Path],
    calibration_dir: Path,
    preprocessed_dir: Path,
    calibration_run_id: Optional[int],
) -> tuple[Path, List[Path]]:
    if calibration_run_id is not None:
        calib_file = calibration_dir / f"run{calibration_run_id}.pt"
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration run not found: {calib_file}")
        return calib_file, [calib_file]

    if calibration_dir == preprocessed_dir:
        # Keep the default safe: use one in-directory run for calibration so the
        # remaining runs are still available for evaluation.
        calib_file = calibration_files[0]
        return calib_file, [calib_file]

    calib_file = calibration_files[0]
    return calib_file, calibration_files


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARAD-LSTM on preprocessed runs")
    parser.add_argument("--preprocessed-dir", type=str, required=True, help="Directory with run*.pt files")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument(
        "--calibration-preprocessed-dir",
        type=str,
        default=None,
        help="Directory with background run*.pt files for threshold calibration (defaults to preprocessed-dir)",
    )
    parser.add_argument("--calibration-run-id", type=int, default=None, help="Run ID for threshold calibration")
    parser.add_argument("--alarms-per-hour", type=float, default=0.5, help="Target FAR for calibration")
    parser.add_argument("--max-eval-runs", type=int, default=None, help="Max runs to evaluate")
    parser.add_argument("--max-calibration-runs", type=int, default=None, help="Optional cap on background runs used for calibration")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for batched sequence scoring")
    parser.add_argument("--csv-output", type=str, default=None, help="Optional path to write per-run results as CSV")
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    calibration_dir = Path(args.calibration_preprocessed_dir) if args.calibration_preprocessed_dir else preprocessed_dir

    run_files = list_runs(preprocessed_dir)
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files in {preprocessed_dir}")

    calibration_files = list_runs(calibration_dir)
    if not calibration_files:
        raise FileNotFoundError(f"No calibration run*.pt files in {calibration_dir}")

    detector = ARADLSTMDetector()
    detector.load(args.model_path)

    calib_file, calibration_eval_files = select_calibration_files(
        calibration_files=calibration_files,
        calibration_dir=calibration_dir,
        preprocessed_dir=preprocessed_dir,
        calibration_run_id=args.calibration_run_id,
    )

    if args.max_calibration_runs is not None:
        calibration_eval_files = calibration_eval_files[: args.max_calibration_runs]
        if not calibration_eval_files:
            raise ValueError("--max-calibration-runs removed all calibration runs.")
        calib_file = calibration_eval_files[0]

    _, calibration_far, calibration_alarm_count = calibrate_threshold_from_runs(
        detector,
        calibration_eval_files,
        alarms_per_hour=args.alarms_per_hour,
        batch_size=args.batch_size,
    )

    # Evaluation runs — exclude all calibration files to prevent data leakage
    calibration_file_set = set(calibration_eval_files)
    if calibration_dir == preprocessed_dir:
        eval_files = [p for p in run_files if p not in calibration_file_set]
    else:
        eval_files = run_files
    if args.max_eval_runs is not None:
        eval_files = eval_files[: args.max_eval_runs]
    if not eval_files:
        raise ValueError(
            "No evaluation runs remain after excluding calibration data. "
            "Use --calibration-preprocessed-dir to supply separate background runs, "
            "or pass --calibration-run-id to reserve only one in-directory run for calibration."
        )

    far_values = []
    max_scores = []
    mean_scores = []
    score_stds = []
    total_alarms = 0
    total_hours = 0.0
    all_scores = []
    per_run_rows = []
    first_valid_idx = get_first_valid_index(detector)

    for run_file in eval_files:
        ts = load_time_series(run_file)
        scores, alarms = detector.detect(ts, batch_size=args.batch_size)
        valid_scores = scores[first_valid_idx:] if len(scores) > first_valid_idx else np.array([], dtype=np.float32)
        acquisition_times = get_acquisition_times(ts)
        starts, ends = make_scored_window_bounds(ts.timestamps, acquisition_times, first_valid_idx)
        run_hours = float(compute_unique_coverage_seconds(starts, ends) / 3600.0)

        if valid_scores.size > 0:
            max_scores.append(float(np.max(valid_scores)))
            mean_scores.append(float(np.mean(valid_scores)))
            score_stds.append(float(np.std(valid_scores)))
            all_scores.append(valid_scores)
            score_p95 = float(np.percentile(valid_scores, 95))
            score_p99 = float(np.percentile(valid_scores, 99))
        else:
            max_scores.append(float("nan"))
            mean_scores.append(float("nan"))
            score_stds.append(float("nan"))
            score_p95 = float("nan")
            score_p99 = float("nan")

        total_hours += run_hours
        total_alarms += len(alarms)
        far = len(alarms) / run_hours if run_hours > 0 else 0.0
        far_values.append(far)

        alarm_duration_stats = summarize_alarm_durations(alarms)
        per_run_rows.append({
            "run_id": get_run_id(run_file),
            "scored_hours": run_hours,
            "alarms": len(alarms),
            "far": far,
            "score_mean": mean_scores[-1],
            "score_std": score_stds[-1],
            "score_max": max_scores[-1],
            "score_p95": score_p95,
            "score_p99": score_p99,
            **alarm_duration_stats,
        })

    combined_scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)

    if args.csv_output and per_run_rows:
        csv_rows = []
        for row in per_run_rows:
            csv_rows.append({
                **row,
                "threshold": float(detector.threshold),
                "calibration_run": calib_file.name,
                "warmup_spectra_excluded": int(first_valid_idx),
            })
        write_csv_rows(Path(args.csv_output), csv_rows)

    print("Calibration dir:", calibration_dir)
    print("Calibration run:", calib_file.name)
    print(f"Threshold: {detector.threshold:.6f}")
    print(f"Calibration runs used: {len(calibration_eval_files)}")
    print(f"Calibration FAR: {calibration_far:.2f} alarms/hour ({calibration_alarm_count} alarms)")
    print(f"Eval runs: {len(eval_files)}")
    print(f"Warm-up excluded per run: first {first_valid_idx} spectra")
    print(f"Total scored hours: {total_hours:.2f}")
    print(f"Total alarms: {total_alarms}")
    if total_hours > 0:
        print(f"Overall FAR: {total_alarms / total_hours:.2f} alarms/hour")
    if far_values:
        print(f"Mean FAR per run: {np.mean(far_values):.2f}")
        print(f"Median FAR per run: {np.median(far_values):.2f}")
    if mean_scores:
        print(f"Score mean (avg run mean): {format_metric(float(np.nanmean(mean_scores)), 6)}")
    if score_stds:
        print(f"Score std (avg run std): {format_metric(float(np.nanmean(score_stds)), 6)}")
    if max_scores:
        print(f"Score max (avg run max): {format_metric(float(np.nanmean(max_scores)), 6)}")

    if combined_scores.size > 0:
        print("Score quantiles across all scored spectra:")
        print(f"  P50: {np.percentile(combined_scores, 50):.6f}")
        print(f"  P90: {np.percentile(combined_scores, 90):.6f}")
        print(f"  P95: {np.percentile(combined_scores, 95):.6f}")
        print(f"  P99: {np.percentile(combined_scores, 99):.6f}")

    if per_run_rows:
        if args.csv_output:
            print(f"CSV output: {args.csv_output}")
        mean_alarm_duration = np.mean([row["mean_alarm_seconds"] for row in per_run_rows])
        max_alarm_duration = np.max([row["max_alarm_seconds"] for row in per_run_rows])
        total_alarm_seconds = np.sum([row["total_alarm_seconds"] for row in per_run_rows])
        print("Alarm duration summary:")
        print(f"  Total alarm seconds: {total_alarm_seconds:.2f}")
        print(f"  Mean alarm duration per run: {mean_alarm_duration:.2f} s")
        print(f"  Max alarm duration observed: {max_alarm_duration:.2f} s")

        print("\nPer-run summary:")
        header = (
            f"{'Run':>6} {'Hours':>8} {'Alarms':>8} {'FAR/hr':>8} "
            f"{'Mean':>10} {'Std':>10} {'P95':>10} {'P99':>10} {'Max':>10} {'Alarm_s':>10}"
        )
        print(header)
        print("-" * len(header))
        for row in per_run_rows:
            print(
                f"{row['run_id']:>6} "
                f"{row['scored_hours']:>8.2f} "
                f"{row['alarms']:>8d} "
                f"{row['far']:>8.2f} "
                f"{format_metric(row['score_mean'], 6):>10} "
                f"{format_metric(row['score_std'], 6):>10} "
                f"{format_metric(row['score_p95'], 6):>10} "
                f"{format_metric(row['score_p99'], 6):>10} "
                f"{format_metric(row['score_max'], 6):>10} "
                f"{row['total_alarm_seconds']:>10.2f}"
            )


if __name__ == "__main__":
    main()
