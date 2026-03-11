"""
Evaluate ARAD-LSTM on preprocessed runs with source-aware truth from RADAI HDF5.

This script:
- Loads a trained ARAD-LSTM model
- Calibrates threshold on a chosen calibration run (ideally background-only)
- Evaluates preprocessed runs against source truth from RADAI HDF5 `sources`
- Reports overlap-based detection metrics and per-run summaries

Typical usage:
    python examples/evaluate_arad_lstm_sources.py \
        --preprocessed-dir TESTING/with-sources-1.0-0.5 \
        --truth-h5-path RADAI-dataset/developer_v4.3.h5 \
        --calibration-preprocessed-dir per-spectrum-norm/no-sources-1.0-0.5 \
        --model-path models/softmax_10_chi2_1.0-.5_seq10.pt
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
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


def get_run_key(run_file: Path) -> str:
    return f"run{get_run_id(run_file)}"


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


def load_truth_map(truth_h5_path: Path, source_time_scale: float) -> Dict[str, Dict[str, object]]:
    truth_map: Dict[str, Dict[str, object]] = {}
    with h5py.File(truth_h5_path, "r") as f:
        if "runs" not in f:
            raise ValueError(f"Truth HDF5 has no 'runs' group: {truth_h5_path}")

        for run_key, run_group in f["runs"].items():
            if "sources" not in run_group:
                truth_map[run_key] = {
                    "source_present": False,
                    "source_ids": [],
                    "source_times": [],
                    "distance": [],
                    "shielding": [],
                }
                continue

            source_group = run_group["sources"]
            source_ids = np.asarray(source_group.get("id", []))
            source_times = np.asarray(source_group.get("time", []), dtype=np.float64) * source_time_scale
            distances = np.asarray(source_group.get("distance", []), dtype=np.float64)
            shielding = np.asarray(source_group.get("shielding", []), dtype=np.int64)

            truth_map[run_key] = {
                "source_present": bool(source_ids.size > 0),
                "source_ids": source_ids.astype(int).tolist(),
                "source_times": source_times.tolist(),
                "distance": distances.tolist(),
                "shielding": shielding.tolist(),
            }

    return truth_map


def build_source_windows(source_times: List[float], before: float, after: float) -> List[Dict[str, float]]:
    windows = []
    for source_time in source_times:
        windows.append({
            "start": max(0.0, float(source_time) - before),
            "end": float(source_time) + after,
            "time": float(source_time),
        })
    return windows


def overlaps(window_start: float, window_end: float, alarm_start: float, alarm_end: float) -> bool:
    return max(window_start, alarm_start) <= min(window_end, alarm_end)


def classify_alarms_from_target_windows(
    alarms: List[Dict[str, float]],
    target_timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    tp_alarms: List[Dict[str, float]] = []
    fp_alarms: List[Dict[str, float]] = []
    tp_detection_times: List[float] = []

    for alarm in alarms:
        alarm_start = float(alarm["start_time"])
        alarm_end = float(alarm["end_time"])
        in_alarm = (
            (target_timestamps >= alarm_start)
            & (target_timestamps <= alarm_end)
            & y_pred
        )

        if np.any(in_alarm & y_true):
            tp_alarms.append(alarm)
            tp_detection_times.extend(target_timestamps[in_alarm & y_true].astype(float).tolist())
        else:
            fp_alarms.append(alarm)

    return {
        "tp_alarms": tp_alarms,
        "fp_alarms": fp_alarms,
        "detected": bool(tp_alarms),
        "tp_detection_times": tp_detection_times,
    }


def first_detection_delay(tp_detection_times: List[float], source_times: List[float]) -> float:
    if not tp_detection_times or not source_times:
        return float("nan")
    first_detection_time = min(float(value) for value in tp_detection_times)
    matched_source_time = min(
        (float(value) for value in source_times),
        key=lambda value: abs(first_detection_time - value),
    )
    return first_detection_time - matched_source_time


def score_peak_in_windows(
    scores: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    source_windows: List[Dict[str, float]],
    first_valid_idx: int,
) -> float:
    if len(scores) <= first_valid_idx or not source_windows:
        return float("nan")

    valid_scores = scores[first_valid_idx:]
    if valid_scores.size == 0 or window_starts.size == 0:
        return float("nan")

    mask = np.zeros(valid_scores.shape[0], dtype=bool)
    for window in source_windows:
        mask |= (window_starts <= window["end"]) & (window_ends >= window["start"])

    if not np.any(mask):
        return float("nan")
    return float(np.max(valid_scores[mask]))


def make_scored_window_bounds(
    timestamps: np.ndarray,
    acquisition_times: np.ndarray,
    first_valid_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(timestamps) <= first_valid_idx or len(acquisition_times) <= first_valid_idx:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    valid_timestamps = np.asarray(timestamps[first_valid_idx:], dtype=np.float64)
    valid_times = np.asarray(acquisition_times[first_valid_idx:], dtype=np.float64)
    if valid_timestamps.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    starts = valid_timestamps - 0.5 * valid_times
    ends = valid_timestamps + 0.5 * valid_times
    return starts, ends


def make_sequence_span_bounds(
    timestamps: np.ndarray,
    acquisition_times: np.ndarray,
    detector: ARADLSTMDetector,
    first_valid_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(timestamps) <= first_valid_idx or len(acquisition_times) <= first_valid_idx:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    all_timestamps = np.asarray(timestamps, dtype=np.float64)
    all_times = np.asarray(acquisition_times, dtype=np.float64)
    score_indices = np.arange(first_valid_idx, len(all_timestamps), dtype=np.int64)
    if score_indices.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    if detector.target_mode == "next":
        start_indices = score_indices - detector.sequence_length
    else:
        start_indices = score_indices - detector.sequence_length + 1

    starts = all_timestamps[start_indices] - 0.5 * all_times[start_indices]
    ends = all_timestamps[score_indices] + 0.5 * all_times[score_indices]
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
) -> Tuple[float, float, int]:
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
    truth_map: Dict[str, Dict[str, object]],
) -> Tuple[Path, List[Path], Optional[bool]]:
    calibration_run_has_source: Optional[bool] = None

    if calibration_run_id is not None:
        calib_file = calibration_dir / f"run{calibration_run_id}.pt"
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration run not found: {calib_file}")
        if calibration_dir == preprocessed_dir:
            calibration_run_has_source = bool(truth_map.get(get_run_key(calib_file), {}).get("source_present", False))
        return calib_file, [calib_file], calibration_run_has_source

    if calibration_dir == preprocessed_dir:
        background_calibration_files = [
            path for path in calibration_files
            if not bool(truth_map.get(get_run_key(path), {}).get("source_present", False))
        ]
        if background_calibration_files:
            calib_file = background_calibration_files[0]
            return calib_file, [calib_file], False

        calib_file = calibration_files[0]
        calibration_run_has_source = bool(truth_map.get(get_run_key(calib_file), {}).get("source_present", False))
        return calib_file, [calib_file], calibration_run_has_source

    calib_file = calibration_files[0]
    return calib_file, calibration_files, calibration_run_has_source


def compute_window_truth_and_preds(
    scores: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    source_times: List[float],
    first_valid_idx: int,
    threshold: float,
    source_window_before: float,
    source_window_after: float,
) -> Dict[str, object]:
    if len(scores) <= first_valid_idx or starts.size == 0 or ends.size == 0:
        empty = np.empty(0, dtype=bool)
        return {
            "y_true": empty,
            "y_pred": empty,
            "n_windows": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    valid_scores = np.asarray(scores[first_valid_idx:], dtype=np.float64)
    y_pred = valid_scores > threshold

    y_true = np.zeros(valid_scores.shape[0], dtype=bool)
    for source_time in source_times:
        source_start = max(0.0, float(source_time) - source_window_before)
        source_end = float(source_time) + source_window_after
        y_true |= (starts <= source_end) & (ends >= source_start)

    tp = int(np.count_nonzero(y_true & y_pred))
    fp = int(np.count_nonzero((~y_true) & y_pred))
    tn = int(np.count_nonzero((~y_true) & (~y_pred)))
    fn = int(np.count_nonzero(y_true & (~y_pred)))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "n_windows": int(valid_scores.shape[0]),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ARAD-LSTM with source-aware RADAI truth")
    parser.add_argument("--preprocessed-dir", type=str, required=True, help="Directory with evaluation run*.pt files")
    parser.add_argument("--truth-h5-path", type=str, required=True, help="RADAI HDF5 file containing sources metadata")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument(
        "--calibration-preprocessed-dir",
        type=str,
        default=None,
        help="Directory with background run*.pt files for threshold calibration (defaults to preprocessed-dir)",
    )
    parser.add_argument("--calibration-run-id", type=int, default=None, help="Run ID for threshold calibration")
    parser.add_argument("--alarms-per-hour", type=float, default=0.5, help="Target nuisance alarm rate for calibration")
    parser.add_argument("--max-eval-runs", type=int, default=None, help="Max runs to evaluate")
    parser.add_argument("--max-calibration-runs", type=int, default=None, help="Optional cap on background runs used for calibration")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for batched sequence scoring")
    parser.add_argument("--source-window-before", type=float, default=1.0, help="Seconds before source time to count overlap")
    parser.add_argument("--source-window-after", type=float, default=1.0, help="Seconds after source time to count overlap")
    parser.add_argument(
        "--source-time-scale",
        type=float,
        default=1e-3,
        help="Scale factor to convert RADAI sources/time to seconds (default: 1e-3 for ms -> s)",
    )
    parser.add_argument("--csv-output", type=str, default=None, help="Optional path to write per-run results as CSV")
    args = parser.parse_args()

    preprocessed_dir = Path(args.preprocessed_dir)
    truth_h5_path = Path(args.truth_h5_path)
    calibration_dir = Path(args.calibration_preprocessed_dir) if args.calibration_preprocessed_dir else preprocessed_dir

    run_files = list_runs(preprocessed_dir)
    if not run_files:
        raise FileNotFoundError(f"No run*.pt files in {preprocessed_dir}")
    if not truth_h5_path.exists():
        raise FileNotFoundError(f"Truth HDF5 not found: {truth_h5_path}")

    calibration_files = list_runs(calibration_dir)
    if not calibration_files:
        raise FileNotFoundError(f"No calibration run*.pt files in {calibration_dir}")

    truth_map = load_truth_map(truth_h5_path, source_time_scale=args.source_time_scale)

    detector = ARADLSTMDetector()
    detector.load(args.model_path)
    first_valid_idx = get_first_valid_index(detector)

    calib_file, calibration_eval_files, calibration_run_has_source = select_calibration_files(
        calibration_files=calibration_files,
        calibration_dir=calibration_dir,
        preprocessed_dir=preprocessed_dir,
        calibration_run_id=args.calibration_run_id,
        truth_map=truth_map,
    )

    if args.max_calibration_runs is not None:
        calibration_eval_files = calibration_eval_files[: args.max_calibration_runs]
        if not calibration_eval_files:
            raise ValueError("--max-calibration-runs removed all calibration runs.")
        calib_file = calibration_eval_files[0]
        if calibration_dir == preprocessed_dir:
            calibration_run_has_source = bool(truth_map.get(get_run_key(calib_file), {}).get("source_present", False))

    _, calibration_far, calibration_alarm_count = calibrate_threshold_from_runs(
        detector,
        calibration_eval_files,
        alarms_per_hour=args.alarms_per_hour,
        batch_size=args.batch_size,
    )

    # Exclude ALL calibration files to prevent data leakage
    calibration_file_set = set(calibration_eval_files)
    eval_files = run_files
    if calibration_dir == preprocessed_dir:
        eval_files = [path for path in eval_files if path not in calibration_file_set]
    if args.max_eval_runs is not None:
        eval_files = eval_files[: args.max_eval_runs]
    if not eval_files:
        raise ValueError(
            "No evaluation runs remain after excluding calibration data. "
            "Use --calibration-preprocessed-dir to supply separate background runs, "
            "or pass --calibration-run-id to reserve only one in-directory run for calibration."
        )

    source_run_count = 0
    background_run_count = 0
    detected_run_count = 0
    missed_run_count = 0
    background_alarm_run_count = 0
    tp_alarm_count = 0
    fp_alarm_count = 0
    target_window_tp = 0
    target_window_fp = 0
    target_window_tn = 0
    target_window_fn = 0
    sequence_window_tp = 0
    sequence_window_fp = 0
    sequence_window_tn = 0
    sequence_window_fn = 0
    total_scored_windows = 0
    total_hours = 0.0
    detection_delays = []
    per_run_rows = []
    source_id_stats: Dict[int, Dict[str, int]] = {}

    for run_file in eval_files:
        run_key = get_run_key(run_file)
        truth = truth_map.get(run_key)
        if truth is None:
            continue

        ts = load_time_series(run_file)
        scores, alarms = detector.detect(ts, batch_size=args.batch_size)
        acquisition_times = get_acquisition_times(ts)
        target_starts, target_ends = make_scored_window_bounds(ts.timestamps, acquisition_times, first_valid_idx)
        sequence_starts, sequence_ends = make_sequence_span_bounds(ts.timestamps, acquisition_times, detector, first_valid_idx)
        run_hours = float(compute_unique_coverage_seconds(target_starts, target_ends) / 3600.0)
        total_hours += run_hours

        source_present = bool(truth["source_present"])
        source_ids = [int(value) for value in truth["source_ids"]]
        source_times = [float(value) for value in truth["source_times"]]
        source_windows = build_source_windows(source_times, args.source_window_before, args.source_window_after)
        target_window_stats = compute_window_truth_and_preds(
            scores=scores,
            starts=target_starts,
            ends=target_ends,
            source_times=source_times,
            first_valid_idx=first_valid_idx,
            threshold=detector.threshold,
            source_window_before=args.source_window_before,
            source_window_after=args.source_window_after,
        )
        valid_target_timestamps = np.asarray(ts.timestamps[first_valid_idx:], dtype=np.float64)
        alarm_classes = classify_alarms_from_target_windows(
            alarms=alarms,
            target_timestamps=valid_target_timestamps,
            y_true=np.asarray(target_window_stats["y_true"], dtype=bool),
            y_pred=np.asarray(target_window_stats["y_pred"], dtype=bool),
        )
        tp_alarms = alarm_classes["tp_alarms"]
        fp_alarms = alarm_classes["fp_alarms"]
        tp_detection_times = alarm_classes["tp_detection_times"]
        detected = bool(alarm_classes["detected"])

        if source_present:
            source_run_count += 1
            if detected:
                detected_run_count += 1
                delay = first_detection_delay(tp_detection_times, source_times)
                if np.isfinite(delay):
                    detection_delays.append(delay)
            else:
                missed_run_count += 1
            for source_id, source_time in zip(source_ids, source_times):
                source_specific_stats = compute_window_truth_and_preds(
                    scores=scores,
                    starts=target_starts,
                    ends=target_ends,
                    source_times=[source_time],
                    first_valid_idx=first_valid_idx,
                    threshold=detector.threshold,
                    source_window_before=args.source_window_before,
                    source_window_after=args.source_window_after,
                )
                source_detected = bool(int(source_specific_stats["tp"]) > 0)
                stats = source_id_stats.setdefault(source_id, {"runs": 0, "detected": 0})
                stats["runs"] += 1
                if source_detected:
                    stats["detected"] += 1
        else:
            background_run_count += 1
            if alarms:
                background_alarm_run_count += 1

        tp_alarm_count += len(tp_alarms)
        fp_alarm_count += len(fp_alarms)

        sequence_window_stats = compute_window_truth_and_preds(
            scores=scores,
            starts=sequence_starts,
            ends=sequence_ends,
            source_times=source_times,
            first_valid_idx=first_valid_idx,
            threshold=detector.threshold,
            source_window_before=args.source_window_before,
            source_window_after=args.source_window_after,
        )
        target_window_tp += int(target_window_stats["tp"])
        target_window_fp += int(target_window_stats["fp"])
        target_window_tn += int(target_window_stats["tn"])
        target_window_fn += int(target_window_stats["fn"])
        sequence_window_tp += int(sequence_window_stats["tp"])
        sequence_window_fp += int(sequence_window_stats["fp"])
        sequence_window_tn += int(sequence_window_stats["tn"])
        sequence_window_fn += int(sequence_window_stats["fn"])
        total_scored_windows += int(target_window_stats["n_windows"])

        valid_scores = scores[first_valid_idx:] if len(scores) > first_valid_idx else np.array([], dtype=np.float32)
        score_mean = float(np.mean(valid_scores)) if valid_scores.size > 0 else float("nan")
        score_max = float(np.max(valid_scores)) if valid_scores.size > 0 else float("nan")
        score_p95 = float(np.percentile(valid_scores, 95)) if valid_scores.size > 0 else float("nan")
        peak_score_in_source_window = score_peak_in_windows(scores, target_starts, target_ends, source_windows, first_valid_idx)
        alarm_duration_stats = summarize_alarm_durations(alarms)

        per_run_rows.append({
            "run_id": get_run_id(run_file),
            "source_present": source_present,
            "source_id": source_ids[0] if source_ids else -1,
            "source_time": source_times[0] if source_times else float("nan"),
            "detected": detected,
            "n_alarms": len(alarms),
            "n_tp_alarms": len(tp_alarms),
            "n_fp_alarms": len(fp_alarms),
            "n_sources": len(source_times),
            "target_win_tp": int(target_window_stats["tp"]),
            "target_win_fp": int(target_window_stats["fp"]),
            "target_win_tn": int(target_window_stats["tn"]),
            "target_win_fn": int(target_window_stats["fn"]),
            "sequence_win_tp": int(sequence_window_stats["tp"]),
            "sequence_win_fp": int(sequence_window_stats["fp"]),
            "sequence_win_tn": int(sequence_window_stats["tn"]),
            "sequence_win_fn": int(sequence_window_stats["fn"]),
            "n_scored_windows": int(target_window_stats["n_windows"]),
            "score_mean": score_mean,
            "score_max": score_max,
            "score_p95": score_p95,
            "peak_score_in_source_window": peak_score_in_source_window,
            "time_to_first_detection": first_detection_delay(tp_detection_times, source_times),
            "scored_hours": run_hours,
            **alarm_duration_stats,
        })

    if args.csv_output and per_run_rows:
        csv_rows = []
        for row in per_run_rows:
            csv_rows.append({
                **row,
                "threshold": float(detector.threshold),
                "calibration_run": calib_file.name,
                "calibration_far": calibration_far,
                "warmup_spectra_excluded": int(first_valid_idx),
                "source_window_before": float(args.source_window_before),
                "source_window_after": float(args.source_window_after),
            })
        write_csv_rows(Path(args.csv_output), csv_rows)

    print("Calibration run:", calib_file.name)
    print(f"Threshold: {detector.threshold:.6f}")
    if np.isfinite(calibration_far):
        print(f"Calibration FAR: {calibration_far:.2f} alarms/hour ({calibration_alarm_count} alarms across {len(calibration_eval_files)} runs)")
    print(f"Calibration runs used: {len(calibration_eval_files)}")
    if args.csv_output:
        print(f"CSV output: {args.csv_output}")
    if calibration_run_has_source is True:
        print("WARNING: Calibration run contains source truth; use --calibration-preprocessed-dir or a background-only run for a proper nuisance threshold.")
    print(f"Eval runs: {len(per_run_rows)}")
    print(f"Warm-up excluded per run: first {first_valid_idx} spectra")
    print(f"Truth source window: [{args.source_window_before:.2f}s before, {args.source_window_after:.2f}s after source time]")
    print(f"Total scored hours: {total_hours:.2f}")
    print()
    print("Run-level summary:")
    print(f"  Source runs: {source_run_count}")
    print(f"  Background runs: {background_run_count}")
    print(f"  Detected source runs: {detected_run_count}")
    print(f"  Missed source runs: {missed_run_count}")
    if source_run_count > 0:
        print(f"  Detection rate: {detected_run_count / source_run_count:.3f}")
    if background_run_count > 0:
        print(f"  Background runs with alarms: {background_alarm_run_count}")
        print(f"  Background run false-alarm rate: {background_alarm_run_count / background_run_count:.3f}")
    print()
    print("Alarm-level summary:")
    print(f"  Total TP alarms: {tp_alarm_count}")
    print(f"  Total FP alarms: {fp_alarm_count}")
    if total_hours > 0:
        print(f"  FP alarms/hour: {fp_alarm_count / total_hours:.2f}")
    if detection_delays:
        print(f"  Median time to first detection: {np.median(detection_delays):.2f} s")
        print(f"  Mean time to first detection: {np.mean(detection_delays):.2f} s")

    def print_window_summary(title: str, tp: int, fp: int, tn: int, fn: int) -> None:
        print()
        print(title)
        print(f"  Total scored windows: {total_scored_windows}")
        print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        balanced_accuracy = (
            0.5 * (recall + specificity)
            if np.isfinite(recall) and np.isfinite(specificity)
            else float("nan")
        )
        accuracy = (tp + tn) / total_scored_windows if total_scored_windows > 0 else float("nan")
        positive_prevalence = (tp + fn) / total_scored_windows if total_scored_windows > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else float("nan")
        )
        print(f"  Positive prevalence: {format_metric(positive_prevalence, 4)}")
        print(f"  Precision: {format_metric(precision, 4)}")
        print(f"  Recall: {format_metric(recall, 4)}")
        print(f"  Specificity: {format_metric(specificity, 4)}")
        print(f"  Balanced accuracy: {format_metric(balanced_accuracy, 4)}")
        print(f"  F1: {format_metric(f1, 4)}")
        print(f"  Accuracy (TN-dominated): {format_metric(accuracy, 4)}")

    print_window_summary(
        "Window-level summary (target spectrum overlap):",
        target_window_tp,
        target_window_fp,
        target_window_tn,
        target_window_fn,
    )
    print_window_summary(
        "Window-level summary (full sequence span overlap):",
        sequence_window_tp,
        sequence_window_fp,
        sequence_window_tn,
        sequence_window_fn,
    )

    if source_id_stats:
        print()
        print("Per-source-id detection:")
        for source_id in sorted(source_id_stats):
            stats = source_id_stats[source_id]
            rate = stats["detected"] / stats["runs"] if stats["runs"] > 0 else float("nan")
            print(f"  Source {source_id:>2}: {stats['detected']}/{stats['runs']} detected ({rate:.3f})")

    if per_run_rows:
        print()
        print("Per-run summary:")
        header = (
            f"{'Run':>6} {'Src':>4} {'Det':>4} {'Alarms':>7} {'TP':>5} {'FP':>5} {'TTP':>5} {'TFP':>5} {'TFN':>5} "
            f"{'Mean':>10} {'P95':>10} {'Max':>10} {'Peak@Src':>10} {'Delay':>8}"
        )
        print(header)
        print("-" * len(header))
        for row in per_run_rows:
            print(
                f"{row['run_id']:>6} "
                f"{(row['source_id'] if row['source_present'] else 0):>4} "
                f"{('Y' if row['detected'] else 'N'):>4} "
                f"{row['n_alarms']:>7d} "
                f"{row['n_tp_alarms']:>5d} "
                f"{row['n_fp_alarms']:>5d} "
                f"{row['target_win_tp']:>5d} "
                f"{row['target_win_fp']:>5d} "
                f"{row['target_win_fn']:>5d} "
                f"{format_metric(row['score_mean'], 6):>10} "
                f"{format_metric(row['score_p95'], 6):>10} "
                f"{format_metric(row['score_max'], 6):>10} "
                f"{format_metric(row['peak_score_in_source_window'], 6):>10} "
                f"{format_metric(row['time_to_first_detection'], 2):>8}"
            )


if __name__ == "__main__":
    main()