"""
Preprocess RADAI listmode runs into per-run tensors.

This script converts listmode (dt, energy) into spectra with a fixed
integration/stride, and saves each run as a standalone tensor file.

Performance optimizations over gammaflow's from_list_mode:
- Uses np.searchsorted for O(n_windows * log(n_events)) window lookup
  instead of O(n_windows * n_events) boolean masking per window
- Fully vectorized histogram via np.bincount scatter-add:
  eliminates the Python for-loop over windows entirely (~10-50x speedup)
- Single HDF5 file handle shared across all runs

Normalization:
- per-spectrum-l1: Each spectrum normalized to sum to 1 (shape-only, no rate info)

Outputs:
- <output_dir>/runXXXX.pt with:
  - spectra: torch.FloatTensor (n_spectra, n_bins) — L1-normalized
  - count_rates: torch.FloatTensor (n_spectra,) — gross count rate (counts/s) per spectrum
  - timestamps: np.ndarray (n_spectra,)
  - live_times: np.ndarray (n_spectra,)
  - real_times: np.ndarray (n_spectra,)
  - energy_edges: np.ndarray (n_bins+1,)
  - integration_time, stride_time, energy_range, time_units
  - normalization: str ('per-spectrum-l1')

Optional:
- <output_dir>/preprocess_stats.json with summary.
"""

import argparse
import json
import time as _time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError("PyTorch is required for preprocessing. Install with: pip install torch") from exc

try:
    import h5py
except ImportError as exc:
    raise ImportError("h5py is required for preprocessing. Install with: pip install h5py") from exc

# Add src to path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.arad_lstm import ARADLSTMDetector


def resolve_output_dir(output_dir_arg: str) -> Path:
    """Resolve and create the output directory with actionable error messages."""
    output_dir = Path(output_dir_arg).expanduser()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        hint = ""
        if output_dir.is_absolute() and len(output_dir.parts) > 1 and output_dir.parts[1].startswith("per-"):
            hint = (
                "\nHint: this path is absolute and points at the filesystem root. "
                "If you meant a repo-relative folder, remove the leading '/'."
            )
        raise PermissionError(
            f"Cannot create output directory '{output_dir}'. {exc}.{hint}"
        ) from exc
    return output_dir


def get_all_run_ids(h5_path: Path) -> List[int]:
    """Get all run IDs from an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        run_keys = list(f["runs"].keys())
        run_ids = [int(k.replace("run", "")) for k in run_keys]
    return sorted(run_ids)


def _load_filtered_listmode_from_group(
    run_group: h5py.Group,
    run_key: str,
    time_units: str,
    max_events_per_run: Optional[int],
    event_stride: int,
    exclude_source_ids: bool,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load listmode dt/energy from an already-open HDF5 run group.

    Accepts an h5py Group instead of a file path so the caller can keep
    the file open across runs and avoid repeated open/close overhead.
    """
    listmode = run_group["listmode"]

    dt = np.asarray(listmode["dt"])
    energy = np.asarray(listmode["energy"])
    event_ids = np.asarray(listmode["id"]) if "id" in listmode else None

    n_events = min(len(dt), len(energy))
    if event_ids is not None:
        n_events = min(n_events, len(event_ids))

    if max_events_per_run is not None:
        n_events = min(n_events, max_events_per_run)

    dt = dt[:n_events]
    energy = energy[:n_events]
    if event_ids is not None:
        event_ids = event_ids[:n_events]

    if event_stride > 1:
        indices = slice(0, n_events, event_stride)
        dt = dt[indices]
        energy = energy[indices]
        if event_ids is not None:
            event_ids = event_ids[indices]

    removed_source = 0
    if exclude_source_ids and event_ids is not None:
        source_group = None
        if "source" in run_group:
            source_group = run_group["source"]
        elif "sources" in run_group:
            source_group = run_group["sources"]

        if source_group is not None and "id" in source_group:
            source_ids = np.asarray(source_group["id"])
            if source_ids.size > 0:
                mask = ~np.isin(event_ids, source_ids)
                removed_source = int(np.count_nonzero(~mask))
                dt = dt[mask]
                energy = energy[mask]

    time_deltas = ARADLSTMDetector._convert_time_deltas(np.asarray(dt), time_units)
    energies = np.asarray(energy, dtype=np.float64)

    return time_deltas, energies, int(n_events), removed_source


# ---------------------------------------------------------------------------
# Fast vectorized spectra creation (replaces gammaflow from_list_mode)
# ---------------------------------------------------------------------------

def _fast_create_spectra(
    time_deltas: np.ndarray,
    energies: np.ndarray,
    integration_time: float,
    stride_time: float,
    energy_bins: int,
    energy_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create spectra from listmode data — fully vectorized, no Python loop.

    Performance strategy:
      1. np.cumsum for absolute event times
      2. np.digitize once to pre-bin all energies
      3. np.searchsorted for O(n_windows * log(n_events)) window boundaries
      4. Assign each event a window ID, then use np.bincount on a
         flattened (window, bin) index to scatter-add counts across
         all windows simultaneously — pure numpy, no scipy needed.
         This replaces the O(n_windows) Python for-loop with a single
         vectorized pass, giving ~10-50x speedup on large runs.

    Returns
    -------
    counts : (n_windows, energy_bins)
    timestamps : (n_windows,)
    real_times : (n_windows,)
    energy_edges : (energy_bins + 1,)
    events_per_window : (n_windows,) int — raw event count per window
    """
    if len(time_deltas) == 0:
        edges = np.linspace(energy_range[0], energy_range[1], energy_bins + 1)
        empty = np.empty((0, energy_bins), dtype=np.float64)
        return empty, np.empty(0), np.empty(0), edges, np.empty(0, dtype=np.int64)

    # Absolute event times (cumulative sum of deltas)
    abs_times = np.cumsum(time_deltas)
    total_time = abs_times[-1]

    # Energy bin edges
    edges = np.linspace(energy_range[0], energy_range[1], energy_bins + 1)

    # Pre-digitize energies into bin indices once for all windows.
    # np.digitize returns 1-based index; 0 means below first edge,
    # energy_bins means the last valid bin (1-based), energy_bins+1 = above.
    bin_indices = np.digitize(energies, edges)  # 1-based

    # Window starts
    window_starts = np.arange(0.0, total_time, stride_time)
    n_windows = len(window_starts)

    if n_windows == 0:
        empty = np.empty((0, energy_bins), dtype=np.float64)
        return empty, np.empty(0), np.empty(0), edges, np.empty(0, dtype=np.int64)

    window_ends = window_starts + integration_time

    # Use searchsorted to find first and last event index for each window
    left_indices = np.searchsorted(abs_times, window_starts, side="left")
    right_indices = np.searchsorted(abs_times, window_ends, side="left")

    # Events per window (vectorized)
    events_per_window = (right_indices - left_indices).astype(np.int64)

    # ---- Fully vectorized histogram: sparse COO scatter-add ----
    # For each window, we need to histogram its events into energy bins.
    # Instead of looping, we expand events into (window_id, bin_id) pairs
    # and use a sparse matrix to accumulate counts.
    #
    # Key insight: with overlapping windows (stride < integration), each
    # event belongs to multiple windows. We use searchsorted on the event
    # times against window boundaries to find which windows each event
    # belongs to, then replicate accordingly.

    # For each event, find which windows it falls into.
    # Event at abs_times[j] belongs to window i if:
    #   window_starts[i] <= abs_times[j] < window_ends[i]
    # Equivalently: i >= first_window and i <= last_window where
    #   first_window = first i such that window_ends[i] > abs_times[j]
    #     (but also window_starts[i] <= abs_times[j])
    # Simplest: for each event, the range of windows it falls in is:
    #   first_win = max(0, ceil((t - integration_time) / stride_time))
    #   last_win  = min(n_windows-1, floor(t / stride_time))

    # Filter to valid energy bins first (avoids wasting work on out-of-range)
    valid_mask = (bin_indices >= 1) & (bin_indices <= energy_bins)
    valid_abs_times = abs_times[valid_mask]
    valid_bins_0based = bin_indices[valid_mask] - 1  # to 0-based

    if valid_abs_times.size == 0:
        counts = np.zeros((n_windows, energy_bins), dtype=np.float64)
        timestamps = window_starts + integration_time / 2.0
        real_times = np.full(n_windows, integration_time, dtype=np.float64)
        return counts, timestamps, real_times, edges, events_per_window

    # For each valid event, compute range of windows it belongs to
    ev_first_win = np.maximum(
        0,
        np.ceil((valid_abs_times - integration_time) / stride_time).astype(np.int64)
    )
    ev_last_win = np.minimum(
        n_windows - 1,
        np.floor(valid_abs_times / stride_time).astype(np.int64)
    )
    # Clamp to ensure non-negative last_win (edge case: event before first window end)
    ev_last_win = np.maximum(ev_last_win, 0)

    # Number of windows each event contributes to
    spans = (ev_last_win - ev_first_win + 1).astype(np.int64)
    spans = np.maximum(spans, 0)  # safety clamp
    total_pairs = spans.sum()

    if total_pairs == 0:
        counts = np.zeros((n_windows, energy_bins), dtype=np.float64)
        timestamps = window_starts + integration_time / 2.0
        real_times = np.full(n_windows, integration_time, dtype=np.float64)
        return counts, timestamps, real_times, edges, events_per_window

    # Build (window_id, bin_id) pairs via np.repeat, then scatter-add
    # with np.bincount on a flattened index.  Pure numpy, no scipy.
    #
    # Memory note: each event contributes to `span` windows, so the
    # expanded arrays have `total_pairs` elements.  For very large
    # overlaps we process in chunks to cap memory usage.
    MAX_PAIRS = 200_000_000  # ~1.6 GB safety limit

    def _scatter_add_chunk(first_win, bins_0, chunk_spans, n_win, n_bins):
        """Expand events → (win, bin) pairs and bincount into a matrix."""
        ct = int(chunk_spans.sum())
        win_ids = np.repeat(first_win, chunk_spans)
        offsets = np.arange(ct) - np.repeat(
            np.cumsum(chunk_spans) - chunk_spans, chunk_spans
        )
        win_ids = win_ids + offsets
        bin_ids = np.repeat(bins_0, chunk_spans)
        # Flatten (win, bin) → single index, then bincount
        flat = win_ids * n_bins + bin_ids
        return np.bincount(flat, minlength=n_win * n_bins).reshape(n_win, n_bins).astype(np.float64)

    if total_pairs <= MAX_PAIRS:
        counts = _scatter_add_chunk(
            ev_first_win, valid_bins_0based, spans, n_windows, energy_bins
        )
    else:
        # Chunked fallback for extreme overlap (very rare)
        counts = np.zeros((n_windows, energy_bins), dtype=np.float64)
        CHUNK = 500_000
        for start in range(0, len(valid_abs_times), CHUNK):
            end = min(start + CHUNK, len(valid_abs_times))
            c_spans = spans[start:end]
            if c_spans.sum() == 0:
                continue
            counts += _scatter_add_chunk(
                ev_first_win[start:end],
                valid_bins_0based[start:end],
                c_spans,
                n_windows,
                energy_bins,
            )

    timestamps = window_starts + integration_time / 2.0
    real_times = np.full(n_windows, integration_time, dtype=np.float64)

    return counts, timestamps, real_times, edges, events_per_window


def preprocess_run(
    run_group: h5py.Group,
    run_key: str,
    run_id: int,
    integration_time: float,
    stride_time: float,
    energy_bins: int,
    energy_range: Tuple[float, float],
    time_units: str,
    max_events_per_run: Optional[int],
    event_stride: int,
    exclude_source_ids: bool,
) -> dict:
    """Convert a single run into L1-normalized spectra and metadata.

    Each output spectrum sums to 1 (per-spectrum L1 normalization).
    """
    time_deltas, energies, total_events, removed_source = _load_filtered_listmode_from_group(
        run_group=run_group,
        run_key=run_key,
        time_units=time_units,
        max_events_per_run=max_events_per_run,
        event_stride=event_stride,
        exclude_source_ids=exclude_source_ids,
    )

    # --- Fast vectorized spectra creation (replaces gammaflow) ---
    counts, timestamps, real_times, energy_edges, events_per_window = _fast_create_spectra(
        time_deltas=time_deltas,
        energies=energies,
        integration_time=integration_time,
        stride_time=stride_time,
        energy_bins=energy_bins,
        energy_range=energy_range,
    )

    if counts.shape[0] == 0:
        # Empty run — return minimal structure
        return {
            "spectra": torch.FloatTensor(np.empty((0, energy_bins))),
            "count_rates": torch.FloatTensor(np.empty(0)),
            "timestamps": timestamps,
            "live_times": real_times,
            "real_times": real_times,
            "energy_edges": energy_edges,
            "integration_time": integration_time,
            "stride_time": stride_time,
            "energy_range": energy_range,
            "time_units": time_units,
            "events_total": total_events,
            "events_removed_source": removed_source,
            "normalization": "per-spectrum-l1",
        }

    # live_times — listmode has no dead-time info, so live = real
    live_times = real_times.copy()

    # Raw count rates per bin (counts per second per bin)
    spectra = counts / live_times[:, np.newaxis]

    # Gross count rate per spectrum (total counts/second) — saved before L1 normalization
    count_rates = spectra.sum(axis=1)  # shape (n_spectra,)

    # Apply L1 normalization: each spectrum sums to 1
    row_sums = count_rates.copy()[:, np.newaxis]
    row_sums = np.maximum(row_sums, 1e-10)
    spectra = spectra / row_sums

    return {
        "spectra": torch.FloatTensor(spectra),
        "count_rates": torch.FloatTensor(count_rates),
        "timestamps": timestamps,
        "live_times": live_times,
        "real_times": real_times,
        "energy_edges": energy_edges,
        "integration_time": integration_time,
        "stride_time": stride_time,
        "energy_range": energy_range,
        "time_units": time_units,
        "events_total": total_events,
        "events_removed_source": removed_source,
        "normalization": "per-spectrum-l1",
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess RADAI runs into tensors")
    parser.add_argument("--h5-path", type=str, required=True, help="Path to RADAI HDF5 file")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "RADAI-preprocessed"),
                        help="Output directory for .pt tensors; created if needed (default: RADAI-preprocessed)")
    parser.add_argument("--integration-time", type=float, default=1.0, help="Integration time per spectrum (s)")
    parser.add_argument("--stride-time", type=float, default=1.0, help="Stride between spectra (s)")
    parser.add_argument("--energy-bins", type=int, default=128, help="Number of energy bins")
    parser.add_argument("--energy-min", type=float, default=0.0, help="Min energy (keV)")
    parser.add_argument("--energy-max", type=float, default=3000.0, help="Max energy (keV)")
    parser.add_argument("--time-units", type=str, default="us", help="Time units in HDF5 file (us, ms, s)")
    parser.add_argument("--max-runs", type=int, default=None, help="Max runs to process")
    parser.add_argument("--max-events-per-run", type=int, default=None, help="Max events per run")
    parser.add_argument("--event-stride", type=int, default=1, help="Downsample events by stride")
    parser.add_argument(
        "--include-sources",
        action="store_true",
        help="Do not filter out listmode events with source IDs"
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    output_dir = resolve_output_dir(args.output_dir)

    energy_range = (args.energy_min, args.energy_max)

    run_ids = get_all_run_ids(h5_path)
    if args.max_runs is not None:
        run_ids = run_ids[: args.max_runs]

    print(f"Processing {len(run_ids)} runs from {h5_path}")
    print(f"Normalization: per-spectrum-l1")
    print(f"Output directory: {output_dir}")

    total_spectra = 0
    wall_start = _time.perf_counter()

    # Open HDF5 once — avoids per-run open/close overhead
    with h5py.File(h5_path, "r") as f:
        runs_group = f["runs"]
        for i, run_id in enumerate(run_ids, start=1):
            run_key = ARADLSTMDetector._format_run_key(run_id)
            if run_key not in runs_group:
                print(f"  [SKIP] {run_key} not found")
                continue

            t0 = _time.perf_counter()
            data = preprocess_run(
                run_group=runs_group[run_key],
                run_key=run_key,
                run_id=run_id,
                integration_time=args.integration_time,
                stride_time=args.stride_time,
                energy_bins=args.energy_bins,
                energy_range=energy_range,
                time_units=args.time_units,
                max_events_per_run=args.max_events_per_run,
                event_stride=args.event_stride,
                exclude_source_ids=not args.include_sources,
            )

            spectra = data["spectra"]
            total_spectra += spectra.shape[0]
            elapsed = _time.perf_counter() - t0

            extra = ""
            if data.get("events_removed_source", 0) > 0:
                extra = f"  (removed {data['events_removed_source']} source events)"

            print(f"[{i}/{len(run_ids)}] Run {run_id}: "
                  f"{spectra.shape[0]} spectra, {spectra.shape[1]} features, "
                  f"{elapsed:.2f}s{extra}")

            out_path = output_dir / f"run{run_id}.pt"
            torch.save(data, out_path)

    wall_elapsed = _time.perf_counter() - wall_start

    stats = {
        "h5_path": str(h5_path),
        "output_dir": str(output_dir),
        "runs_processed": len(run_ids),
        "total_spectra": total_spectra,
        "integration_time": args.integration_time,
        "stride_time": args.stride_time,
        "energy_bins": args.energy_bins,
        "energy_range": list(energy_range),
        "time_units": args.time_units,
        "normalization": "per-spectrum-l1",
        "count_rate_definition": "gross_cps=sum(counts_per_bin/live_time)",
        "count_rate_features_recommended": ["gross_cps", "log1p(gross_cps)", "delta_log_cps"],
        "wall_time_seconds": round(wall_elapsed, 2),
    }
    stats_path = output_dir / "preprocess_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDone. Wrote {total_spectra} spectra across {len(run_ids)} runs to {output_dir}")
    print(f"Total wall time: {wall_elapsed:.1f}s ({wall_elapsed/max(len(run_ids),1):.2f}s/run)")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
