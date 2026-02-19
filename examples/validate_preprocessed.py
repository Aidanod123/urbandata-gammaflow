"""
Validate preprocessed per-spectrum-l1 datasets.

This script checks that all preprocessed datasets were created correctly:
- Verifies output directories exist with expected files
- Checks preprocess_stats.json for correct normalization mode
- Validates that spectra are properly L1-normalized (sum to 1.0)
- Reports any issues found

Usage:
    python examples/validate_preprocessed.py --data-root pre-spectrum-norm
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)


def validate_stats_file(stats_path: Path, expected_norm: str) -> Tuple[bool, List[str]]:
    """Validate preprocess_stats.json file."""
    issues = []
    
    if not stats_path.exists():
        return False, [f"Missing preprocess_stats.json"]
    
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON in stats file: {e}"]
    
    # Check required fields
    required_fields = ['runs_processed', 'total_spectra', 'integration_time', 
                       'stride_time', 'energy_bins', 'normalization']
    for field in required_fields:
        if field not in stats:
            issues.append(f"Missing field in stats: {field}")
    
    # Check normalization mode
    if stats.get('normalization') != expected_norm:
        issues.append(f"Wrong normalization: expected '{expected_norm}', got '{stats.get('normalization')}'")
    
    return len(issues) == 0, issues


def validate_run_file(
    run_path: Path, 
    expected_norm: str, 
    check_l1: bool = True,
    expected_bins: int = 128,
    min_sequence_length: int = 10
) -> Tuple[bool, List[str], Dict]:
    """Validate a single run .pt file.
    
    Parameters
    ----------
    run_path : Path
        Path to the .pt file
    expected_norm : str
        Expected normalization mode
    check_l1 : bool
        Whether to verify L1 normalization (sums to 1)
    expected_bins : int
        Expected number of energy bins (must match ARAD-LSTM n_bins)
    min_sequence_length : int
        Minimum number of spectra needed for training sequences
    
    Returns
    -------
    valid : bool
    issues : List[str]
    metadata : Dict with n_spectra, n_bins, dtype info
    """
    issues = []
    metadata = {'n_spectra': 0, 'n_bins': 0, 'dtype': None}
    
    try:
        data = torch.load(run_path, map_location='cpu', weights_only=False)
    except Exception as e:
        return False, [f"Failed to load: {e}"], metadata
    
    # Check required keys
    required_keys = ['spectra', 'normalization', 'timestamps', 'real_times', 'energy_edges']
    for key in required_keys:
        if key not in data:
            issues.append(f"Missing key: {key}")
    
    if 'spectra' not in data:
        return False, issues, metadata
    
    spectra = data['spectra']
    metadata['dtype'] = str(spectra.dtype)
    
    # Check normalization field
    if data.get('normalization') != expected_norm:
        issues.append(f"Wrong normalization field: expected '{expected_norm}', got '{data.get('normalization')}'")
    
    # Check for NaN/Inf
    if torch.isnan(spectra).any():
        issues.append("Contains NaN values")
    if torch.isinf(spectra).any():
        issues.append("Contains Inf values")
    
    # Check shape dimensions
    if spectra.dim() != 2:
        issues.append(f"Wrong spectra dimensions: expected 2, got {spectra.dim()}")
        return len(issues) == 0, issues, metadata
    
    n_spectra, n_bins = spectra.shape
    metadata['n_spectra'] = n_spectra
    metadata['n_bins'] = n_bins
    
    # Check energy bins match architecture requirement
    if n_bins != expected_bins:
        issues.append(f"Wrong number of energy bins: expected {expected_bins}, got {n_bins} "
                      f"(ARAD-LSTM architecture requires exactly {expected_bins} bins)")
    
    # Check minimum spectra for sequence training
    if n_spectra < min_sequence_length + 1:
        issues.append(f"Too few spectra: {n_spectra} < {min_sequence_length + 1} "
                      f"(need at least sequence_length + 1 for 'next' target mode)")
    
    # Check data type is float32 for efficient GPU training
    if spectra.dtype != torch.float32:
        issues.append(f"Suboptimal dtype: {spectra.dtype} (recommend torch.float32 for training)")
    
    # Check L1 normalization (each row should sum to ~1.0)
    if check_l1 and expected_norm == 'per-spectrum-l1':
        spectra_np = spectra.numpy()
        row_sums = spectra_np.sum(axis=1)
        
        # Allow small tolerance for floating point
        tolerance = 1e-5
        not_normalized = np.abs(row_sums - 1.0) > tolerance
        
        # Skip rows that are all zeros (empty spectra)
        non_empty = spectra_np.sum(axis=1) > 1e-10
        bad_rows = not_normalized & non_empty
        
        if bad_rows.any():
            n_bad = bad_rows.sum()
            bad_sums = row_sums[bad_rows][:5]  # Show first 5
            issues.append(f"{n_bad} spectra not L1-normalized (sums: {bad_sums.tolist()})")
    
    # Check value range for L1 normalized data
    if expected_norm == 'per-spectrum-l1':
        if (spectra < 0).any():
            issues.append("Contains negative values")
        if (spectra > 1).any():
            n_over = (spectra > 1).sum().item()
            max_val = spectra.max().item()
            issues.append(f"{n_over} values > 1.0 (max: {max_val:.4f})")
    
    return len(issues) == 0, issues, metadata


def validate_dataset(
    data_dir: Path, 
    expected_norm: str, 
    sample_runs: int = 5,
    expected_bins: int = 128,
    min_sequence_length: int = 10
) -> Dict:
    """Validate an entire preprocessed dataset directory."""
    result = {
        'path': str(data_dir),
        'exists': data_dir.exists(),
        'stats_valid': False,
        'stats_issues': [],
        'run_files': 0,
        'runs_checked': 0,
        'runs_valid': 0,
        'run_issues': [],
        'total_spectra': 0,
        'integration_time': None,
        'stride_time': None,
        'expected_bins': expected_bins,
        'min_sequence_length': min_sequence_length,
        'min_spectra_in_run': None,
        'max_spectra_in_run': None,
        'runs_too_short': 0,
    }
    
    if not data_dir.exists():
        return result
    
    # Validate stats file
    stats_path = data_dir / 'preprocess_stats.json'
    stats_valid, stats_issues = validate_stats_file(stats_path, expected_norm)
    result['stats_valid'] = stats_valid
    result['stats_issues'] = stats_issues
    
    if stats_path.exists():
        try:
            with open(stats_path) as f:
                stats = json.load(f)
                result['total_spectra'] = stats.get('total_spectra', 0)
                result['integration_time'] = stats.get('integration_time')
                result['stride_time'] = stats.get('stride_time')
        except:
            pass
    
    # Find and validate run files
    run_files = sorted(data_dir.glob('run*.pt'))
    result['run_files'] = len(run_files)
    
    if not run_files:
        result['run_issues'].append("No run*.pt files found")
        return result
    
    # Sample runs to check (first, last, and random middle ones)
    if len(run_files) <= sample_runs:
        runs_to_check = run_files
    else:
        # First, last, and evenly spaced middle ones
        indices = [0, len(run_files) - 1]
        step = len(run_files) // (sample_runs - 2)
        indices.extend(range(step, len(run_files) - 1, step)[:sample_runs - 2])
        runs_to_check = [run_files[i] for i in sorted(set(indices))]
    
    result['runs_checked'] = len(runs_to_check)
    
    spectra_counts = []
    for run_path in runs_to_check:
        valid, issues, metadata = validate_run_file(
            run_path, expected_norm, 
            expected_bins=expected_bins,
            min_sequence_length=min_sequence_length
        )
        spectra_counts.append(metadata['n_spectra'])
        
        if metadata['n_spectra'] < min_sequence_length + 1:
            result['runs_too_short'] += 1
        
        if valid:
            result['runs_valid'] += 1
        else:
            result['run_issues'].append({
                'file': run_path.name,
                'issues': issues,
                'n_spectra': metadata['n_spectra'],
                'n_bins': metadata['n_bins']
            })
    
    if spectra_counts:
        result['min_spectra_in_run'] = min(spectra_counts)
        result['max_spectra_in_run'] = max(spectra_counts)
    
    return result


def print_report(results: List[Dict], expected_norm: str, expected_bins: int, min_seq: int):
    """Print a formatted validation report."""
    print("\n" + "=" * 70)
    print(f"  PREPROCESSED DATASET VALIDATION REPORT")
    print(f"  Expected normalization: {expected_norm}")
    print(f"  Expected energy bins: {expected_bins} (ARAD-LSTM architecture)")
    print(f"  Min sequence length: {min_seq} (for training)")
    print("=" * 70)
    
    all_valid = True
    
    for r in results:
        name = Path(r['path']).name
        print(f"\n{'─' * 70}")
        print(f"  Dataset: {name}")
        print(f"{'─' * 70}")
        
        if not r['exists']:
            print(f"  ❌ Directory does not exist")
            all_valid = False
            continue
        
        # Stats info
        if r['integration_time']:
            print(f"  Integration/Stride: {r['integration_time']}s / {r['stride_time']}s")
        print(f"  Total spectra: {r['total_spectra']:,}")
        print(f"  Run files: {r['run_files']}")
        if r['min_spectra_in_run'] is not None:
            print(f"  Spectra per run: {r['min_spectra_in_run']} - {r['max_spectra_in_run']} (sampled)")
        
        # Stats validation
        if r['stats_valid']:
            print(f"  ✅ preprocess_stats.json: Valid")
        else:
            print(f"  ❌ preprocess_stats.json: Invalid")
            for issue in r['stats_issues']:
                print(f"      - {issue}")
            all_valid = False
        
        # Run file validation
        if r['runs_checked'] > 0:
            if r['runs_valid'] == r['runs_checked']:
                print(f"  ✅ Run files checked: {r['runs_valid']}/{r['runs_checked']} valid")
            else:
                print(f"  ❌ Run files checked: {r['runs_valid']}/{r['runs_checked']} valid")
                all_valid = False
                for run_issue in r['run_issues'][:3]:  # Show first 3
                    print(f"      {run_issue['file']} ({run_issue.get('n_spectra', '?')} spectra, {run_issue.get('n_bins', '?')} bins):")
                    for issue in run_issue['issues']:
                        print(f"        - {issue}")
                if len(r['run_issues']) > 3:
                    print(f"      ... and {len(r['run_issues']) - 3} more files with issues")
        
        # Architecture compatibility summary
        if r['runs_too_short'] > 0:
            print(f"  ⚠️  {r['runs_too_short']} runs too short for sequence_length={min_seq}")
    
    print(f"\n{'=' * 70}")
    if all_valid:
        print("  ✅ ALL DATASETS VALIDATED SUCCESSFULLY")
        print(f"     Ready for ARAD-LSTM training with {expected_bins} bins")
    else:
        print("  ❌ SOME DATASETS HAVE ISSUES - SEE ABOVE")
    print("=" * 70 + "\n")
    
    return all_valid


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed datasets")
    parser.add_argument("--data-root", type=str, default="per-spectrum-norm",
                        help="Root directory containing dataset subdirectories")
    parser.add_argument("--expected-norm", type=str, default="per-spectrum-l1",
                        help="Expected normalization mode (per-spectrum-l1)")
    parser.add_argument("--sample-runs", type=int, default=5,
                        help="Number of run files to sample per dataset")
    parser.add_argument("--expected-bins", type=int, default=128,
                        help="Expected number of energy bins (ARAD-LSTM default: 128)")
    parser.add_argument("--min-sequence-length", type=int, default=10,
                        help="Minimum sequence length for training (default: 10)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific dataset names to check (default: all)")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        print(f"ERROR: Data root '{data_root}' does not exist")
        sys.exit(1)
    
    # Find datasets to validate
    if args.datasets:
        dataset_dirs = [data_root / name for name in args.datasets]
    else:
        # Default: check all no-sources-* directories
        dataset_dirs = sorted(data_root.glob("no-sources-*"))
        if not dataset_dirs:
            # Fallback: check all subdirectories
            dataset_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    if not dataset_dirs:
        print(f"ERROR: No dataset directories found in '{data_root}'")
        sys.exit(1)
    
    print(f"Validating {len(dataset_dirs)} datasets in '{data_root}'...")
    print(f"  Architecture requirements: {args.expected_bins} bins, min {args.min_sequence_length} spectra/run")
    
    results = []
    for dataset_dir in dataset_dirs:
        print(f"  Checking {dataset_dir.name}...", end=" ", flush=True)
        result = validate_dataset(
            dataset_dir, 
            args.expected_norm, 
            args.sample_runs,
            expected_bins=args.expected_bins,
            min_sequence_length=args.min_sequence_length
        )
        results.append(result)
        
        if not result['exists']:
            print("NOT FOUND")
        elif result['stats_valid'] and result['runs_valid'] == result['runs_checked']:
            print("OK")
        else:
            print("ISSUES FOUND")
    
    all_valid = print_report(results, args.expected_norm, args.expected_bins, args.min_sequence_length)
    
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
