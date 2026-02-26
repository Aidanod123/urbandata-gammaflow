"""
ARAD-LSTM Training Script

Train an ARAD-LSTM detector on background data from RADAI HDF5 files
and visualize training results and reconstructions.

This script demonstrates the advanced LSTM architecture with:
- Spectral feature extraction via 1D CNN
- Temporal attention mechanism
- Multiple loss functions (JSD, Chi-squared, MSE)
- Data augmentation for robustness
- Multi-run training capability
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import json
import argparse

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detectors.arad_lstm import ARADLSTMDetector


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data settings
PREPROCESSED_DIR = ROOT / 'RADAI-preprocessed'

# Spectrum parameters
INTEGRATION_TIME = 1.0  # seconds per spectrum
STRIDE_TIME = 1.0       # step between spectra
ENERGY_BINS = 128       # number of energy bins (must work with CNN architecture)
ENERGY_RANGE = (0.0, 3000.0)  # keV

# Model hyperparameters
SEQUENCE_LENGTH = 10    # temporal context window
HIDDEN_SIZE = 128       # LSTM hidden size
LATENT_DIM = 32         # bottleneck dimension
NUM_LAYERS = 2          # LSTM layers
DROPOUT = 0.2           # regularization
BIDIRECTIONAL = False   # False for streaming, True for offline
USE_ATTENTION = True    # temporal attention mechanism
NUM_ATTENTION_HEADS = 4

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
L1_LAMBDA = 1e-5
L2_LAMBDA = 1e-4
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.2
LOSS_TYPE = 'chi2'      # 'chi2', 'jsd', or 'mse'
USE_AUGMENTATION = True

# Output settings
MODEL_DIR = ROOT / 'models'
MODEL_NAME = 'arad_lstm_background.pt'
N_RECONSTRUCTIONS = 5


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_preprocessed_run_ids(preprocessed_dir: Path) -> List[int]:
    """Get all run IDs from preprocessed tensor files."""
    run_ids = []
    for path in preprocessed_dir.glob('run*.pt'):
        stem = path.stem
        if stem.startswith('run'):
            run_ids.append(int(stem.replace('run', '')))
    return sorted(run_ids)


def select_training_runs(
    preprocessed_dir: Path,
    max_runs: Optional[int] = None,
    seed: int = 42
) -> List[int]:
    """
    Select runs for training.
    
    In a real scenario, you might filter for background-only runs
    or use specific criteria. Here we just sample from available runs.
    """
    all_runs = get_all_preprocessed_run_ids(preprocessed_dir)
    
    np.random.seed(seed)
    np.random.shuffle(all_runs)
    
    if max_runs is not None:
        all_runs = all_runs[:max_runs]
    
    return all_runs


def plot_training_history(history: dict, save_path: Path):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('ARAD-LSTM Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to: {save_path}")


def plot_reconstructions(
    detector: ARADLSTMDetector,
    time_series_list: list,
    n_samples: int,
    save_path: Path
):
    """Plot sample reconstructions to verify model quality."""
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))
    
    for i in range(n_samples):
        # Select a random time series and position
        ts_idx = np.random.randint(len(time_series_list))
        ts = time_series_list[ts_idx]
        
        # Extract spectra
        counts = ts.counts
        times = ts.live_times
        if times is None or np.any(np.isnan(times)):
            times = ts.real_times
        spectra = counts / times[:, np.newaxis]
        
        # Select a random valid position
        pos = np.random.randint(detector.sequence_length, len(spectra))
        sequence = spectra[pos - detector.sequence_length + 1:pos + 1]
        
        # Get reconstruction
        reconstructed = detector.reconstruct_spectrum(sequence)
        original = sequence[-1]
        
        # Plot original
        ax1 = axes[0, i]
        ax1.plot(original, 'b-', alpha=0.7, label='Original')
        ax1.set_yscale('log')
        ax1.set_ylabel('Count Rate' if i == 0 else '')
        ax1.set_title(f'Sample {i + 1}')
        ax1.grid(True, alpha=0.3)
        if i == 0:
            ax1.legend()
        
        # Plot reconstructed
        ax2 = axes[1, i]
        ax2.plot(reconstructed, 'r-', alpha=0.7, label='Reconstructed')
        ax2.set_yscale('log')
        ax2.set_xlabel('Energy Bin')
        ax2.set_ylabel('Count Rate' if i == 0 else '')
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.legend()
        
        # Compute error
        mse = np.mean((original - reconstructed) ** 2)
        ax2.set_title(f'MSE: {mse:.2e}')
    
    axes[0, n_samples // 2].set_title('Original Spectra', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Reconstructions saved to: {save_path}")


def plot_error_distribution(
    detector: ARADLSTMDetector,
    time_series_list: list,
    n_samples: int,
    save_path: Path
):
    """Plot distribution of reconstruction errors."""
    scores = []
    
    for ts in time_series_list[:min(5, len(time_series_list))]:
        counts = ts.counts
        times = ts.live_times
        if times is None or np.any(np.isnan(times)):
            times = ts.real_times
        spectra = counts / times[:, np.newaxis]
        
        for pos in range(detector.sequence_length, len(spectra)):
            sequence = spectra[pos - detector.sequence_length + 1:pos + 1]
            score = detector.score_sequence(sequence)
            scores.append(score)
    
    scores = np.array(scores)
    
    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=50, density=True, alpha=0.7, edgecolor='black')
    plt.axvline(np.percentile(scores, 95), color='r', linestyle='--', 
                label=f'95th percentile: {np.percentile(scores, 95):.4f}')
    plt.axvline(np.percentile(scores, 99), color='orange', linestyle='--',
                label=f'99th percentile: {np.percentile(scores, 99):.4f}')
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Reconstruction Errors (Background)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Error distribution saved to: {save_path}")


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated list of ints (e.g., '5,10,20')."""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(',') if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ARAD-LSTM detector')
    parser.add_argument('--max-runs', type=int, default=20,
                        help='Maximum number of runs to use for training')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size for training (larger = better GPU utilization)')
    parser.add_argument('--loss', type=str, default=LOSS_TYPE,
                        choices=['jsd', 'chi2', 'mse'],
                        help='Loss function to use')
    parser.add_argument('--output-activation', type=str, default='sigmoid',
                        choices=['sigmoid', 'softmax'],
                        help='Decoder output activation for reconstructed spectra')
    parser.add_argument('--sequence-length', type=int, default=SEQUENCE_LENGTH,
                        help='Sequence length (time steps) for the LSTM')
    parser.add_argument('--sequence-lengths', type=str, default=None,
                        help='Comma-separated list of sequence lengths to sweep (e.g., 5,10,20)')
    parser.add_argument('--no-attention', action='store_true',
                        help='Disable temporal attention')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional LSTM (offline mode)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--no-count-rate', action='store_true',
                        help='Disable count rate side-channel (shape-only mode)')
    parser.add_argument('--preprocessed-dir', type=str, default=str(PREPROCESSED_DIR),
                        help='Directory with preprocessed run*.pt tensors')
    parser.add_argument('--cache-size', type=int, default=50,
                        help='Number of runs to cache in memory')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader workers for parallel loading (0=main thread, 2-4 recommended)')
    parser.add_argument('--require-runs', type=int, default=None,
                        help='Fail if fewer than this many preprocessed runs are found')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Name for the saved model (e.g. "shape_only_chi2"). '
                             'Defaults to arad_lstm_background')
    args = parser.parse_args()
    
    print("=" * 80)
    print("ARAD-LSTM TRAINING SCRIPT")
    print("=" * 80)
    print()
    
    preprocessed_dir = Path(args.preprocessed_dir)
    if not preprocessed_dir.exists():
        print(f"ERROR: Preprocessed data directory not found at {preprocessed_dir}")
        print("Run examples/preprocess_radai_runs.py first.")
        return
    
    # Create models directory
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Select training runs
    print("Selecting training runs...")
    run_ids = select_training_runs(preprocessed_dir, max_runs=args.max_runs)
    print(f"Selected {len(run_ids)} runs for training")
    if args.require_runs is not None and len(run_ids) < args.require_runs:
        print(f"ERROR: Found {len(run_ids)} runs, but --require-runs={args.require_runs}")
        return
    
    print()
    
    # Display configuration
    print("Configuration:")
    print(f"  Data source: {preprocessed_dir}")

    stats_path = preprocessed_dir / 'preprocess_stats.json'
    if stats_path.exists():
        try:
            with stats_path.open('r', encoding='utf-8') as f:
                stats = json.load(f)
            print(f"  Integration time: {stats.get('integration_time', 'n/a')} s")
            print(f"  Stride time: {stats.get('stride_time', 'n/a')} s")
            print(f"  Energy bins: {stats.get('energy_bins', 'n/a')}")
            print(f"  Energy range: {stats.get('energy_range', 'n/a')} keV")
        except Exception:
            print("  Integration time: n/a")
            print("  Stride time: n/a")
            print(f"  Energy bins: {ENERGY_BINS}")
            print(f"  Energy range: {ENERGY_RANGE} keV")
    else:
        print(f"  Integration time: {INTEGRATION_TIME} s")
        print(f"  Stride time: {STRIDE_TIME} s")
        print(f"  Energy bins: {ENERGY_BINS}")
        print(f"  Energy range: {ENERGY_RANGE} keV")

    sequence_lengths = parse_int_list(args.sequence_lengths) or [args.sequence_length]
    print(f"  Sequence lengths: {sequence_lengths}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  LSTM layers: {NUM_LAYERS}")
    print(f"  Dropout: {DROPOUT}")
    print(f"  Bidirectional: {args.bidirectional}")
    print(f"  Attention: {not args.no_attention}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Loss function: {args.loss.upper()}")
    print(f"  Output activation: {args.output_activation}")
    print("  Spatial backbone: ARAD CNN encoder/decoder")
    print(f"  Data augmentation: {not args.no_augmentation}")
    print(f"  Count rate side-channel: {not args.no_count_rate}")
    print(f"  Cache size: {args.cache_size} runs")
    print()
    
    for seq_len in sequence_lengths:
        print("Initializing ARAD-CNN + LSTM detector...")
        detector = ARADLSTMDetector(
            sequence_length=seq_len,
            hidden_size=HIDDEN_SIZE,
            latent_dim=LATENT_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            bidirectional=args.bidirectional,
            use_attention=not args.no_attention,
            num_attention_heads=NUM_ATTENTION_HEADS,
            batch_size=args.batch_size,
            learning_rate=LEARNING_RATE,
            epochs=args.epochs,
            l1_lambda=L1_LAMBDA,
            l2_lambda=L2_LAMBDA,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            loss_type=args.loss,
            output_activation=args.output_activation,
            use_augmentation=not args.no_augmentation,
            use_count_rate=not args.no_count_rate,
            verbose=True
        )
        
        # Print GPU status before training
        detector.print_gpu_status()
        
        print("=" * 80)
        print(f"TRAINING (Preprocessed Runs) - seq_len={seq_len}")
        print("=" * 80)
        detector.fit_from_preprocessed(
            data_dir=str(preprocessed_dir),
            run_ids=run_ids,
            validation_split_runs=VALIDATION_SPLIT,
            cache_size_runs=args.cache_size,
            num_workers=args.num_workers
        )
    
        print("=" * 80)
        print()

        # Save the model
        model_stem = args.model_name if args.model_name else MODEL_NAME.replace('.pt', '')
        model_path = MODEL_DIR / f"{model_stem}_seq{seq_len}.pt"
        detector.save(str(model_path))
        print()

        # Plot training history
        print("Generating training visualizations...")
        history = detector.get_training_history()
        plot_training_history(history, MODEL_DIR / f"arad_lstm_training_history_seq{seq_len}.png")

        print("  (Skipping reconstruction/error plots for preprocessed runs)")
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Models saved to: {MODEL_DIR}")
    print(f"Visualizations saved to: {MODEL_DIR}")
    print()
    print("Next steps:")
    print("  1. Run threshold calibration on background data")
    print("  2. Test on data with known sources")
    print("  3. Evaluate detection performance (ROC, FAR, etc.)")


if __name__ == "__main__":
    main()
