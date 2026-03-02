"""
ARAD-LSTM: Advanced Temporal LSTM-based Anomaly Detection for gamma-ray spectra.

A sophisticated deep learning approach for detecting anomalies in streaming gamma-ray
spectral data. The model leverages temporal dependencies between consecutive spectra
to learn the normal evolution patterns of background radiation.

Architecture Highlights:
========================
1. Spectral Feature Extraction: Original ARAD 1D CNN encoder captures local spectral patterns
   (peaks, edges, broadening) before LSTM processing
2. Temporal Attention: Self-attention mechanism allows the model to focus on
   relevant historical spectra when predicting the current spectrum
3. Bidirectional Option: For offline analysis, bidirectional LSTM provides
   richer context; for streaming, causal (unidirectional) mode is used
4. Latent-Only Decoder: Original ARAD 1D CNN decoder reconstructs from the LSTM latent
   representation — no skip connections — forcing the model to learn
   genuine temporal patterns rather than shortcutting via direct features

Why this matters for radiation detection:
=========================================
- Background radiation varies with location, time, environment
- A spectrum that looks "normal" in isolation might be anomalous given recent history
- Temporal context helps distinguish gradual background changes from sudden anomalies
- Spectral features (peaks at specific energies) characterize different isotopes

Loss Functions:
===============
- JSD (Jensen-Shannon Divergence): Robust distribution comparison
- Chi-squared: Appropriate for Poisson-distributed count data
- MSE: Simple baseline loss

Data Processing:
================
- Per-spectrum L1 normalization: Each spectrum sums to 1, preserving spectral shape
- Optional data augmentation: Noise injection, spectral shift for robustness
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Union
import warnings
from pathlib import Path
import math
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. ARAD-LSTM detector requires PyTorch. "
        "Install with: pip install torch"
    )

from gammaflow.core.time_series import SpectralTimeSeries
from gammaflow.core.spectrum import Spectrum
from src.detectors.arad import ARADEncoderBlock, ARADDecoderBlock


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class SpectralAugmentation:
    """
    Data augmentation for gamma-ray spectra.
    
    Applies physically-motivated augmentations that preserve the statistical
    properties of gamma-ray detection while improving model robustness.
    
    Parameters
    ----------
    noise_level : float
        Standard deviation of Gaussian noise as fraction of signal (0.0 to 0.2)
    poisson_noise : bool
        If True, apply Poisson-like noise based on count statistics
    energy_shift : int
        Maximum bins to shift spectrum (simulates energy calibration drift)
    intensity_scale : tuple
        Range of intensity scaling factors (min, max), e.g., (0.8, 1.2)
    enabled : bool
        Whether augmentation is enabled (disable for validation/test)
    """
    
    def __init__(
        self,
        noise_level: float = 0.05,
        poisson_noise: bool = True,
        energy_shift: int = 2,
        intensity_scale: Tuple[float, float] = (0.9, 1.1),
        enabled: bool = True
    ):
        self.noise_level = noise_level
        self.poisson_noise = poisson_noise
        self.energy_shift = energy_shift
        self.intensity_scale = intensity_scale
        self.enabled = enabled
    
    def __call__(self, spectrum: np.ndarray) -> np.ndarray:
        """Apply augmentation to a single spectrum."""
        if not self.enabled:
            return spectrum
        
        augmented = spectrum.copy()
        
        # Intensity scaling (simulates distance/shielding variation)
        if self.intensity_scale[0] != 1.0 or self.intensity_scale[1] != 1.0:
            scale = np.random.uniform(*self.intensity_scale)
            augmented = augmented * scale
        
        # Poisson-like noise (realistic count statistics)
        if self.poisson_noise:
            # For count rate data, noise ~ sqrt(counts) / time
            # Approximate by adding noise proportional to sqrt(value)
            noise = np.random.normal(0, 1, augmented.shape) * np.sqrt(np.maximum(augmented, 0.1))
            augmented = np.maximum(augmented + noise * 0.1, 0)
        
        # Gaussian noise
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * np.mean(augmented), augmented.shape)
            augmented = np.maximum(augmented + noise, 0)
        
        # Energy shift (simulates calibration drift)
        if self.energy_shift > 0:
            shift = np.random.randint(-self.energy_shift, self.energy_shift + 1)
            if shift != 0:
                augmented = np.roll(augmented, shift)
                # Zero out wrapped values
                if shift > 0:
                    augmented[:shift] = 0
                else:
                    augmented[shift:] = 0
        
        return augmented


class PreprocessedRunDataset(Dataset):
    """
    Dataset backed by preprocessed per-run tensor files.

    Each run is stored as a .pt file containing:
        - spectra: torch.FloatTensor (n_spectra, n_bins) — L1-normalized (each row sums to 1)
        - timestamps, live_times, real_times, energy_edges (optional)

    Returns 2-tuple: (sequence, target).

    This dataset builds a lightweight index over runs and sequences, and
    optionally caches a configurable number of runs in memory.
    """

    def __init__(
        self,
        data_dir: str,
        run_ids: Optional[List[Any]],
        sequence_length: int,
        augmentation: Optional[SpectralAugmentation] = None,
        target_mode: str = 'next',
        cache_size_runs: int = 50
    ):
        self.data_dir = Path(data_dir)
        self.run_ids = run_ids
        self.sequence_length = sequence_length
        self.augmentation = augmentation
        self.target_mode = target_mode
        self.cache_size_runs = cache_size_runs

        from collections import OrderedDict
        self._cache = OrderedDict()

        self.run_files = self._resolve_run_files()
        self._build_index()

    def _resolve_run_files(self) -> List[Path]:
        if self.run_ids is None:
            return sorted(self.data_dir.glob("run*.pt"))

        run_files = []
        for run_id in self.run_ids:
            run_key = f"run{int(run_id)}.pt" if not str(run_id).startswith("run") else f"{run_id}.pt"
            run_files.append(self.data_dir / run_key)
        return run_files

    def _build_index(self):
        self.index = []  # list of (run_file, seq_start)
        self.run_metadata = {}

        for run_file in self.run_files:
            if not run_file.exists():
                continue

            data = torch.load(run_file, map_location='cpu', weights_only=False)
            spectra = data.get("spectra")
            if spectra is None:
                continue

            n_spectra = spectra.shape[0]
            self.run_metadata[run_file] = n_spectra

            if self.target_mode == 'next':
                n_sequences = n_spectra - self.sequence_length
            else:
                n_sequences = n_spectra - self.sequence_length + 1

            for seq_start in range(max(0, n_sequences)):
                self.index.append((run_file, seq_start))

        self.n_samples = len(self.index)

    def _load_run_spectra(self, run_file: Path) -> torch.Tensor:
        """Load L1-normalized spectra from a run file.
        
        Returns
        -------
        spectra : torch.Tensor
            L1-normalized spectra, shape (n_spectra, n_bins)
        """
        if run_file in self._cache:
            self._cache.move_to_end(run_file)
            return self._cache[run_file]

        data = torch.load(run_file, map_location='cpu', weights_only=False)
        spectra = data["spectra"].float()

        self._cache[run_file] = spectra
        while len(self._cache) > self.cache_size_runs:
            self._cache.popitem(last=False)

        return spectra

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        run_file, seq_start = self.index[idx]

        spectra = self._load_run_spectra(run_file)

        if self.target_mode == 'next':
            sequence = spectra[seq_start:seq_start + self.sequence_length].clone()
            target = spectra[seq_start + self.sequence_length].clone()
        else:
            sequence = spectra[seq_start:seq_start + self.sequence_length].clone()
            target = sequence[-1].clone()

        if self.augmentation is not None:
            sequence_np = sequence.numpy()
            for i in range(len(sequence_np)):
                sequence_np[i] = self.augmentation(sequence_np[i])
            sequence = torch.FloatTensor(sequence_np)

        return torch.FloatTensor(sequence), torch.FloatTensor(target)


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class SpectralFeatureExtractor(nn.Module):
    """
    1D CNN for extracting local spectral features.
    
    .. deprecated::
        Legacy feature extractor kept for backward compatibility with older
        checkpoints. Use ``ARADCNNSpectralFeatureExtractor`` (via
        ``use_arad_cnn=True``) for new models.
    
    Parameters
    ----------
    n_bins : int
        Number of energy bins per spectrum
    feature_dim : int
        Output feature dimension per timestep
    n_layers : int
        Number of convolutional layers
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        n_bins: int,
        feature_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        warnings.warn(
            "SpectralFeatureExtractor is deprecated and kept only for loading "
            "legacy checkpoints. Use ARADCNNSpectralFeatureExtractor "
            "(use_arad_cnn=True) for new models.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        self.n_bins = n_bins
        self.feature_dim = feature_dim
        
        layers = []
        in_channels = 1
        out_channels = 32
        
        for i in range(n_layers):
            # Kernel sizes decrease for each layer (capture different scales)
            kernel_size = max(7 - 2 * i, 3)
            
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
                nn.BatchNorm1d(out_channels),
                nn.Mish(),
                nn.Dropout(dropout)
            ])
            
            in_channels = out_channels
            if i < n_layers - 1:
                out_channels = min(out_channels * 2, 128)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling + linear projection to feature_dim
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(in_channels, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Mish()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spectral features.
        
        Parameters
        ----------
        x : torch.Tensor
            Normalized spectra, shape (batch, seq_len, n_bins)
        
        Returns
        -------
        features : torch.Tensor
            Spectral features, shape (batch, seq_len, feature_dim)
        """
        batch_size, seq_len, n_bins = x.shape
        
        # Reshape for conv: (batch * seq_len, 1, n_bins)
        x_flat = x.reshape(batch_size * seq_len, 1, n_bins)
        
        # Apply convolutions
        conv_out = self.conv_layers(x_flat)  # (batch * seq_len, channels, n_bins)
        
        # Global features (pooled)
        global_pooled = self.pool(conv_out).squeeze(-1)  # (batch * seq_len, channels)
        features = self.global_proj(global_pooled)  # (batch * seq_len, feature_dim)
        
        # Reshape back to sequence format
        features = features.reshape(batch_size, seq_len, -1)
        
        return features


class TemporalAttention(nn.Module):
    """
    Self-attention mechanism for temporal sequences.
    
    Allows the model to attend to relevant historical spectra when
    predicting the current/next spectrum. Uses causal masking for
    streaming applications.
    
    Parameters
    ----------
    hidden_size : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout rate
    causal : bool
        If True, applies causal masking (can only attend to past)
    """
    
    def __init__(
        self,
        hidden_size: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        causal: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = causal
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence, shape (batch, seq_len, hidden_size)
        
        Returns
        -------
        torch.Tensor
            Attended sequence, shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask if needed
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        return self.layer_norm(x + attn_output)


class SpectralDecoder(nn.Module):
    """
    Decoder for spectrum reconstruction from latent representation.
    
    .. deprecated::
        Legacy MLP decoder kept for backward compatibility with older
        checkpoints. Use ``ARADCNNSpectralDecoder`` (via
        ``use_arad_cnn=True``) for new models.
    
    Parameters
    ----------
    n_bins : int
        Number of energy bins in output spectrum
    latent_dim : int
        Latent representation dimension
    hidden_size : int
        Hidden layer size
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        n_bins: int,
        latent_dim: int,
        hidden_size: int,
        dropout: float = 0.2,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        warnings.warn(
            "SpectralDecoder is deprecated and kept only for loading legacy "
            "checkpoints. Use ARADCNNSpectralDecoder (use_arad_cnn=True) "
            "for new models.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        self.n_bins = n_bins
        self.output_activation = output_activation.lower()
        if self.output_activation not in ["sigmoid", "softmax"]:
            raise ValueError(
                f"output_activation must be 'sigmoid' or 'softmax', got '{output_activation}'"
            )
        
        # Decoder path: latent -> hidden layers -> spectrum
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_bins),
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to spectrum.
        
        Parameters
        ----------
        latent : torch.Tensor
            Latent representation, shape (batch, latent_dim)
        
        Returns
        -------
        torch.Tensor
            Reconstructed spectrum, shape (batch, n_bins)
        """
        logits = self.decoder_mlp(latent)
        if self.output_activation == "softmax":
            return F.softmax(logits, dim=-1)
        return torch.sigmoid(logits)


class ARADCNNSpectralFeatureExtractor(nn.Module):
    """
    ARAD-style 1D CNN encoder for per-spectrum feature extraction.

    Uses the original ARAD encoder block stack and projects each spectrum
    to a fixed-dimensional embedding for temporal modeling.
    """

    def __init__(
        self,
        n_bins: int,
        feature_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        if n_bins % 32 != 0:
            raise ValueError(
                f"n_bins must be divisible by 32 for ARAD CNN encoder/decoder, got {n_bins}"
            )

        self.n_bins = n_bins
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            ARADEncoderBlock(1, 8, 7, dropout),
            ARADEncoderBlock(8, 8, 5, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            ARADEncoderBlock(8, 8, 3, dropout),
            nn.Flatten(),
            nn.Linear(8 * (n_bins // 32), feature_dim),
            nn.Mish(),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Spectra, shape (batch, seq_len, n_bins)

        Returns
        -------
        torch.Tensor
            Features, shape (batch, seq_len, feature_dim)
        """
        batch_size, seq_len, n_bins = x.shape
        x_flat = x.reshape(batch_size * seq_len, 1, n_bins)
        features = self.encoder(x_flat)
        return features.reshape(batch_size, seq_len, self.feature_dim)


class ARADCNNSpectralDecoder(nn.Module):
    """
    ARAD-style CNN decoder for spectrum reconstruction from latent vectors.
    """

    def __init__(
        self,
        n_bins: int,
        latent_dim: int,
        dropout: float = 0.2,
        output_activation: str = "softmax",
    ):
        super().__init__()

        if n_bins % 32 != 0:
            raise ValueError(
                f"n_bins must be divisible by 32 for ARAD CNN encoder/decoder, got {n_bins}"
            )

        self.n_bins = n_bins
        self.output_activation = output_activation.lower()
        if self.output_activation not in ["sigmoid", "softmax"]:
            raise ValueError(
                f"output_activation must be 'sigmoid' or 'softmax', got '{output_activation}'"
            )

        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, 8 * (n_bins // 32)),
            nn.Mish(),
            nn.BatchNorm1d(8 * (n_bins // 32)),
        )

        self.decoder_body = nn.Sequential(
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 3, dropout),
            ARADDecoderBlock(8, 8, 5, dropout),
        )

        if self.output_activation == "sigmoid":
            self.output_block = ARADDecoderBlock(8, 1, 7, dropout, is_output=True)
            self.output_layer = None
        else:
            self.output_block = None
            # Softmax needs logits, so keep ARAD upsample+deconv shape path.
            self.output_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose1d(8, 1, 7, padding=3),
            )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        latent : torch.Tensor
            Latent representation, shape (batch, latent_dim)

        Returns
        -------
        torch.Tensor
            Reconstructed spectrum, shape (batch, n_bins)
        """
        decoded = self.decoder_linear(latent)
        decoded = decoded.view(decoded.size(0), 8, self.n_bins // 32)
        decoded = self.decoder_body(decoded)

        if self.output_block is not None:
            reconstructed = self.output_block(decoded)
            return reconstructed.squeeze(1)

        logits = self.output_layer(decoded).squeeze(1)
        return F.softmax(logits, dim=-1)


class TemporalLSTMAutoencoder(nn.Module):
    """
    Advanced Temporal LSTM Autoencoder for gamma-ray spectra.
    
    Architecture:
    1. Spectral Feature Extraction: ARAD 1D CNN encoder captures local spectral patterns
    2. Temporal LSTM Encoding: Processes the sequence of spectral features
    3. Temporal Attention: Self-attention for focusing on relevant history
    4. Latent Projection: Compress to latent representation
    5. Latent-Only Decoder: ARAD CNN decoder reconstructs from LSTM latent
    
    Parameters
    ----------
    n_bins : int
        Number of energy bins per spectrum
    hidden_size : int
        LSTM hidden state size
    latent_dim : int
        Size of the latent/bottleneck representation
    num_layers : int
        Number of stacked LSTM layers
    dropout : float
        Dropout rate
    bidirectional : bool
        If True, use bidirectional LSTM (better for offline analysis)
        If False, use unidirectional/causal LSTM (required for streaming)
    use_attention : bool
        Whether to use temporal attention mechanism
    num_attention_heads : int
        Number of attention heads (if using attention)
    use_arad_cnn : bool
        If True (default), use original ARAD CNN encoder/decoder blocks
        for per-spectrum feature extraction and reconstruction.
        If False, use the legacy ARAD-LSTM feature extractor + MLP decoder
        for backward compatibility with older checkpoints.
    """
    
    def __init__(
        self,
        n_bins: int,
        hidden_size: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = True,
        num_attention_heads: int = 4,
        output_activation: str = "softmax",
        use_arad_cnn: bool = True,
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.output_activation = output_activation.lower()
        self.use_arad_cnn = use_arad_cnn

        if self.use_arad_cnn and n_bins % 32 != 0:
            raise ValueError(
                f"n_bins must be divisible by 32 for ARAD CNN hybrid, got {n_bins}"
            )
        
        # 1. Spectral Feature Extraction
        if self.use_arad_cnn:
            self.feature_extractor = ARADCNNSpectralFeatureExtractor(
                n_bins=n_bins,
                feature_dim=hidden_size,
                dropout=dropout,
            )
        else:
            self.feature_extractor = SpectralFeatureExtractor(
                n_bins=n_bins,
                feature_dim=hidden_size,
                n_layers=3,
                dropout=dropout
            )
        
        # 2. Temporal LSTM Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Adjust for bidirectional
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # 3. Temporal Attention (optional)
        if use_attention:
            self.temporal_attention = TemporalAttention(
                hidden_size=lstm_output_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                causal=not bidirectional
            )
        
        # 4. Latent Space Projection
        self.to_latent = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # 5. Decoder (latent-only, no skip connections)
        if self.use_arad_cnn:
            self.decoder = ARADCNNSpectralDecoder(
                n_bins=n_bins,
                latent_dim=latent_dim,
                dropout=dropout,
                output_activation=self.output_activation,
            )
        else:
            self.decoder = SpectralDecoder(
                n_bins=n_bins,
                latent_dim=latent_dim,
                hidden_size=hidden_size,
                dropout=dropout,
                output_activation=self.output_activation,
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        """Initialize weights with proper initialization."""
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            # Match ARAD initialization for convolutional layers.
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
                    # Set forget gate bias to 1 for better gradient flow
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1)
    
    def _normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectra using per-spectrum L1 normalization.
        
        Data is already L1-normalized (each spectrum sums to 1).
        Just clamps to ensure valid [0, 1] range.
        
        Parameters
        ----------
        x : torch.Tensor
            Spectra, shape (batch, seq_len, n_bins) or (batch, n_bins)
        
        Returns
        -------
        normalized : torch.Tensor
            Clamped spectra in [0, 1]
        """
        return torch.clamp(x, 0.0, 1.0)
    
    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of spectra to latent representation.
        
        The latent captures the full temporal context needed for
        reconstruction. No skip connections are used — the decoder
        must reconstruct entirely from this latent.
        
        Parameters
        ----------
        sequence : torch.Tensor
            Input sequence, shape (batch, seq_len, n_bins)
        Returns
        -------
        latent : torch.Tensor
            Latent representation, shape (batch, latent_dim)
        """
        batch_size, seq_len, n_bins = sequence.shape

        # Normalize (L1-normalized data: clamp to [0, 1])
        normalized = self._normalize_batch(sequence)

        # 1. Extract spectral features
        features = self.feature_extractor(normalized)
        # features: (batch, seq_len, hidden_size)

        # 2. LSTM temporal encoding
        lstm_out, (h_n, c_n) = self.encoder_lstm(features)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        
        # 3. Temporal attention (if enabled)
        if self.use_attention:
            lstm_out = self.temporal_attention(lstm_out)
        
        # 4. Use the final timestep's representation
        temporal_context = lstm_out[:, -1, :]  # (batch, hidden_size * num_directions)
        
        # 5. Project to latent space
        latent = self.to_latent(temporal_context)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed spectrum.
        
        Parameters
        ----------
        latent : torch.Tensor
            Latent representation, shape (batch, latent_dim)
        
        Returns
        -------
        torch.Tensor
            Reconstructed spectrum, shape (batch, n_bins)
        """
        return self.decoder(latent)
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode sequence and reconstruct target spectrum.
        
        Parameters
        ----------
        sequence : torch.Tensor
            Input sequence of spectra, shape (batch, seq_len, n_bins)
        Returns
        -------
        torch.Tensor
            Reconstructed spectrum, shape (batch, n_bins)
        """
        latent = self.encode(sequence)
        reconstructed = self.decode(latent)
        return reconstructed


# =============================================================================
# DETECTOR CLASS
# =============================================================================

class ARADLSTMDetector:
    """
    Advanced Temporal LSTM-based Anomaly Detection for gamma-ray spectra.
    
    Uses a causal (unidirectional) or bidirectional LSTM with attention to model
    the temporal evolution of background spectra. Anomalies are detected when a 
    spectrum cannot be well-reconstructed given its recent temporal history.
    
    Key Features:
    - Spectral feature extraction via 1D CNN
    - Temporal attention for focusing on relevant history
    - Bidirectional option for offline analysis
    - Multiple loss functions (JSD, Chi-squared, MSE)
    - Data augmentation for robustness
    - Multi-run training from HDF5 files
    
    Parameters
    ----------
    sequence_length : int, default=10
        Number of consecutive spectra to use as temporal context
    hidden_size : int, default=128
        LSTM hidden state size
    latent_dim : int, default=32
        Dimensionality of the latent space
    num_layers : int, default=2
        Number of stacked LSTM layers
    dropout : float, default=0.2
        Dropout rate for regularization
    bidirectional : bool, default=False
        If True, use bidirectional LSTM (better for offline)
        If False, use causal LSTM (required for streaming)
    use_attention : bool, default=True
        Whether to use temporal attention mechanism
    num_attention_heads : int, default=4
        Number of attention heads
    batch_size : int, default=32
        Training batch size
    learning_rate : float, default=0.001
        Initial learning rate
    epochs : int, default=100
        Maximum number of training epochs
    l1_lambda : float, default=1e-5
        L1 regularization weight
    l2_lambda : float, default=1e-4
        L2 regularization weight (via AdamW)
    early_stopping_patience : int, default=15
        Patience for early stopping
    validation_split : float, default=0.2
        Fraction of training data to use for validation
    device : str, optional
        Device to use ('cuda', 'mps', 'cpu'). If None, auto-selects.
    threshold : float, optional
        Anomaly detection threshold
    aggregation_gap : float, default=2.0
        Time gap (seconds) for aggregating consecutive alarms
    min_training_samples : int, default=200
        Minimum number of training samples required
    loss_type : str, default='chi2'
        Loss function: 'chi2', 'jsd', or 'mse'
    gradient_clip : float, default=1.0
        Gradient clipping value for stability
    use_augmentation : bool, default=False
        Whether to use data augmentation during training
    target_mode : str, default='next'
        Target mode: 'next' (predict next spectrum after sequence) or
        'last' (reconstruct last spectrum in sequence).
        'next' is recommended for anomaly detection as it compares
        predicted vs actual next spectrum.
    output_activation : str, default='softmax'
        Output activation for decoder: 'softmax' or 'sigmoid'.
        'softmax' is recommended for L1-normalized data because it produces
        a valid probability distribution (sums to 1) by construction.
        'sigmoid' may be needed for backward compatibility with older
        checkpoints trained on max-normalized data.
    use_arad_cnn : bool, default=True
        If True, use original ARAD CNN encoder/decoder blocks as the
        spatial front/back ends around the temporal LSTM core.
        If False, use legacy ARAD-LSTM CNN/MLP modules for compatibility.
    verbose : bool, default=True
        Print training progress
    
    Examples
    --------
    >>> detector = ARADLSTMDetector(sequence_length=10, use_attention=True)
    >>> detector.fit(background_data)
    >>> detector.set_threshold_by_far(background_data, alarms_per_hour=0.5)
    >>> scores = detector.process_time_series(test_data)
    >>> print(f"Detected {len(detector.alarms)} anomalies")
    """
    
    def __init__(
        self,
        sequence_length: int = 10,
        hidden_size: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        use_attention: bool = True,
        num_attention_heads: int = 4,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 100,
        l1_lambda: float = 1e-5,
        l2_lambda: float = 1e-4,
        early_stopping_patience: int = 15,
        validation_split: float = 0.2,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        aggregation_gap: float = 2.0,
        min_training_samples: int = 200,
        loss_type: str = 'chi2',
        gradient_clip: float = 1.0,
        use_augmentation: bool = False,
        target_mode: str = 'next',
        output_activation: str = 'softmax',
        use_arad_cnn: bool = True,
        verbose: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("ARAD-LSTM requires PyTorch. Install with: pip install torch")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_attention_heads = num_attention_heads
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.threshold = threshold
        self.aggregation_gap = aggregation_gap
        self.min_training_samples = min_training_samples
        self.loss_type = loss_type.lower()
        self.gradient_clip = gradient_clip
        self.use_augmentation = use_augmentation
        self.target_mode = target_mode
        self.output_activation = output_activation.lower()
        self.use_arad_cnn = use_arad_cnn
        self.verbose = verbose
        
        if self.loss_type not in ['jsd', 'mse', 'chi2']:
            raise ValueError(f"loss_type must be 'jsd', 'chi2', or 'mse', got '{loss_type}'")
        if self.output_activation not in ['sigmoid', 'softmax']:
            raise ValueError(
                f"output_activation must be 'sigmoid' or 'softmax', got '{output_activation}'"
            )
        
        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        
        # Enable CUDA optimizations at initialization
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmul (Ampere+)
            torch.backends.cudnn.allow_tf32 = True  # TF32 for cuDNN (Ampere+)
        
        if self.verbose:
            print(f"ARAD-LSTM using device: {self.device}")
        
        self.model_ = None
        self.n_bins_ = None
        self.is_fitted_ = False
        self.training_history_ = {}
        
        # Detection state
        self.alarms: List[Dict[str, Any]] = []
        
        # Buffer for streaming detection
        self._spectrum_buffer: List[np.ndarray] = []
        
        # Augmentation config
        self.augmentation = SpectralAugmentation(
            noise_level=0.02,
            poisson_noise=True,
            energy_shift=1,
            intensity_scale=(0.95, 1.05),
            enabled=use_augmentation
        )

    @staticmethod
    def _format_run_key(run_id: Any) -> str:
        """Normalize run identifier to HDF5 run key (e.g., 'run0')."""
        if isinstance(run_id, str):
            return run_id if run_id.startswith("run") else f"run{run_id}"
        return f"run{int(run_id)}"

    @staticmethod
    def _convert_time_deltas(time_deltas: np.ndarray, time_units: str) -> np.ndarray:
        """Convert time deltas to seconds based on provided units."""
        units = time_units.lower()
        if units in ["us", "microsecond", "microseconds"]:
            return time_deltas.astype(np.float64) * 1e-6
        if units in ["ms", "millisecond", "milliseconds"]:
            return time_deltas.astype(np.float64) * 1e-3
        if units in ["s", "sec", "second", "seconds"]:
            return time_deltas.astype(np.float64)
        raise ValueError(f"Unsupported time_units '{time_units}'. Use 'us', 'ms', or 's'.")

    def get_gpu_info(self) -> dict:
        """
        Get GPU information and memory usage.
        
        Returns
        -------
        dict
            Dictionary with GPU info including:
            - device_name: GPU name or 'CPU'
            - cuda_available: Whether CUDA is available
            - memory_allocated: Current GPU memory in use (MB)
            - memory_reserved: Current GPU memory reserved (MB)
            - memory_total: Total GPU memory (MB)
            - utilization: Memory utilization percentage
        """
        info = {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
        }
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            info['device_name'] = torch.cuda.get_device_name(0)
            info['memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            info['memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            info['memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            info['utilization_pct'] = (info['memory_allocated_mb'] / info['memory_total_mb']) * 100
            info['cudnn_benchmark'] = torch.backends.cudnn.benchmark
            info['tf32_enabled'] = getattr(torch.backends.cuda.matmul, 'allow_tf32', False)
        else:
            info['device_name'] = 'CPU'
            info['memory_allocated_mb'] = 0
            info['memory_reserved_mb'] = 0
            info['memory_total_mb'] = 0
            info['utilization_pct'] = 0
        
        return info

    def print_gpu_status(self) -> None:
        """Print current GPU status and memory usage."""
        info = self.get_gpu_info()
        print(f"\n{'='*50}")
        print(f"GPU Status: {info['device_name']}")
        print(f"{'='*50}")
        print(f"  Device: {info['device']}")
        print(f"  CUDA Available: {info['cuda_available']}")
        
        if info['cuda_available'] and 'cuda' in str(self.device):
            print(f"  Memory Allocated: {info['memory_allocated_mb']:.1f} MB")
            print(f"  Memory Reserved: {info['memory_reserved_mb']:.1f} MB")
            print(f"  Memory Total: {info['memory_total_mb']:.1f} MB")
            print(f"  Utilization: {info['utilization_pct']:.1f}%")
            print(f"  cuDNN Benchmark: {info['cudnn_benchmark']}")
            print(f"  TF32 Enabled: {info['tf32_enabled']}")
        print(f"{'='*50}\n")

    def fit_from_preprocessed(
        self,
        data_dir: str,
        run_ids: Optional[List[Any]] = None,
        validation_split_runs: float = 0.2,
        cache_size_runs: int = 50,
        num_workers: int = 0,
        stats_path: Optional[str] = None
    ) -> 'ARADLSTMDetector':
        """
        Train using preprocessed per-run tensors.

        Parameters
        ----------
        data_dir : str
            Directory containing run*.pt files from preprocessing
        run_ids : List[Any], optional
            Optional list of run IDs to include
        validation_split_runs : float
            Fraction of runs for validation
        cache_size_runs : int
            Number of runs to keep in LRU cache
        num_workers : int
            DataLoader workers
        stats_path : str, optional
            Optional path to preprocess_stats.json
        """
        data_dir_path = Path(data_dir)
        if not data_dir_path.exists():
            raise FileNotFoundError(f"Preprocessed data directory not found: {data_dir}")

        if stats_path is None:
            stats_path = str(data_dir_path / "preprocess_stats.json")

        # Resolve run files
        if run_ids is None:
            run_files = sorted(data_dir_path.glob("run*.pt"))
        else:
            run_files = []
            for run_id in run_ids:
                run_key = f"run{int(run_id)}.pt" if not str(run_id).startswith("run") else f"{run_id}.pt"
                run_files.append(data_dir_path / run_key)
            run_files = [p for p in run_files if p.exists()]

        if not run_files:
            raise ValueError("No preprocessed run files found")

        # Deterministic shuffle and split runs (seed ensures reproducibility)
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(run_files))
        run_files = [run_files[i] for i in indices]
        n_runs = len(run_files)
        if n_runs == 1 or validation_split_runs <= 0:
            # Keep training possible for tiny experiments.
            train_run_files = run_files
            val_run_files = run_files
        else:
            # Ensure both splits are non-empty.
            n_val = int(n_runs * validation_split_runs)
            n_val = min(max(1, n_val), n_runs - 1)
            val_run_files = run_files[:n_val]
            train_run_files = run_files[n_val:]

        if self.verbose:
            print(f"Preprocessed runs: {len(run_files)}")
            print(f"  Training runs: {len(train_run_files)}, Validation runs: {len(val_run_files)}")

        # Determine n_bins and normalization mode
        first = torch.load(train_run_files[0], map_location='cpu', weights_only=False)
        spectra_first = first.get("spectra")
        if spectra_first is None:
            raise ValueError("Preprocessed file missing 'spectra'")
        self.n_bins_ = spectra_first.shape[1]

        if self.verbose:
            print(f"  Energy bins: {self.n_bins_}")

        augmentation = self.augmentation if self.use_augmentation else None

        train_dataset = PreprocessedRunDataset(
            data_dir=str(data_dir_path),
            run_ids=[p.stem.replace("run", "") for p in train_run_files],
            sequence_length=self.sequence_length,
            augmentation=augmentation,
            target_mode=self.target_mode,
            cache_size_runs=cache_size_runs
        )
        val_dataset = PreprocessedRunDataset(
            data_dir=str(data_dir_path),
            run_ids=[p.stem.replace("run", "") for p in val_run_files],
            sequence_length=self.sequence_length,
            augmentation=None,
            target_mode=self.target_mode,
            cache_size_runs=cache_size_runs
        )

        if self.verbose:
            print(f"  Training sequences: {len(train_dataset)}")
            print(f"  Validation sequences: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(num_workers > 0)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(num_workers > 0)
        )

        # Initialize model
        self.model_ = TemporalLSTMAutoencoder(
            n_bins=self.n_bins_,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention,
            num_attention_heads=self.num_attention_heads,
            output_activation=self.output_activation,
            use_arad_cnn=self.use_arad_cnn,
        ).to(self.device)

        return self._fit_with_dataloaders(train_loader, val_loader)

    def _fit_with_dataloaders(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> 'ARADLSTMDetector':
        """
        Core training loop using pre-built DataLoaders.
        
        Includes GPU optimizations:
        - Mixed precision training (AMP) for ~2x speedup on modern GPUs
        - Non-blocking data transfers
        - Gradient accumulation for effective larger batch sizes
        - torch.compile (PyTorch 2.0+) for kernel fusion
        
        Used by fit_from_preprocessed() for consistent training.
        """
        # Check for mixed precision support
        use_amp = (self.device.type == 'cuda' and 
                   hasattr(torch.cuda, 'amp') and 
                   torch.cuda.is_available())
        
        if use_amp:
            scaler = torch.amp.GradScaler('cuda')
            if self.verbose:
                print("  Mixed precision training: Enabled (AMP)")
        
        # Try to compile model for better performance (PyTorch 2.0+)
        compiled_model = self.model_
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            try:
                compiled_model = torch.compile(self.model_, mode='reduce-overhead')
                if self.verbose:
                    print("  torch.compile: Enabled")
            except Exception:
                compiled_model = self.model_
                if self.verbose:
                    print("  torch.compile: Not available")
        
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        self.training_history_ = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        if self.verbose:
            total_params = sum(p.numel() for p in self.model_.parameters())
            trainable = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
            print(f"Architecture: {'Bidirectional' if self.bidirectional else 'Causal'} LSTM")
            print(f"  Spatial backbone: {'ARAD CNN encoder/decoder' if self.use_arad_cnn else 'Legacy CNN/MLP'}")
            print(f"  Attention: {'Enabled' if self.use_attention else 'Disabled'}")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable:,}")
            print(f"  Batch size: {self.batch_size}")
            if self.device.type == 'cuda':
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print()
            print(f"Starting training for up to {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model_.train()
            train_losses = []
            
            for batch in train_loader:
                batch_x, batch_target = batch[0], batch[1]
                # Non-blocking transfer to GPU
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_target = batch_target.to(self.device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                if use_amp:
                    # Mixed precision forward pass
                    with torch.amp.autocast('cuda'):
                        reconstructed = compiled_model(batch_x)
                        
                        target_norm = self._normalize_spectrum(batch_target)
                        reconstructed_norm = self._normalize_spectrum(reconstructed)
                        
                        if self.loss_type == 'jsd':
                            loss = self._jsd_loss(target_norm, reconstructed_norm)
                        elif self.loss_type == 'chi2':
                            loss = self._chi2_loss(target_norm, reconstructed_norm)
                        else:
                            loss = self._mse_loss(target_norm, reconstructed_norm)
                        
                        if self.l1_lambda > 0:
                            l1_reg = sum(p.abs().sum() for p in self.model_.parameters())
                            loss = loss + self.l1_lambda * l1_reg
                    
                    # Scaled backward pass
                    scaler.scale(loss).backward()
                    
                    if self.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard precision forward pass
                    reconstructed = compiled_model(batch_x)
                    
                    target_norm = self._normalize_spectrum(batch_target)
                    reconstructed_norm = self._normalize_spectrum(reconstructed)
                    
                    if self.loss_type == 'jsd':
                        loss = self._jsd_loss(target_norm, reconstructed_norm)
                    elif self.loss_type == 'chi2':
                        loss = self._chi2_loss(target_norm, reconstructed_norm)
                    else:
                        loss = self._mse_loss(target_norm, reconstructed_norm)
                    
                    if self.l1_lambda > 0:
                        l1_reg = sum(p.abs().sum() for p in self.model_.parameters())
                        loss = loss + self.l1_lambda * l1_reg
                    
                    loss.backward()
                    
                    if self.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip)
                    
                    optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation phase
            self.model_.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch_x, batch_target = batch[0], batch[1]
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_target = batch_target.to(self.device, non_blocking=True)
                    
                    if use_amp:
                        with torch.amp.autocast('cuda'):
                            reconstructed = compiled_model(batch_x)
                            target_norm = self._normalize_spectrum(batch_target)
                            reconstructed_norm = self._normalize_spectrum(reconstructed)
                            
                            if self.loss_type == 'jsd':
                                loss = self._jsd_loss(target_norm, reconstructed_norm)
                            elif self.loss_type == 'chi2':
                                loss = self._chi2_loss(target_norm, reconstructed_norm)
                            else:
                                loss = self._mse_loss(target_norm, reconstructed_norm)
                    else:
                        reconstructed = compiled_model(batch_x)
                        target_norm = self._normalize_spectrum(batch_target)
                        reconstructed_norm = self._normalize_spectrum(reconstructed)
                        
                        if self.loss_type == 'jsd':
                            loss = self._jsd_loss(target_norm, reconstructed_norm)
                        elif self.loss_type == 'chi2':
                            loss = self._chi2_loss(target_norm, reconstructed_norm)
                        else:
                            loss = self._mse_loss(target_norm, reconstructed_norm)
                    
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            scheduler.step(avg_val_loss)
            
            self.training_history_['train_loss'].append(avg_train_loss)
            self.training_history_['val_loss'].append(avg_val_loss)
            self.training_history_['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)
            self.model_.to(self.device)
        
        if self.verbose:
            print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
        
        self.is_fitted_ = True
        return self
    
    def _normalize_spectrum(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize spectrum using per-spectrum L1 normalization.
        
        Data is already L1-normalized (each spectrum sums to 1).
        Just clamps to ensure valid [0, 1] range.
        """
        return torch.clamp(x, 0.0, 1.0)
    
    def _jsd_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Jensen-Shannon Divergence loss.
        
        JSD is a metric designed for probability distributions (values must sum to 1).
        Since our normalized spectra are intensity values in [0, 1] but don't sum to 1,
        we first convert them to proper probability distributions via L1 normalization.
        This ensures valid gradients and mathematically correct divergence computation.
        """
        # Clamp to ensure positive values before normalization
        y_true = torch.clamp(y_true, min=1e-10)
        y_pred = torch.clamp(y_pred, min=1e-10)
        
        # Convert to probability distributions by L1 normalization (sum to 1)
        # This is necessary because JSD is only mathematically valid for distributions
        p = y_true / y_true.sum(dim=-1, keepdim=True)
        q = y_pred / y_pred.sum(dim=-1, keepdim=True)
        
        m = 0.5 * (p + q)
        kld_pm = torch.sum(p * torch.log(p / m), dim=-1)
        kld_qm = torch.sum(q * torch.log(q / m), dim=-1)
        
        return torch.sqrt(torch.clamp(0.5 * (kld_pm + kld_qm), min=0.0)).mean()
    
    def _mse_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error loss."""
        return F.mse_loss(y_pred, y_true)
    
    def _chi2_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chi-squared loss for L1-normalized spectral data.
        
        Computes chi-squared statistic directly on the L1-normalized spectra,
        treating each bin's value as a probability.
        
        Parameters
        ----------
        y_true : torch.Tensor
            True L1-normalized spectra
        y_pred : torch.Tensor
            Predicted L1-normalized spectra
        
        Returns
        -------
        torch.Tensor
            Mean chi-squared loss
        """
        eps = 1e-8
        
        # Chi-squared: sum of (observed - expected)^2 / expected
        expected = torch.clamp(y_pred, min=eps)
        chi2 = torch.sum((y_true - expected) ** 2 / expected, dim=-1)
        
        # Normalize by number of bins for interpretability
        chi2 = chi2 / y_true.shape[-1]
        
        return chi2.mean()
    
    def _compute_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total loss (reconstruction + L1 regularization)."""
        y_true_norm = self._normalize_spectrum(y_true)
        y_pred_norm = self._normalize_spectrum(y_pred)
        
        if self.loss_type == 'jsd':
            recon_loss = self._jsd_loss(y_true_norm, y_pred_norm)
        elif self.loss_type == 'chi2':
            recon_loss = self._chi2_loss(y_true_norm, y_pred_norm)
        else:
            recon_loss = self._mse_loss(y_true_norm, y_pred_norm)
        
        # L1 regularization
        l1_norm = sum(param.abs().sum() for param in self.model_.parameters())
        
        return recon_loss + self.l1_lambda * l1_norm
    
    def score_sequence(
        self, 
        sequence: np.ndarray, 
        target: Optional[np.ndarray] = None,
    ) -> float:
        """
        Score a sequence by comparing model prediction to target spectrum.
        
        For target_mode='next': predicts next spectrum, compares to provided target.
        For target_mode='last': reconstructs last spectrum in sequence.
        
        Parameters
        ----------
        sequence : np.ndarray
            Sequence of spectra, shape (sequence_length, n_bins)
        target : np.ndarray, optional
            The actual target spectrum to compare against.
            Required for target_mode='next' (the actual next spectrum).
            For target_mode='last', this is ignored and sequence[-1] is used.
        Returns
        -------
        float
            Anomaly score (prediction error)
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before scoring")
        
        if len(sequence) != self.sequence_length:
            raise ValueError(
                f"Sequence has {len(sequence)} spectra, expected {self.sequence_length}"
            )
        
        if sequence.shape[1] != self.n_bins_:
            raise ValueError(
                f"Spectra have {sequence.shape[1]} bins, expected {self.n_bins_}"
            )
        
        # Determine target based on mode
        if self.target_mode == 'next':
            if target is None:
                raise ValueError(
                    "target must be provided for target_mode='next'. "
                    "Pass the actual next spectrum to compare against."
                )
            target_spectrum = target
        else:  # 'last'
            target_spectrum = sequence[-1]
        
        # Convert to tensors
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        target_tensor = torch.FloatTensor(target_spectrum).unsqueeze(0).to(self.device)

        # Get prediction/reconstruction
        self.model_.eval()
        with torch.no_grad():
            predicted = self.model_(x)
        
        # Compute score
        target_norm = self._normalize_spectrum(target_tensor)
        predicted_norm = self._normalize_spectrum(predicted)
        
        if self.loss_type == 'jsd':
            score = self._jsd_loss(target_norm, predicted_norm).item()
        elif self.loss_type == 'chi2':
            score = self._chi2_loss(target_norm, predicted_norm).item()
        else:
            score = self._mse_loss(target_norm, predicted_norm).item()
        
        return score

    def score_sequences_batch(
        self,
        sequences: np.ndarray,
        targets: Optional[np.ndarray] = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Score multiple sequences efficiently using batched GPU processing.
        
        This method avoids the overhead of repeated CPU-GPU transfers by
        processing sequences in large batches.
        
        Parameters
        ----------
        sequences : np.ndarray
            Array of sequences, shape (n_sequences, sequence_length, n_bins)
        targets : np.ndarray, optional
            Array of target spectra, shape (n_sequences, n_bins).
            Required for target_mode='next' (the actual next spectra).
            For target_mode='last', ignored and sequences[:, -1, :] is used.
        batch_size : int
            Number of sequences to process in each batch. Larger batches are
            more efficient but use more GPU memory. Default 256.
        
        Returns
        -------
        np.ndarray
            Anomaly scores for each sequence, shape (n_sequences,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before scoring")
        
        n_sequences = len(sequences)
        if n_sequences == 0:
            return np.array([])
        
        # Validate shapes
        if sequences.shape[1] != self.sequence_length:
            raise ValueError(
                f"Sequences have {sequences.shape[1]} time steps, expected {self.sequence_length}"
            )
        if sequences.shape[2] != self.n_bins_:
            raise ValueError(
                f"Spectra have {sequences.shape[2]} bins, expected {self.n_bins_}"
            )
        
        # Determine targets based on mode
        if self.target_mode == 'next':
            if targets is None:
                raise ValueError(
                    "targets must be provided for target_mode='next'. "
                    "Pass the actual next spectra to compare against."
                )
            target_spectra = targets
        else:  # 'last'
            target_spectra = sequences[:, -1, :]
        
        scores = np.zeros(n_sequences, dtype=np.float32)
        
        self.model_.eval()
        with torch.no_grad():
            for start_idx in range(0, n_sequences, batch_size):
                end_idx = min(start_idx + batch_size, n_sequences)
                batch = sequences[start_idx:end_idx]
                batch_targets = target_spectra[start_idx:end_idx]
                
                # Convert batch to tensors
                x = torch.FloatTensor(batch).to(self.device)
                targets_tensor = torch.FloatTensor(batch_targets).to(self.device)
                
                # Forward pass for entire batch
                predicted = self.model_(x)
                
                # Normalize
                targets_norm = self._normalize_spectrum(targets_tensor)
                predicted_norm = self._normalize_spectrum(predicted)
                
                # Compute per-sample scores (not reduced to single value)
                if self.loss_type == 'jsd':
                    batch_scores = self._jsd_loss_batch(targets_norm, predicted_norm)
                elif self.loss_type == 'chi2':
                    batch_scores = self._chi2_loss_batch(targets_norm, predicted_norm)
                else:
                    batch_scores = self._mse_loss_batch(targets_norm, predicted_norm)
                
                scores[start_idx:end_idx] = batch_scores.cpu().numpy()
        
        return scores

    def _jsd_loss_batch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Jensen-Shannon Divergence loss returning per-sample scores (not reduced).
        
        Parameters
        ----------
        y_true : torch.Tensor
            True normalized spectra, shape (batch, n_bins)
        y_pred : torch.Tensor
            Predicted normalized spectra, shape (batch, n_bins)
        
        Returns
        -------
        torch.Tensor
            Per-sample JSD scores, shape (batch,)
        """
        # Clamp to ensure positive values before normalization
        y_true = torch.clamp(y_true, min=1e-10)
        y_pred = torch.clamp(y_pred, min=1e-10)
        
        # Convert to probability distributions by L1 normalization (sum to 1)
        p = y_true / y_true.sum(dim=-1, keepdim=True)
        q = y_pred / y_pred.sum(dim=-1, keepdim=True)
        
        m = 0.5 * (p + q)
        kld_pm = torch.sum(p * torch.log(p / m), dim=-1)
        kld_qm = torch.sum(q * torch.log(q / m), dim=-1)
        
        return torch.sqrt(torch.clamp(0.5 * (kld_pm + kld_qm), min=0.0))

    def _mse_loss_batch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Mean Squared Error loss returning per-sample scores (not reduced).
        
        Parameters
        ----------
        y_true : torch.Tensor
            True normalized spectra, shape (batch, n_bins)
        y_pred : torch.Tensor
            Predicted normalized spectra, shape (batch, n_bins)
        
        Returns
        -------
        torch.Tensor
            Per-sample MSE scores, shape (batch,)
        """
        return torch.mean((y_pred - y_true) ** 2, dim=-1)

    def _chi2_loss_batch(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Chi-squared loss returning per-sample scores (not reduced).
        
        Computes chi-squared statistic directly on the L1-normalized spectra,
        treating each bin's value as a probability.
        
        Parameters
        ----------
        y_true : torch.Tensor
            True normalized spectra, shape (batch, n_bins)
        y_pred : torch.Tensor
            Predicted normalized spectra, shape (batch, n_bins)
        
        Returns
        -------
        torch.Tensor
            Per-sample chi-squared scores, shape (batch,)
        """
        eps = 1e-8
        
        # Chi-squared: sum of (observed - expected)^2 / expected
        expected = torch.clamp(y_pred, min=eps)
        chi2 = torch.sum((y_true - expected) ** 2 / expected, dim=-1)
        
        # Normalize by number of bins for interpretability
        chi2 = chi2 / y_true.shape[-1]
        
        return chi2
    
    def score_spectrum(self, spectrum: Spectrum) -> float:
        """
        Score a single spectrum using the internal buffer.
        
        Maintains an internal buffer of recent spectra for streaming use.
        Spectra should be added in chronological order.
        
        For target_mode='next' (default):
            The buffer holds sequence_length + 1 spectra.  The first L
            spectra are the context (input to the model) and the most
            recent spectrum is the *target* — the model predicts what it
            thinks the next spectrum should look like, and the anomaly
            score is how far the prediction is from what actually arrived.
        
        For target_mode='last':
            The buffer holds sequence_length spectra.  The model
            reconstructs the last spectrum in the window.
        
        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to score
        
        Returns
        -------
        float
            Anomaly score (or 0.0 if buffer not yet full)
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before scoring")
        
        # Convert to rate per bin
        counts = spectrum.counts
        time = spectrum.live_time if (spectrum.live_time is not None and not np.isnan(spectrum.live_time)) else spectrum.real_time
        spectrum_data = counts / time
        
        if len(spectrum_data) != self.n_bins_:
            raise ValueError(
                f"Spectrum has {len(spectrum_data)} bins, expected {self.n_bins_}"
            )
        
        # L1-normalize the spectrum (consistent with detect() and training)
        spec_sum = max(float(spectrum_data.sum()), 1e-10)
        spectrum_normalized = spectrum_data / spec_sum
        
        # Add to buffers
        self._spectrum_buffer.append(spectrum_normalized)
        
        # Buffer size depends on target_mode:
        #   'next': need sequence_length + 1 (L context spectra + 1 target)
        #   'last': need sequence_length
        buf_size = self.sequence_length + 1 if self.target_mode == 'next' else self.sequence_length
        
        # Trim buffer to required size
        if len(self._spectrum_buffer) > buf_size:
            self._spectrum_buffer = self._spectrum_buffer[-buf_size:]
        
        # If buffer not full yet, return 0 (can't score)
        if len(self._spectrum_buffer) < buf_size:
            return 0.0
        
        # Score using the buffer
        if self.target_mode == 'next':
            # First L spectra are context, last spectrum is the target
            sequence = np.array(self._spectrum_buffer[:-1])
            target = np.array(self._spectrum_buffer[-1])
            return self.score_sequence(sequence, target=target)
        else:
            sequence = np.array(self._spectrum_buffer)
            return self.score_sequence(sequence)
    
    def reset_buffer(self):
        """Reset the internal spectrum buffer for streaming detection."""
        self._spectrum_buffer = []
    
    def detect(
        self,
        time_series: SpectralTimeSeries,
        batch_size: int = 256
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect anomalies in a time series using batched GPU processing.
        
        This method uses efficient batched inference to process sliding windows
        in large batches, avoiding the overhead of repeated CPU-GPU transfers.
        For a 1-hour run at 10Hz (~36k spectra), this is ~50-100x faster than
        scoring each sequence individually.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to analyze
        batch_size : int
            Number of sequences to process per GPU batch. Larger values are
            more efficient but use more memory. Default 256.
        
        Returns
        -------
        scores : np.ndarray
            Anomaly scores for each spectrum (first sequence_length-1 are 0)
        alarms : List[Dict[str, Any]]
            List of alarm events
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before detection")
        
        if self.threshold is None:
            raise ValueError(
                "Threshold must be set before detection. "
                "Use set_threshold_by_far() or set threshold manually."
            )
        
        # Convert to rates and apply L1 normalization
        counts = time_series.counts
        times = time_series.live_times
        if times is None or times.dtype == object or (hasattr(times, 'dtype') and times.dtype in [np.float32, np.float64] and np.any(np.isnan(times))):
            times = time_series.real_times
        times = np.asarray(times, dtype=np.float64)
        if np.any(~np.isfinite(times)) or np.any(times <= 0):
            raise ValueError(
                "Invalid live/real times found. All acquisition times must be finite and > 0."
            )
        spectra = counts / times[:, np.newaxis]
        
        # Apply L1 normalization to each spectrum
        row_sums = spectra.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        spectra = spectra / row_sums
        
        n_spectra = time_series.n_spectra
        scores = np.zeros(n_spectra, dtype=np.float32)
        
        # Build sequences and targets based on target_mode
        if self.target_mode == 'next':
            # For 'next' mode: sequence of length L predicts spectrum at position L
            # We need at least sequence_length + 1 spectra
            n_windows = n_spectra - self.sequence_length
            if n_windows > 0:
                # Sequences: spectra[0:L], spectra[1:L+1], ...
                sequences = self._create_sliding_windows(spectra[:-1], self.sequence_length)
                # Targets: the actual next spectrum after each sequence
                targets = spectra[self.sequence_length:]
                
                # Score all sequences in batches
                window_scores = self.score_sequences_batch(
                    sequences,
                    targets=targets,
                    batch_size=batch_size,
                )
                
                # Place scores at the correct positions
                # Score at position i corresponds to the target at position i
                # which is spectrum[sequence_length + window_idx]
                scores[self.sequence_length:] = window_scores
        else:
            # For 'last' mode: sequence of length L reconstructs spectrum at position L-1
            n_windows = n_spectra - self.sequence_length + 1
            if n_windows > 0:
                sequences = self._create_sliding_windows(spectra, self.sequence_length)
                
                # Score all sequences in batches (targets=None uses last in sequence)
                window_scores = self.score_sequences_batch(
                    sequences,
                    targets=None,
                    batch_size=batch_size,
                )
                
                # Place scores at the correct positions
                scores[self.sequence_length - 1:] = window_scores
        
        # Detect alarms
        alarms = self._detect_alarms(scores, time_series.timestamps)
        
        return scores, alarms

    def _create_sliding_windows(
        self, 
        spectra: np.ndarray, 
        window_size: int
    ) -> np.ndarray:
        """
        Create sliding window views of spectra array efficiently.
        
        Uses numpy stride tricks for memory-efficient window creation when possible,
        falls back to explicit copy for safety with non-contiguous arrays.
        
        Parameters
        ----------
        spectra : np.ndarray
            Spectra array, shape (n_spectra, n_bins)
        window_size : int
            Size of each window
        
        Returns
        -------
        np.ndarray
            Sliding windows, shape (n_windows, window_size, n_bins)
        """
        n_spectra, n_bins = spectra.shape
        n_windows = n_spectra - window_size + 1
        
        # Ensure contiguous array for stride tricks
        if not spectra.flags['C_CONTIGUOUS']:
            spectra = np.ascontiguousarray(spectra)
        
        # Try using stride tricks for zero-copy windowing
        try:
            from numpy.lib.stride_tricks import sliding_window_view
            # sliding_window_view creates a view without copying data
            windows = sliding_window_view(spectra, window_size, axis=0)
            # Result shape is (n_windows, n_bins, window_size), need to transpose
            windows = np.moveaxis(windows, -1, 1)
            # Now shape is (n_windows, window_size, n_bins)
            # Must copy to ensure contiguous memory for PyTorch
            return np.ascontiguousarray(windows)
        except (ImportError, AttributeError):
            # Fallback for older numpy versions
            pass
        
        # Fallback: explicit loop (still faster than Python loop with GPU inference)
        windows = np.zeros((n_windows, window_size, n_bins), dtype=spectra.dtype)
        for i in range(n_windows):
            windows[i] = spectra[i:i + window_size]
        return windows

    def _detect_alarms(
        self, 
        scores: np.ndarray, 
        timestamps: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Detect alarm events from scores and timestamps.
        
        Groups consecutive above-threshold scores into alarm events with
        start time, end time, and peak score information.
        
        Parameters
        ----------
        scores : np.ndarray
            Anomaly scores for each time step
        timestamps : np.ndarray
            Timestamps for each time step
        
        Returns
        -------
        List[Dict[str, Any]]
            List of alarm events
        """
        # First pass: contiguous above-threshold segments only.
        raw_alarms: List[Dict[str, Any]] = []
        current_alarm: Optional[Dict[str, Any]] = None
        first_valid_idx = self.sequence_length if self.target_mode == 'next' else (self.sequence_length - 1)

        for i, (score, t) in enumerate(zip(scores, timestamps)):
            # Skip initial spectra that can't be scored
            if i < first_valid_idx:
                continue

            if score > self.threshold:
                if current_alarm is None:
                    current_alarm = {
                        'start_time': t,
                        'end_time': t,
                        'peak_score': score,
                        'peak_time': t,
                    }
                else:
                    current_alarm['end_time'] = t
                    if score > current_alarm['peak_score']:
                        current_alarm['peak_score'] = score
                        current_alarm['peak_time'] = t
            elif current_alarm is not None:
                raw_alarms.append(current_alarm)
                current_alarm = None

        if current_alarm is not None:
            raw_alarms.append(current_alarm)

        if not raw_alarms:
            return []

        # Second pass: merge nearby alarm segments separated by short gaps.
        merged_alarms = [raw_alarms[0].copy()]
        for alarm in raw_alarms[1:]:
            prev = merged_alarms[-1]
            if (alarm['start_time'] - prev['end_time']) < self.aggregation_gap:
                prev['end_time'] = alarm['end_time']
                if alarm['peak_score'] > prev['peak_score']:
                    prev['peak_score'] = alarm['peak_score']
                    prev['peak_time'] = alarm['peak_time']
            else:
                merged_alarms.append(alarm.copy())

        return merged_alarms
    
    def process_time_series(
        self, 
        time_series: SpectralTimeSeries,
        batch_size: int = 256
    ) -> np.ndarray:
        """
        Process an entire time series for anomaly detection.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to process
        batch_size : int
            Batch size for efficient GPU processing. Default 256.
        
        Returns
        -------
        np.ndarray
            Array of scores for each time point
        """
        scores, alarms = self.detect(time_series, batch_size=batch_size)
        self.alarms = alarms
        return scores
    
    def set_threshold_by_far(
        self,
        background_data: SpectralTimeSeries,
        alarms_per_hour: float,
        max_iterations: int = 20
    ) -> float:
        """
        Set detection threshold based on desired false alarm rate.
        
        Parameters
        ----------
        background_data : SpectralTimeSeries
            Background data to calibrate threshold
        alarms_per_hour : float
            Target false alarm rate (alarms per hour)
        max_iterations : int
            Maximum number of binary search iterations
        
        Returns
        -------
        float
            Calibrated threshold
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before setting threshold")
        
        # Convert to rates and apply L1 normalization
        counts = background_data.counts
        times = background_data.live_times
        if times is None or times.dtype == object or (hasattr(times, 'dtype') and times.dtype in [np.float32, np.float64] and np.any(np.isnan(times))):
            times = background_data.real_times
        times = np.asarray(times, dtype=np.float64)
        if np.any(~np.isfinite(times)) or np.any(times <= 0):
            raise ValueError(
                "Invalid live/real times found. All acquisition times must be finite and > 0."
            )
        spectra = counts / times[:, np.newaxis]
        
        # Apply L1 normalization to each spectrum
        row_sums = spectra.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        spectra = spectra / row_sums

        # Build sequences and targets based on target_mode
        if self.target_mode == 'next':
            n_windows = len(spectra) - self.sequence_length
            if n_windows > 0:
                sequences = self._create_sliding_windows(spectra[:-1], self.sequence_length)
                targets = spectra[self.sequence_length:]
                scores = self.score_sequences_batch(
                    sequences, targets=targets,
                    batch_size=256,
                )
            else:
                scores = np.array([])
            total_time_seconds = np.sum(times[self.sequence_length:])
        else:
            n_windows = len(spectra) - self.sequence_length + 1
            if n_windows > 0:
                sequences = self._create_sliding_windows(spectra, self.sequence_length)
                scores = self.score_sequences_batch(
                    sequences, targets=None,
                    batch_size=256,
                )
            else:
                scores = np.array([])
            total_time_seconds = np.sum(times[self.sequence_length - 1:])
        
        if scores.size == 0:
            min_required = self.sequence_length + 1 if self.target_mode == 'next' else self.sequence_length
            raise ValueError(
                f"Not enough spectra to calibrate threshold: got {len(spectra)}, need at least {min_required}."
            )

        total_time_hours = total_time_seconds / 3600.0
        
        if total_time_hours <= 0:
            raise ValueError(f"Invalid observation time: {total_time_hours} hours")
        
        # Build a full per-spectrum scores array for _detect_alarms.
        # Scores are computed once; only alarm detection is repeated per threshold.
        n_spectra = len(background_data.timestamps)
        full_scores = np.zeros(n_spectra, dtype=np.float32)
        if self.target_mode == 'next':
            full_scores[self.sequence_length:] = scores
        else:
            full_scores[self.sequence_length - 1:] = scores

        timestamps = background_data.timestamps

        # Binary search for threshold
        low_threshold = np.min(scores)
        high_threshold = np.max(scores) * 1.5
        
        best_threshold = float(np.median(scores))
        best_far_diff = float('inf')
        
        if self.verbose:
            print(f"\nCalibrating threshold for {alarms_per_hour:.2f} alarms/hour...")
            print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
            print(f"  Observation time: {total_time_hours:.2f} hours")
        
        for iteration in range(max_iterations):
            test_threshold = (low_threshold + high_threshold) / 2
            self.threshold = test_threshold
            
            alarms = self._detect_alarms(full_scores, timestamps)
            n_alarms = len(alarms)
            observed_far = n_alarms / total_time_hours
            
            far_diff = abs(observed_far - alarms_per_hour)
            
            if far_diff < best_far_diff:
                best_far_diff = far_diff
                best_threshold = test_threshold
            
            if self.verbose:
                print(f"  Iter {iteration + 1}: threshold={test_threshold:.4f}, "
                      f"FAR={observed_far:.2f}/hr (target: {alarms_per_hour:.2f})")
            
            if observed_far > alarms_per_hour:
                low_threshold = test_threshold
            else:
                high_threshold = test_threshold
            
            if far_diff < 0.01:
                break
        
        self.threshold = best_threshold
        self.alarms = self._detect_alarms(full_scores, timestamps)
        final_far = len(self.alarms) / total_time_hours
        
        if self.verbose:
            print(f"\nFinal threshold: {self.threshold:.4f}")
            print(f"Achieved FAR: {final_far:.2f} alarms/hour")
        
        return self.threshold
    
    def save(self, path: str):
        """Save trained model to file."""
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted before saving")
        
        save_dict = {
            'model_state': self.model_.state_dict(),
            'n_bins': self.n_bins_,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'use_attention': self.use_attention,
            'num_attention_heads': self.num_attention_heads,
            'output_activation': self.output_activation,
            'use_arad_cnn': self.use_arad_cnn,
            'threshold': self.threshold,
            'loss_type': self.loss_type,
            'target_mode': self.target_mode,
            'training_history': self.training_history_,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.n_bins_ = checkpoint['n_bins']
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_size = checkpoint['hidden_size']
        self.latent_dim = checkpoint['latent_dim']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.bidirectional = checkpoint.get('bidirectional', False)
        self.use_attention = checkpoint.get('use_attention', True)
        self.num_attention_heads = checkpoint.get('num_attention_heads', 4)
        self.output_activation = checkpoint.get('output_activation', 'sigmoid').lower()
        # Default ARAD-CNN path to True, but detect legacy checkpoints that
        # were saved before this flag existed and still use the old CNN/MLP stack.
        _legacy_feature_keys = (
            'feature_extractor.conv_layers',
            'feature_extractor.global_proj',
            'decoder.decoder_mlp',
        )
        _has_legacy_modules = any(
            any(k.startswith(prefix) for prefix in _legacy_feature_keys)
            for k in checkpoint['model_state']
        )
        self.use_arad_cnn = checkpoint.get('use_arad_cnn', not _has_legacy_modules)
        self.threshold = checkpoint['threshold']
        self.loss_type = checkpoint['loss_type']
        self.target_mode = checkpoint.get('target_mode', 'next')
        self.training_history_ = checkpoint['training_history']
        
        self.model_ = TemporalLSTMAutoencoder(
            n_bins=self.n_bins_,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            use_attention=self.use_attention,
            num_attention_heads=self.num_attention_heads,
            output_activation=self.output_activation,
            use_arad_cnn=self.use_arad_cnn,
        ).to(self.device)

        self.model_.load_state_dict(checkpoint['model_state'])
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"Model loaded from {path}")
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        if not self.training_history_:
            raise RuntimeError("No training history available")
        return self.training_history_
    
    def get_latent_representation(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:
        """
        Get the latent representation of a spectrum sequence.
        
        Parameters
        ----------
        sequence : np.ndarray
            Sequence of spectra, shape (sequence_length, n_bins)
        Returns
        -------
        np.ndarray
            Latent representation, shape (latent_dim,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted first")
        
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            latent = self.model_.encode(x)
        
        return latent.cpu().numpy().squeeze()
    
    def reconstruct_spectrum(
        self, 
        sequence: np.ndarray,
    ) -> np.ndarray:
        """
        Predict / reconstruct a spectrum from a sequence of input spectra.
        
        For target_mode='next' (default):
            Returns the model's *prediction* of the spectrum that should
            follow the input sequence.  Compare this to the actual next
            spectrum to obtain an anomaly score.
        
        For target_mode='last':
            Returns the model's reconstruction of the last spectrum in
            the input sequence.
        
        Parameters
        ----------
        sequence : np.ndarray
            Sequence of spectra, shape (sequence_length, n_bins)
        Returns
        -------
        np.ndarray
            Predicted (next) or reconstructed (last) L1-normalized
            spectrum, shape (n_bins,).
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted first")
        
        # Apply L1 normalization to input sequence
        row_sums = sequence.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        x_normalized = sequence / row_sums
        x = torch.FloatTensor(x_normalized).unsqueeze(0).to(self.device)

        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(x)
        
        return reconstructed.cpu().numpy().squeeze()
    
    def get_reconstruction_error_map(
        self, 
        sequence: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Get per-bin reconstruction error for anomaly localization.
        
        Useful for identifying which energy bins contribute most to the
        anomaly score, potentially revealing the isotope signature.
        
        For target_mode='next':
            The model predicts the spectrum that *follows* the input
            sequence.  Pass the actual next spectrum as ``target`` so
            the error map shows where prediction differs from reality.
        
        For target_mode='last':
            The model reconstructs the last spectrum in the sequence.
            ``target`` is ignored; ``sequence[-1]`` is used.
        
        Parameters
        ----------
        sequence : np.ndarray
            Sequence of spectra, shape (sequence_length, n_bins)
        target : np.ndarray, optional
            The actual next spectrum, shape (n_bins,).
            Required for target_mode='next'.
            Ignored for target_mode='last'.
        Returns
        -------
        np.ndarray
            Per-bin squared error, shape (n_bins,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Detector must be fitted first")
        
        # Determine reference spectrum based on target_mode
        if self.target_mode == 'next':
            if target is None:
                raise ValueError(
                    "target must be provided for target_mode='next'. "
                    "Pass the actual next spectrum to compare against."
                )
            reference = target
        else:
            reference = sequence[-1]
        
        # Apply L1 normalization to input sequence
        row_sums = sequence.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        x_normalized = sequence / row_sums
        x = torch.FloatTensor(x_normalized).unsqueeze(0).to(self.device)

        # L1-normalize the reference spectrum
        ref_sum = reference.sum()
        if ref_sum == 0:
            ref_sum = 1.0
        reference_norm = reference / ref_sum
        
        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(x)
        
        # Get reconstructed normalized values
        recon_norm = reconstructed.cpu().numpy().squeeze()
        
        # Per-bin squared error
        error_map = (reference_norm - recon_norm) ** 2
        
        return error_map