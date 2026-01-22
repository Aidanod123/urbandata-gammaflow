"""
ARAD-LSTM: Temporal LSTM-based Anomaly Detection for gamma-ray spectra.

Uses a CAUSAL (unidirectional) LSTM to model temporal sequences of spectra, 
where each timestep is a COMPLETE spectrum. The model learns normal temporal 
evolution of background radiation and detects anomalies when the current 
spectrum deviates from what would be expected given the recent history.

Key concept:
- Input: Sequence of recent spectra [s_{t-k}, ..., s_{t-1}, s_t]
- Each spectrum is a full vector of n_bins energy channels
- LSTM processes these temporally (only past informs present - CAUSAL)
- Output: Reconstruction of current spectrum s_t based on temporal context

Why this matters for radiation detection:
- Background radiation varies with location, time, environment
- A spectrum that looks "normal" in isolation might be anomalous given recent history
- Temporal context helps distinguish gradual background changes from sudden anomalies
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import warnings
from pathlib import Path

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


class TemporalSpectraDataset(Dataset):
    """
    Dataset for temporal sequences of spectra.
    
    Creates sliding windows of consecutive spectra for training the LSTM.
    Each sample is a sequence of spectra where we want to reconstruct the
    final spectrum given the temporal context.
    
    Parameters
    ----------
    spectra : np.ndarray
        Array of spectra, shape (n_spectra, n_bins)
    sequence_length : int
        Number of consecutive spectra in each sequence
    """
    
    def __init__(self, spectra: np.ndarray, sequence_length: int):
        self.spectra = spectra
        self.sequence_length = sequence_length
        # Need sequence_length spectra for input + 1 for target (no overlap)
        self.n_samples = len(spectra) - sequence_length
        
        if self.n_samples <= 0:
            raise ValueError(
                f"Not enough spectra ({len(spectra)}) for sequence length {sequence_length} + 1 target"
            )
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a sequence of spectra.
        
        Returns
        -------
        sequence : torch.Tensor
            Shape (sequence_length, n_bins) - the input sequence of past spectra
        target : torch.Tensor
            Shape (n_bins,) - the NEXT spectrum to predict (strictly after sequence)
        """
        # Input: spectra [idx, idx+1, ..., idx+sequence_length-1]
        sequence = self.spectra[idx:idx + self.sequence_length]
        # Target: spectrum at idx+sequence_length (no overlap with input)
        target = self.spectra[idx + self.sequence_length]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


class TemporalLSTMAutoencoder(nn.Module):
    """
    Temporal LSTM autoencoder for sequences of gamma-ray spectra.
    
    Architecture:
    - Input: Sequence of spectra [s_{t-k}, ..., s_{t-1}, s_t], each spectrum is n_bins
    - Encoder: Unidirectional (CAUSAL) LSTM processes the temporal sequence
    - Latent: Final hidden state captures temporal context
    - Decoder: MLP reconstructs the current spectrum s_t from temporal context
    
    This is CAUSAL - only past and current spectra are used, never future.
    Suitable for real-time streaming detection.
    
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
    """
    
    def __init__(
        self,
        n_bins: int,
        hidden_size: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2,
        global_log_max: float = 1.0
    ):
        super().__init__()
        
        self.n_bins = n_bins
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        # Global max of log1p-transformed training data for normalization
        self.register_buffer('global_log_max', torch.tensor(global_log_max, dtype=torch.float32))
        
        # Input projection: reduce spectrum dimensionality before LSTM
        self.input_projection = nn.Sequential(
            nn.Linear(n_bins, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        
        # Temporal LSTM encoder - UNIDIRECTIONAL (causal)
        # Each timestep is a full spectrum, LSTM learns temporal dynamics
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # CAUSAL - only uses past, not future
        )
        
        # Latent space projection from final hidden state
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Mish()
        )
        
        # Decoder: MLP to reconstruct spectrum from latent temporal context
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_bins),
            nn.Sigmoid()  # Output in [0, 1] for normalized spectra
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        """Initialize weights with proper initialization."""
        if isinstance(module, nn.Linear):
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
        Normalize spectra using log1p transform + global scale.
        
        This preserves intensity information by using a fixed global max
        from training data rather than per-spectrum normalization.
        
        Parameters
        ----------
        x : torch.Tensor
            Spectra, shape (batch, seq_len, n_bins) or (batch, n_bins)
        
        Returns
        -------
        normalized : torch.Tensor
            Log-transformed and globally scaled spectra in [0, 1]
        """
        # log1p compresses dynamic range while preserving intensity differences
        log_x = torch.log1p(x)
        # Scale by global max from training (intensity preserved across spectra)
        normalized = log_x / (self.global_log_max + 1e-8)
        # Clamp to [0, 1] in case test data exceeds training max
        normalized = torch.clamp(normalized, 0.0, 1.0)
        return normalized
    
    def encode(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of spectra to latent representation.
        
        Parameters
        ----------
        sequence : torch.Tensor
            Input sequence, shape (batch, seq_len, n_bins)
        
        Returns
        -------
        torch.Tensor
            Latent representation, shape (batch, latent_dim)
        """
        batch_size, seq_len, n_bins = sequence.shape
        
        # Normalize using log1p + global scale (preserves intensity)
        normalized = self._normalize_batch(sequence)
        
        # Project input: (batch, seq_len, n_bins) -> (batch, seq_len, hidden_size)
        projected = self.input_projection(normalized)
        
        # Encode with LSTM: process temporal sequence
        # output: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size) - final hidden state
        output, (h_n, c_n) = self.encoder_lstm(projected)
        
        # Use the final hidden state from the last layer as temporal context
        # h_n[-1] shape: (batch, hidden_size)
        temporal_context = h_n[-1]
        
        # Project to latent space
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
        Forward pass: encode sequence and reconstruct final spectrum.
        
        Parameters
        ----------
        sequence : torch.Tensor
            Input sequence of spectra, shape (batch, seq_len, n_bins)
        
        Returns
        -------
        torch.Tensor
            Reconstructed final spectrum, shape (batch, n_bins)
        """
        latent = self.encode(sequence)
        reconstructed = self.decode(latent)
        return reconstructed


class ARADLSTMDetector:
    """
    Temporal LSTM-based Anomaly Detection for gamma-ray spectra.
    
    Uses a causal (unidirectional) LSTM to model the temporal evolution of 
    background spectra. Anomalies are detected when a spectrum cannot be 
    well-reconstructed given its recent temporal history.
    
    Key concept: The model learns "given the last N background spectra, 
    what should the current spectrum look like?" Deviations indicate anomalies.
    
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
    batch_size : int, default=32
        Training batch size
    learning_rate : float, default=0.001
        Initial learning rate
    epochs : int, default=100
        Maximum number of training epochs
    l1_lambda : float, default=1e-4
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
    loss_type : str, default='jsd'
        Loss function: 'jsd' (Jensen-Shannon Divergence) or 'mse'
    gradient_clip : float, default=1.0
        Gradient clipping value for stability
    verbose : bool, default=True
        Print training progress
    
    Examples
    --------
    >>> detector = ARADLSTMDetector(sequence_length=10)
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
        batch_size: int = 32,
        learning_rate: float = 0.001,
        epochs: int = 100,
        l1_lambda: float = 1e-4,
        l2_lambda: float = 1e-4,
        early_stopping_patience: int = 15,
        validation_split: float = 0.2,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        aggregation_gap: float = 2.0,
        min_training_samples: int = 200,
        loss_type: str = 'jsd',
        gradient_clip: float = 1.0,
        verbose: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("ARAD-LSTM requires PyTorch. Install with: pip install torch")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
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
        self.verbose = verbose
        
        if self.loss_type not in ['jsd', 'mse']:
            raise ValueError(f"loss_type must be 'jsd' or 'mse', got '{loss_type}'")
        
        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        
        if self.verbose:
            print(f"ARAD-LSTM using device: {self.device}")
        
        self.model_ = None
        self.n_bins_ = None
        self.is_fitted_ = False
        self.training_history_ = {}
        # Global max of log1p-transformed training data
        self.global_log_max_ = None
        
        # Detection state
        self.alarms: List[Dict[str, Any]] = []
        
        # Buffer for streaming detection
        self._spectrum_buffer: List[np.ndarray] = []
    
    def _normalize_spectrum(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize spectrum using log1p + global scale (preserves intensity)."""
        if self.global_log_max_ is None:
            raise RuntimeError("Detector must be fitted before normalizing")
        log_x = torch.log1p(x)
        normalized = log_x / (self.global_log_max_ + 1e-8)
        return torch.clamp(normalized, 0.0, 1.0)
    
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
        
        return torch.sqrt(0.5 * (kld_pm + kld_qm)).mean()
    
    def _mse_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Mean Squared Error loss."""
        return F.mse_loss(y_pred, y_true)
    
    def _compute_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute total loss (reconstruction + L1 regularization)."""
        y_true_norm = self._normalize_spectrum(y_true)
        y_pred_norm = self._normalize_spectrum(y_pred)
        
        if self.loss_type == 'jsd':
            recon_loss = self._jsd_loss(y_true_norm, y_pred_norm)
        else:
            recon_loss = self._mse_loss(y_true_norm, y_pred_norm)
        
        # L1 regularization
        l1_norm = sum(param.abs().sum() for param in self.model_.parameters())
        
        return recon_loss + self.l1_lambda * l1_norm
    
    def fit(
        self,
        background_training: SpectralTimeSeries,
        validation_data: Optional[SpectralTimeSeries] = None
    ) -> 'ARADLSTMDetector':
        """
        Train the ARAD-LSTM detector on background spectra.
        
        The training data should be CONSECUTIVE spectra to preserve temporal ordering.
        
        Parameters
        ----------
        background_training : SpectralTimeSeries
            Background spectra for training (must be temporally ordered)
        validation_data : SpectralTimeSeries, optional
            Validation data. If None, uses validation_split of training data.
        
        Returns
        -------
        self
            Fitted detector
        """
        # Extract count rate data
        counts = background_training.counts
        times = background_training.live_times
        if times is None or times.dtype == object or (hasattr(times, 'dtype') and times.dtype in [np.float32, np.float64] and np.any(np.isnan(times))):
            times = background_training.real_times
        
        training_spectra = counts / times[:, np.newaxis]
        
        if training_spectra.shape[0] < self.min_training_samples:
            raise ValueError(
                f"Need at least {self.min_training_samples} training samples, "
                f"got {training_spectra.shape[0]}"
            )
        
        if training_spectra.shape[0] < self.sequence_length + 10:
            raise ValueError(
                f"Need at least {self.sequence_length + 10} spectra for "
                f"sequence_length={self.sequence_length}"
            )
        
        self.n_bins_ = training_spectra.shape[1]
        
        # Compute global max of log1p-transformed training data for normalization
        # This preserves intensity information across all spectra
        self.global_log_max_ = float(np.log1p(training_spectra).max())
        if self.verbose:
            print(f"Global log1p max (for normalization): {self.global_log_max_:.4f}")
        
        # Split into train/validation - PRESERVE TEMPORAL ORDER
        # Use first portion for training, last portion for validation
        if validation_data is None:
            n_train = int(len(training_spectra) * (1 - self.validation_split))
            train_data = training_spectra[:n_train]
            val_data = training_spectra[n_train:]
        else:
            train_data = training_spectra
            val_counts = validation_data.counts
            val_times = validation_data.live_times
            if val_times is None or val_times.dtype == object or (hasattr(val_times, 'dtype') and val_times.dtype in [np.float32, np.float64] and np.any(np.isnan(val_times))):
                val_times = validation_data.real_times
            val_data = val_counts / val_times[:, np.newaxis]
        
        if self.verbose:
            print(f"Training on {len(train_data)} spectra, validating on {len(val_data)}")
            print(f"Sequence length: {self.sequence_length} spectra")
            print(f"Loss function: {self.loss_type.upper()}")
            print(f"Architecture: Causal (unidirectional) Temporal LSTM")
            print(f"  Hidden size: {self.hidden_size}")
            print(f"  Latent dim: {self.latent_dim}")
            print(f"  Num layers: {self.num_layers}")
            print(f"  Spectrum bins: {self.n_bins_}")
        
        # Create datasets with sliding windows
        train_dataset = TemporalSpectraDataset(train_data, self.sequence_length)
        val_dataset = TemporalSpectraDataset(val_data, self.sequence_length)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,  # Shuffle sequences, but each sequence preserves temporal order
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            shuffle=False
        )
        
        if self.verbose:
            print(f"  Training sequences: {len(train_dataset)}")
            print(f"  Validation sequences: {len(val_dataset)}")
        
        # Initialize model
        self.model_ = TemporalLSTMAutoencoder(
            n_bins=self.n_bins_,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            global_log_max=self.global_log_max_
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model_.parameters())
        trainable_params = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        if self.verbose:
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
            eps=1e-8
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        if self.verbose:
            print(f"\nStarting training for up to {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model_.train()
            train_loss = 0.0
            
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.model_(sequences)
                loss = self._compute_loss(targets, reconstructed)
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.gradient_clip)
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model_.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    reconstructed = self.model_(sequences)
                    loss = self._compute_loss(targets, reconstructed)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            
            if self.verbose:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{self.epochs} - "
                      f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, "
                      f"LR: {current_lr:.2e}")
            
            # Early stopping with model checkpointing
            if avg_val_loss < best_val_loss - 1e-5:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model_.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
        
        self.training_history_ = {
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
        
        return self
    
    def score_sequence(self, sequence: np.ndarray) -> float:
        """
        Score a sequence of spectra (reconstruction error of final spectrum).
        
        Parameters
        ----------
        sequence : np.ndarray
            Sequence of spectra, shape (sequence_length, n_bins)
        
        Returns
        -------
        float
            Anomaly score for the final spectrum in the sequence
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
        
        # Convert to tensor: (1, seq_len, n_bins)
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        target = torch.FloatTensor(sequence[-1]).unsqueeze(0).to(self.device)
        
        # Get reconstruction
        self.model_.eval()
        with torch.no_grad():
            reconstructed = self.model_(x)
        
        # Compute score
        target_norm = self._normalize_spectrum(target)
        reconstructed_norm = self._normalize_spectrum(reconstructed)
        
        if self.loss_type == 'jsd':
            score = self._jsd_loss(target_norm, reconstructed_norm).item()
        else:
            score = self._mse_loss(target_norm, reconstructed_norm).item()
        
        return score
    
    def score_spectrum(self, spectrum: Spectrum) -> float:
        """
        Score a single spectrum using the internal buffer.
        
        NOTE: This method maintains an internal buffer of recent spectra.
        For proper temporal scoring, spectra should be added in order.
        
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
        
        # Extract count rate
        counts = spectrum.counts
        time = spectrum.live_time if (spectrum.live_time is not None and not np.isnan(spectrum.live_time)) else spectrum.real_time
        spectrum_data = counts / time
        
        if len(spectrum_data) != self.n_bins_:
            raise ValueError(
                f"Spectrum has {len(spectrum_data)} bins, expected {self.n_bins_}"
            )
        
        # Add to buffer
        self._spectrum_buffer.append(spectrum_data)
        
        # Keep only the most recent spectra
        if len(self._spectrum_buffer) > self.sequence_length:
            self._spectrum_buffer = self._spectrum_buffer[-self.sequence_length:]
        
        # If buffer not full, return 0 (can't score yet)
        if len(self._spectrum_buffer) < self.sequence_length:
            return 0.0
        
        # Score using the full buffer
        sequence = np.array(self._spectrum_buffer)
        return self.score_sequence(sequence)
    
    def reset_buffer(self):
        """Reset the internal spectrum buffer for streaming detection."""
        self._spectrum_buffer = []
    
    def detect(
        self,
        time_series: SpectralTimeSeries
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detect anomalies in a time series.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to analyze
        
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
        
        # Extract count rates
        counts = time_series.counts
        times = time_series.live_times
        if times is None or times.dtype == object or (hasattr(times, 'dtype') and times.dtype in [np.float32, np.float64] and np.any(np.isnan(times))):
            times = time_series.real_times
        spectra = counts / times[:, np.newaxis]
        
        # Score all spectra using sliding window
        scores = np.zeros(time_series.n_spectra)
        
        self.model_.eval()
        with torch.no_grad():
            for i in range(self.sequence_length - 1, time_series.n_spectra):
                sequence = spectra[i - self.sequence_length + 1:i + 1]
                scores[i] = self.score_sequence(sequence)
        
        # Detect alarms
        alarms = []
        in_alarm = False
        alarm_start = None
        alarm_scores = []
        alarm_times = []
        
        timestamps = time_series.timestamps
        
        for i, (score, t) in enumerate(zip(scores, timestamps)):
            # Skip initial spectra that can't be scored
            if i < self.sequence_length - 1:
                continue
            
            if score > self.threshold:
                if not in_alarm:
                    in_alarm = True
                    alarm_start = t
                    alarm_scores = [score]
                    alarm_times = [t]
                else:
                    alarm_scores.append(score)
                    alarm_times.append(t)
            else:
                if in_alarm:
                    if i < len(timestamps) - 1 and (timestamps[i + 1] - alarm_times[-1]) < self.aggregation_gap:
                        alarm_scores.append(score)
                        alarm_times.append(t)
                    else:
                        peak_idx = np.argmax(alarm_scores)
                        alarms.append({
                            'start_time': alarm_start,
                            'end_time': alarm_times[-1],
                            'peak_score': alarm_scores[peak_idx],
                            'peak_time': alarm_times[peak_idx]
                        })
                        in_alarm = False
        
        if in_alarm:
            peak_idx = np.argmax(alarm_scores)
            alarms.append({
                'start_time': alarm_start,
                'end_time': alarm_times[-1],
                'peak_score': alarm_scores[peak_idx],
                'peak_time': alarm_times[peak_idx]
            })
        
        return scores, alarms
    
    def process_time_series(self, time_series: SpectralTimeSeries) -> np.ndarray:
        """
        Process an entire time series for anomaly detection.
        
        Parameters
        ----------
        time_series : SpectralTimeSeries
            Time series to process
        
        Returns
        -------
        np.ndarray
            Array of scores for each time point
        """
        scores, alarms = self.detect(time_series)
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
        
        # Extract count rates and score all sequences
        counts = background_data.counts
        times = background_data.live_times
        if times is None or times.dtype == object or (hasattr(times, 'dtype') and times.dtype in [np.float32, np.float64] and np.any(np.isnan(times))):
            times = background_data.real_times
        spectra = counts / times[:, np.newaxis]
        
        # Score all valid positions
        scores = []
        for i in range(self.sequence_length - 1, len(spectra)):
            sequence = spectra[i - self.sequence_length + 1:i + 1]
            scores.append(self.score_sequence(sequence))
        scores = np.array(scores)
        
        total_time_seconds = np.sum(background_data.real_times[self.sequence_length - 1:])
        total_time_hours = total_time_seconds / 3600.0
        
        if total_time_hours <= 0:
            raise ValueError(f"Invalid observation time: {total_time_hours} hours")
        
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
            
            _ = self.process_time_series(background_data)
            n_alarms = len(self.alarms)
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
        self.process_time_series(background_data)
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
            'threshold': self.threshold,
            'loss_type': self.loss_type,
            'training_history': self.training_history_,
            'global_log_max': self.global_log_max_
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)
        
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.n_bins_ = checkpoint['n_bins']
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_size = checkpoint['hidden_size']
        self.latent_dim = checkpoint['latent_dim']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.threshold = checkpoint['threshold']
        self.loss_type = checkpoint['loss_type']
        self.training_history_ = checkpoint['training_history']
        self.global_log_max_ = checkpoint.get('global_log_max', 1.0)  # fallback for older saves
        
        self.model_ = TemporalLSTMAutoencoder(
            n_bins=self.n_bins_,
            hidden_size=self.hidden_size,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            global_log_max=self.global_log_max_
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
    
    def get_latent_representation(self, sequence: np.ndarray) -> np.ndarray:
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
