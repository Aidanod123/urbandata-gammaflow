"""
Anomaly detection algorithms for gamma-ray time series.

This module provides various detection algorithms for identifying
radioactive source presence in time series data.

Algorithms
----------
KSigmaDetector : Simple k-sigma threshold detection with rolling background
SADDetector : Spectral Anomaly Detection using PCA-based reconstruction error
ARADDetector : Autoencoder Reconstruction Anomaly Detection (requires PyTorch)
ARADLSTMDetector : LSTM-based Autoencoder Reconstruction Anomaly Detection (requires PyTorch)
"""

from src.detectors.k_sigma import KSigmaDetector, AlarmEvent
from src.detectors.sad import SADDetector
from src.detectors.arad import ARADDetector
from src.detectors.arad_lstm import ARADLSTMDetector

__all__ = [
    'KSigmaDetector',
    'SADDetector',
    'ARADDetector',
    'ARADLSTMDetector',
    'AlarmEvent',
]

