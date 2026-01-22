# UrbanData-GammaFlow Study Guide

## 1. Quick Facts
- **Goal**: Demonstrate classical + deep learning algorithms for gamma-ray source detection using the TopCoder Urban Radiation Detection dataset and the GammaFlow SDK.
- **Core Modules**: Thresholding ([src/detectors/k_sigma.py](src/detectors/k_sigma.py)), PCA-based anomaly detection ([src/detectors/sad.py](src/detectors/sad.py)), and convolutional autoencoder detection with saliency ([src/detectors/arad.py](src/detectors/arad.py)).
- **Primary Data Abstraction**: `SpectralTimeSeries` & `Spectrum` from GammaFlow encapsulate counts, integration times, and energy bins.
- **Workflows**: Interactive exploration via notebooks in `examples/`, scripted ARAD training in `examples/train_arad.py`, and reusable detector classes exposed through [src/detectors/__init__.py](src/detectors/__init__.py).

## 2. Repository Orientation
```
urbandata-gammaflow/
├── README.md                 # Project overview & setup steps
├── requirements.txt          # Python deps (NumPy, SciPy, torch, scikit-learn, etc.)
├── examples/                 # Notebooks + ARAD training script
└── src/detectors/            # Production-ready detector classes
```
Key directories to inspect first:
1. `README.md` – high-level goals, dataset expectations, and notebook flow.
2. `examples/` – how detectors are exercised end-to-end.
3. `src/detectors/` – reference implementations suitable for reuse in other pipelines.

## 3. Data Model Primer
- **Spectra**: Each `Spectrum` stores `counts`, `real_time`/`live_time`, optional `energy_centers`, and metadata. Detectors typically normalize to count rate by dividing counts by the integration time.
- **Time Series**: `SpectralTimeSeries` aggregates ordered spectra plus timestamps. Many helper properties (`counts`, `live_times`, `real_times`, `timestamps`) are used in detector APIs, so ensure your custom data provides equivalent attributes.
- **Typical Preprocessing**:
  1. Load list-mode CSVs → convert to spectra (see `examples/listmode_processing.ipynb`).
  2. Stack into `SpectralTimeSeries` for downstream detectors.
  3. Split background (source-free) vs. evaluation segments.

## 4. Detection Stack Overview
| Detector | Core Idea | Training Need | Strengths | Limitations |
| --- | --- | --- | --- | --- |
| `KSigmaDetector` | Rolling background stats, alarm when foreground exceeds $k$-sigma | None | Fast, interpretable gross-count monitor | Blind to spectral shape changes |
| `SADDetector` | PCA reconstruction error $SAD(x)=\lVert (I-UU^T)x\rVert^2$ | Background-only PCA fit | Multivariate, unsupervised | Linear subspace only |
| `ARADDetector` | Conv1D autoencoder + Jensen-Shannon or $\chi^2$ loss | Requires GPU-friendly training | Captures nonlinear structure, supports saliency | Heavier training, needs PyTorch |

## 5. K-Sigma Deep Dive ([src/detectors/k_sigma.py](src/detectors/k_sigma.py))
- **Rolling Background**: A deque stores `(time, count_rate)` samples within `background_window`. Only when `min_background_samples` have accumulated does detection begin.
- **Metric**: $k = \frac{f - \mu}{\sigma}$ where $f$ is current count rate, $\mu$ and $\sigma$ are rolling background stats.
- **Foreground/Background Windows**: Foreground uses recent `foreground_window` duration (converted to count rate). Background excludes samples gathered during an active alarm to prevent contamination.
- **Alarm Lifecycle**:
  1. Metric exceeds `k_threshold` → start alarm, record start time and peak metric.
  2. Consecutive high samples update the peak.
  3. When metric drops below threshold, `_end_alarm()` either appends a new `AlarmEvent` or merges with previous if gap < `aggregation_gap`.
- **Outputs**: `process_time_series()` returns per-spectrum k-sigma values (NaN until background ready) and populates `alarms`. Use `get_alarm_summary()` to log performance.
- **Study Tips**:
  - Plot rolling mean/std to understand stabilization time.
  - Sweep `k_threshold` vs. false alarms on background-only runs.

## 6. SAD (Spectral Anomaly Detection) ([src/detectors/sad.py](src/detectors/sad.py))
- **Training**:
  - `fit()` stacks prepared spectra (optionally normalized to unit integral) and trains an `sklearn` PCA model retaining `n_components`.
  - Requires at least `min_training_samples`; `n_bins` is inferred from first spectrum.
- **Scoring**:
  - `score_spectrum()` reconstructs via PCA and measures squared residual norm.
  - `score_time_series()` vectorizes scoring; `process_time_series()` additionally enforces thresholding + alarm aggregation identical in shape to K-sigma’s control flow.
- **Threshold Calibration**:
  - `set_threshold()` for manual control.
  - `set_threshold_by_far()` iteratively binary-searches the threshold that yields a target false-alarm rate (alarms/hour). Observation time comes from cumulative `real_times` (or timestamps fallback) ensuring ANSI-style metrics.
- **Explainability Hooks**:
  - `get_explained_variance_ratio()` reveals component importance.
  - Inspect residual spectra to see which energy regions drive large scores.
- **Study Tips**:
  - Visualize PCA basis vectors; this reveals which energy bands capture background variance.
  - Validate FAR tuning by plotting alarms/hour vs. threshold to internalize binary-search behavior.

## 7. ARAD (Autoencoder Reconstruction Anomaly Detection) ([src/detectors/arad.py](src/detectors/arad.py))
### 7.1 Model Architecture
- **Encoder**: Five `ARADEncoderBlock`s (Conv1D → BatchNorm → Mish + MaxPool) reduce a `(1 × n_bins)` spectrum to a latent vector (`latent_dim`, default 8). Requires `n_bins % 32 == 0` due to pooling chain.
- **Decoder**: Linear projection back to feature maps followed by five `ARADDecoderBlock`s (Upsample + ConvTranspose). Final layer applies Sigmoid to keep reconstructions in [0, 1].
- **Initialization**: He/Xavier initialization for stability; dropout for regularization.

### 7.2 Training Pipeline
1. **Preparation**: Convert `SpectralTimeSeries` counts → count rates (counts divided by live/real time). Randomly split into training/validation via `validation_split` or accept explicit `validation_data`.
2. **DataLoaders**: `TensorDataset` + `DataLoader` provide shuffling and batching.
3. **Optimization**: `AdamW` (with configurable `l1_lambda` + `l2_lambda`) and `ReduceLROnPlateau` scheduler.
4. **Losses**:
   - Jensen-Shannon Divergence (default): compares normalized spectra $P$ & reconstruction $Q$ via $JSD(P||Q)$ computed in `_jsd_loss()`.
   - Chi-squared: denormalizes predictions and evaluates Poisson-aware $\chi^2$.
5. **Early Stopping**: `early_stopping_patience` monitors validation loss; training history stored in `training_history_`.

### 7.3 Detection Flow
- `score_spectrum()` normalizes a single `Spectrum`, forwards through the autoencoder, and returns reconstruction loss consistent with training loss type.
- `detect()` scores an entire `SpectralTimeSeries`, then merges consecutive high scores into alarm dictionaries capturing start/end/peak metadata.
- `process_time_series()` exists for API symmetry with SAD—it simply calls `detect()` and caches `alarms`.
- `set_threshold_by_far()` mirrors the SAD routine but reuses ARAD scoring (important because saliency-aware detectors may have different score distributions).

### 7.4 Explainability Utilities
- `compute_saliency_map()` supports:
  - **Gradient**: $|\partial \mathcal{L} / \partial x_i|$ per energy bin.
  - **Integrated gradients**: averages gradients along a baseline→input path for more stable attributions.
- `plot_saliency()` overlays saliency heatmap atop the log-scaled spectrum and (optionally) reconstruction, accelerating root-cause analysis for anomalies.

### 7.5 Operational Notes
- Persist models via `save()` / `load()` in `models/` (ignored by git).
- Ensure GPU availability (`cuda` or `mps`) for practical training; falls back to CPU gracefully.
- Always confirm `n_bins` divisibility before training (rebinned spectra to 128 bins in notebooks satisfy this constraint).

## 8. Threshold Calibration & Alarm Handling
- All detectors share the concept of **alarm aggregation** using `aggregation_gap` seconds to avoid double-counting a single radioactive pass.
- False-alarm-rate targeting requires accurate wall-clock exposure. Prefer `real_times` sums over naive timestamp span to account for gaps or variable dwell times.
- Logging strategy:
  1. Run detector on background validation runs.
  2. Use `set_threshold_by_far(..., alarms_per_hour=x)` to auto-tune.
  3. Store resulting threshold with metadata (date, dataset subset) for reproducibility.

## 9. Hands-on Learning Path
1. **Read `README.md`** to confirm dataset placement and dependency installation.
2. **Notebook Progression**:
   - `listmode_processing.ipynb` → raw data fundamentals.
   - `k_sigma_detection.ipynb` → understand rolling statistics.
   - `sad_detection.ipynb` → practice threshold calibration.
   - `arad_detection.ipynb` → run pretrained model, inspect saliency.
3. **Train ARAD**: Execute `python examples/train_arad.py`, monitor loss curves, and examine saved `models/arad_background.pt`.
4. **Experiment**: Modify detector hyperparameters, re-run notebooks, and compare ROC/FAR tradeoffs.
5. **Explainability Drill**: Use `ARADDetector.compute_saliency_map()` on known source encounters to verify the model highlights isotope-specific peaks.

## 10. Extending the Codebase
- **New Detector Template**: Create `src/detectors/my_detector.py` mirroring the API (`fit`, `process_time_series`, `set_threshold_by_far`, `alarms`). Add exports in [src/detectors/__init__.py](src/detectors/__init__.py).
- **Additional Data Streams**: Wrap alternative datasets in `SpectralTimeSeries` objects so existing detectors work unchanged.
- **Model Improvements**: For ARAD, try different latent sizes, kernel widths, or attention layers—reuse `ARADAutoencoder` as a starting point.
- **Evaluation Utilities**: Consider adding ROC/FAR plotting helpers or metrics modules if you frequently benchmark detectors.

## 11. Suggested Study Questions
1. How does changing `background_window` in `KSigmaDetector` affect sensitivity to slow-moving sources?
2. Why is normalization optional in `SADDetector`, and when might you disable it?
3. Derive how Jensen-Shannon Divergence behaves when reconstruction perfectly matches the input ($JSD=0$) and when it fails (approaches 1).
4. What happens to ARAD training if `n_bins` is not divisible by 32, and how would you rebin spectra to satisfy this?
5. How would you validate that saliency maps align with known isotope emission lines?

---
Use this guide as a roadmap: start from simple detectors, build intuition about count statistics, then tackle the deep learning stack and interpretability tooling. Happy detecting!
