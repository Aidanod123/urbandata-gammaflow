# ARAD Study Guide (src/detectors/arad.py)

This guide is a “walkthrough for understanding” the ARAD detector implementation in `src/detectors/arad.py`.

ARAD = **A**utoencoder **R**econstruction **A**nomaly **D**etection for gamma-ray spectra.

---

## 1) What problem ARAD solves
ARAD is an **unsupervised anomaly detector** for gamma-ray spectra.

- You train a model on **background-only** spectra.
- At inference time, you pass a new spectrum through the model.
- If the model reconstructs it poorly, the **reconstruction error** becomes an **anomaly score**.

In this implementation, the reconstruction error is measured using either:
- **Jensen–Shannon Divergence (JSD)** between normalized spectra, or
- **Chi-squared (χ²)** loss on denormalized count-rate data.

---

## 2) Key abstractions and where they appear in the file

### External data types
ARAD uses two GammaFlow abstractions:
- `SpectralTimeSeries`: a sequence of spectra with timing and timestamps.
- `Spectrum`: one spectrum with `counts`, integration time (`live_time`/`real_time`), and `energy_centers`.

### Major classes in `arad.py`
1. `ARADEncoderBlock`  
   A Conv1D encoder block: `Conv1d → BatchNorm → Mish → MaxPool(2) → Dropout`.

2. `ARADDecoderBlock`  
   A ConvTranspose1D decoder block: `Upsample(×2) → ConvTranspose1d → activation → (BatchNorm + Dropout)`.
   - The output block uses **Sigmoid** to force reconstruction into `[0, 1]`.

3. `ARADAutoencoder`  
   Convolutional autoencoder that:
   - normalizes input by max value,
   - encodes into a `latent_dim` vector,
   - decodes back into a reconstructed spectrum.

4. `ARADDetector`  
   User-facing detector API:
   - `fit(...)` trains the model
   - `score_spectrum(...)` scores one spectrum
   - `detect(...)` scores a time series and creates alarm events
   - `set_threshold_by_far(...)` calibrates a threshold to match a target false alarm rate
   - `save(...)` / `load(...)` persist a trained model
   - saliency methods: `compute_saliency_map(...)`, `plot_saliency(...)`

---

## 3) Model architecture (ARADAutoencoder)

### Shapes (important!)
- Input spectrum is expected as:
  - a single spectrum: shape `(n_bins,)`, or
  - a batch: shape `(batch, n_bins)`.

Internally, the autoencoder converts to Conv1D format:
- single spectrum becomes `(1, 1, n_bins)`
- batch becomes `(batch, 1, n_bins)`

### Encoder
The encoder has **5 pooling stages**, each halving the length:

- After 5 `MaxPool1d(2)` layers, the length becomes `n_bins / 32`.

This is why training checks:
- `n_bins % 32 == 0`

The encoder produces a latent vector of length `latent_dim`.

### Decoder
Decoder reconstructs length back to `n_bins`:
- `latent_dim → Linear → reshape to (batch, 8, n_bins/32)`
- then 5 decoder blocks with upsampling.

Final layer:
- `Sigmoid` output to keep reconstruction in `[0, 1]`.

### Weight initialization
`ARADAutoencoder._init_weights` initializes:
- Conv/ConvTranspose via Kaiming normal
- Linear via Xavier uniform
- Bias set to small positive constant (`0.01`)

---

## 4) Normalization strategy (crucial detail)
There are two “normalization” concepts in this file:

### A) Model input normalization (`ARADAutoencoder._normalize_input`)
- Divides each spectrum by its **maximum bin value**:  
  $x_{norm} = \frac{x}{\max(x) + \varepsilon}$
- This makes the model focus on **shape** more than absolute magnitude.

### B) Loss-side normalization (`ARADDetector._normalize_spectrum`)
- Also normalizes by max value for computing JSD.
- For chi-squared training, predictions are denormalized back using the same max.

Implication:
- The autoencoder’s output is “shape-like” (`[0, 1]`), and you denormalize when you want to compare back in physical units.

---

## 5) Loss functions and how scoring works

### 5.1 JSD loss
In code:
- clamp values for stability (`[1e-10, 1.0]`)
- compute midpoint distribution `m = 0.5(p+q)`
- compute KLD terms and combine

The returned loss is:
- $\sqrt{0.5\,(KL(p||m) + KL(q||m))}$ averaged over the batch.

Notes:
- This implementation uses a **square-rooted** JSD-like metric.

### 5.2 Chi-squared loss (Poisson-motivated)
Chi-squared is computed on denormalized predictions:
- Denormalize: `y_pred_denorm = y_pred * (max_vals + eps)`
- Then:  
  $\chi^2 = \sum_i \frac{(y_i - \hat{y}_i)^2}{\hat{y}_i}$

Notes:
- Denominator uses predicted expectation to approximate Poisson variance.

### 5.3 Total loss used during training
`ARADDetector._compute_loss(...)`:
- `recon_loss` (JSD or χ²)
- plus L1 regularization on parameters:
  - `l1_norm = sum(|param|)`
  - total: `recon_loss + l1_lambda * l1_norm`

L2 regularization is handled through AdamW weight decay (`l2_lambda`).

### 5.4 Scoring at inference
`score_spectrum(...)`:
- computes count-rate: `counts / time`
- reconstructs using model
- returns score using the same loss type:
  - JSD: normalize both input and reconstruction (note: it normalizes reconstruction again)
  - χ²: denormalize reconstruction to count-rate scale, then χ²

---

## 6) Training pipeline (ARADDetector.fit)

### Inputs
- `background_training: SpectralTimeSeries`
- optional `validation_data: SpectralTimeSeries`

### Steps
1. Convert counts to count-rate spectra:
   - uses `live_times` when available and valid
   - otherwise falls back to `real_times`

2. Check minimum training samples:
   - requires `training_spectra.shape[0] >= min_training_samples`

3. Check bin count constraint:
   - requires `n_bins % 32 == 0`

4. Train/validation split
   - if no validation data is provided, it does a random split using `validation_split`

5. Create Torch loaders
   - `TensorDataset(train_tensor)` and `DataLoader(..., shuffle=True)`

6. Initialize model
   - `ARADAutoencoder(n_bins, latent_dim, dropout)`

7. Optimizer + LR scheduler
   - `optim.AdamW(..., lr, weight_decay=l2_lambda, eps=0.1)`
   - `ReduceLROnPlateau` on validation loss

8. Epoch loop with early stopping
   - stops if validation loss doesn’t improve by `1e-4` for `early_stopping_patience` epochs

Outputs after training:
- `self.training_history_ = {'train_loss': [...], 'val_loss': [...]}`
- `self.is_fitted_ = True`

---

## 7) Detection pipeline (ARADDetector.detect)

### Preconditions
- Must be fitted (`is_fitted_ == True`)
- Must have a threshold (`self.threshold` set)

### Steps
1. Score each spectrum in `time_series` (looping spectrum-by-spectrum)
2. Convert scores into alarms:
   - `score > threshold` starts/continues an alarm
   - `aggregation_gap` merges “close” events to avoid double alarms

Alarm object keys:
- `start_time`, `end_time`, `peak_score`, `peak_time`

`process_time_series(...)` is a thin wrapper that:
- calls `detect`
- stores `self.alarms`
- returns the `scores`

---

## 8) Threshold calibration by FAR (set_threshold_by_far)

Goal: choose a threshold so the background-only run produces a target **false alarm rate**:

- FAR metric: **alarms per hour**
- total time uses `sum(background_data.real_times)` in seconds, then converts to hours

Approach:
- compute scores on background
- binary search a threshold in `[min(scores), 1.5*max(scores)]`
- for each candidate threshold:
  - run `process_time_series(background_data)`
  - count alarms and compute observed FAR

Selection rule:
- prefers thresholds whose FAR is closest to target
- tie-breaking tends to prefer **more sensitive** settings (higher FAR / lower threshold) when equally close

Result:
- sets `self.threshold`
- returns it

---

## 9) Persistence (save/load)

### save(path)
Saves:
- model weights (`state_dict`)
- architecture params: `n_bins`, `latent_dim`, `dropout`
- detector params: `threshold`, `loss_type`
- `training_history`

### load(path)
Loads the same information and rebuilds the autoencoder with the saved shape.

---

## 10) Explainability: saliency maps

### A) Gradient saliency
`_compute_gradient_saliency` computes:
- forward pass
- JSD loss between normalized input and reconstruction
- backprop to input
- returns `abs(grad)` per bin

### B) Integrated gradients
`_compute_integrated_gradients`:
- baseline defaults to all zeros
- interpolates from baseline to input across `n_steps`
- accumulates gradients, averages them, and multiplies by `(input - baseline)`

### C) Plotting
`plot_saliency(...)`:
- overlays saliency (as red spans) on top plot
- optionally shows reconstruction comparison below

---

## 11) Common pitfalls / “gotchas” to watch for

- **Bin divisibility**: ARAD requires `n_bins % 32 == 0`. If your spectra are 511 bins, you must rebin.
- **Time handling**: training and scoring divide by time. If `live_time` is missing/NaN, it falls back to `real_time`.
- **Count-rate vs counts**: this implementation works on count-rate arrays, not raw counts.
- **JSD stability**: values are clamped, but if a spectrum has all zeros, max-normalization can still yield all zeros; consider how your upstream pipeline handles empty spectra.
- **Device selection**: ARAD auto-selects `cuda → mps → cpu` when not specified.

---

## 12) Minimal usage template (mental model)

Training on background:
1. Prepare `background_training` as a `SpectralTimeSeries`.
2. `detector = ARADDetector(loss_type='jsd')`
3. `detector.fit(background_training)`
4. `detector.set_threshold_by_far(background_training, alarms_per_hour=0.5)`

Detecting on new data:
1. `scores, alarms = detector.detect(test_series)`
2. Inspect `alarms` and optionally explain them:
   - `detector.compute_saliency_map(test_series[i])`

---

## 13) Study questions / exercises

1. Why must `n_bins` be divisible by 32? Derive the length after 5 pooling layers.
2. What is gained and lost by normalizing each spectrum by its max bin value?
3. Compare JSD vs χ² here: when would χ² be a better operational score?
4. In `set_threshold_by_far`, why does it re-run alarm aggregation instead of thresholding individual scores directly?
5. Using a known source encounter, does the saliency highlight expected photopeaks? If not, what could that imply (data, preprocessing, or model capacity)?
