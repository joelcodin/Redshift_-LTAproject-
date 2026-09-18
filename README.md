# Redshift_-LTAproject-

Train Condition Monitoring for rail vehicles. All four subsystems run real models through the
Streamlit console: Door (segment + classify + Monte Carlo), ACV (leak localisation), Rail
Corrugation (3-class classification) and SHM (fatigue damage regression).

## Files

| File | Purpose |
|---|---|
| `RsFront.py` | Streamlit frontend console (instrument-console design). Run with `streamlit run RsFront.py`. Upload one or multiple data files (CSV · TXT · XLSX), pick a prediction model, and render card-based results. Single file: full detail view (cycle timeline, risk histogram, Monte Carlo simulation, survival curve, predictions table + CSV download). Multiple files: batch output aggregated across all files with a combined CSV download. |
| `door_pipeline.py` | Door core pipeline: parses the time format, segments a continuous door-controller stream into open/close cycles (gap-based split with a motion-flag safety net), extracts per-cycle features, and classifies cycles as Normal / Abnormal resistance. |
| `train_door.py` | Door training script. Segments `door/Train.csv`, verifies boundaries against `door/Train_Segments_Answer.csv`, evaluates RandomForest / GradientBoosting / XGBoost (if installed) by 5-fold holdout IoU-weighted F1, retrains **all** models, and saves `door_models.joblib`. Run with `python train_door.py`. |
| `door_dashboard.py` | Dashboard visuals for the Streamlit console — hand-built SVG cycle timeline, risk histogram, and a Monte Carlo reliability simulation. |
| `door_models.joblib` | Door model bundle: `{models: {name: {model, scaler}}, scores, best, feature_names}` produced by `train_door.py`. |
| `acv_pipeline.py` | ACV pipeline: per-car telemetry features (temperatures, cooling error, control-mode entropy) plus fleet-relative deviations, then leak-probability ranking of all 8 cars in a case. |
| `train_acv.py` | ACV training script. Builds features from `ACV/Train/*.xlsx`, trains models, scores with leave-one-case-out rank-decay, and saves `acv_models.joblib`. Run with `python train_acv.py`. |
| `acv_models.joblib` | ACV model bundle produced by `train_acv.py`. |
| `rail_pipeline.py` | Rail pipeline: aggregates axle-box vibration/shock channels by side and sensor type (RMS, peak, kurtosis, crest factor, FFT peak magnitude/frequency) and classifies recordings as Normal / Side I / Side II. |
| `train_rail.py` | Rail training script. Builds features from `Rail_Corrugation/Train/*.csv`, trains models, scores with macro F1, and saves `rail_models.joblib`. Run with `python train_rail.py`. |
| `rail_models.joblib` | Rail model bundle produced by `train_rail.py`. |
| `shm_pipeline.py` | SHM pipeline: rainflow-style cycle-range histogram, Miner's-rule damage proxies, signal statistics and exceedance counts from a dynamic-stress series; tree regressor on `log(damage)`. |
| `train_shm.py` | SHM training script. Builds features from `SHM/Train/*.csv`, trains models, scores with MAPE, and saves `shm_models.joblib`. Run with `python train_shm.py`. |
| `shm_models.joblib` | SHM model bundle produced by `train_shm.py`. |
| `predict.py` | CLI inference script for all four subsystems. Usage: `python predict.py --subsystem shm --input SHM/Test --output shm_predictions.csv` (also `door`, `rail`, `acv`). |
| `door/` | Door dataset: `Train.csv` (one continuous, unsegmented stream of door-controller readings), `Train_Segments_Answer.csv` (ground-truth segments), `Test.csv`. |
| `ACV/`, `Rail_Corrugation/`, `SHM/` | Datasets for the other three subsystems (train + test data and labels). Not committed to the repo — keep them local at the repo root. |
| `04_Example_Submission/` | Example submission files (`acv_predictions.csv`, `door_predictions.csv`, `rail_predictions.csv`, `shm_predictions.csv`). |
| `PS3_Specifications.md` | Problem statement and specification for the four subsystems. |
| `.gitignore` | Excludes `__pycache__/`, `*.pyc`, generated predictions and the local dataset folders. |
| `LICENSE` | Project license. |

## How each subsystem predicts

**Door — abnormal resistance per cycle** (`door_pipeline.py`)
1. Splits the stream into cycles on time gaps > 5 s and motion-flag rises (Door is opening / closing 0 -> 1).
2. Per cycle, extracts motor current/voltage/back-EMF statistics and the door-position stroke profile binned into 10 bins.
3. A GradientBoosting classifier labels each cycle Normal / Abnormal resistance, with a confidence score. A Monte Carlo simulation then projects fault counts over future cycles.

**ACV — leaking car ranking** (`acv_pipeline.py`)
1. Extracts per-car telemetry features (indoor/outdoor temperatures, cooling error, control-mode entropy) plus fleet-relative deviations (each car vs the median of the other cars in the same file).
2. A model scores each car's probability of being the leaking car.
3. All 8 cars are ranked — the top-ranked car is the suspected leak.

**Rail Corrugation — 3-class verdict** (`rail_pipeline.py`)
1. Takes a 1-second axle-box vibration recording (10 kHz, 128 channels = 8 cars x 8 bearing positions x vibration/shock).
2. Groups channels by side (odd positions = Side I, even positions = Side II) and sensor type; computes RMS, peak, kurtosis, crest factor and FFT peak magnitude/frequency per channel.
3. A classifier returns Normal / Side I / Side II with class probabilities.

**SHM — cumulative fatigue damage** (`shm_pipeline.py`)
1. Reads the single dynamic-stress column and derives rainflow-style cycle ranges from successive extrema.
2. Extracts features: signal statistics (mean, std, RMS, skew...), a 48-bin log cycle-range histogram, Miner's-rule damage proxies (`mean(range^m)` for m = 2-5), high-amplitude exceedance rates and zero-crossing rate.
3. A tree regressor trained on `log(damage)` predicts, and the result is exponentiated back to cumulative damage (0-1, where 1.0 is the fatigue failure threshold).

All four: features are standardized with the saved scaler, then the trained model in the corresponding
`*_models.joblib` bundle makes the prediction.

## Quick start

```bash
python train_door.py                                    # train + save door_models.joblib
python train_acv.py                                     # train + save acv_models.joblib
python train_rail.py                                    # train + save rail_models.joblib
python train_shm.py                                     # train + save shm_models.joblib
python predict.py --subsystem shm --input SHM/Test --output shm_predictions.csv
streamlit run RsFront.py                                # launch the web console
```
