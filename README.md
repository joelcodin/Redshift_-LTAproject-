# Redshift_-LTAproject-

Train Condition Monitoring for rail vehicles. All four subsystems run real models through the
Streamlit console: Door (segment + classify + Monte Carlo), ACV (leak localisation), Rail
Corrugation (3-class classification) and SHM (fatigue damage regression).

## Files

| File | Purpose |
|---|---|
| `RsFront.py` | Streamlit frontend console (instrument-console design). Run with `streamlit run RsFront.py`. Upload one or multiple data files (CSV · TXT · XLSX), pick a prediction model, and render card-based results. Single file: full detail view (cycle timeline, risk histogram, Monte Carlo simulation, survival curve, predictions table + CSV download). Multiple files: batch output aggregated across all files with a combined CSV download. |
| `predict.py` | CLI inference script for all four subsystems. Usage: `python predict.py --subsystem shm --input SHM/Test --output shm_predictions.csv` (also `door`, `rail`, `acv`). |
| `door/` | Door subsystem: `door_pipeline.py` (core pipeline — time parsing, cycle segmentation, per-cycle features, classification), `door_dashboard.py` (SVG dashboard visuals + Monte Carlo simulation), `train_door.py` (trains and saves `door_models.joblib`), plus the dataset (`Train.csv`, `Train_Segments_Answer.csv`, `Test.csv`). |
| `acv/` | ACV subsystem: `acv_pipeline.py` (per-car telemetry features + leak-probability ranking), `train_acv.py` (leave-one-case-out training, saves `acv_models.joblib`). |
| `rail/` | Rail Corrugation subsystem: `rail_pipeline.py` (side/sensor-type aggregated features, 3-class classification), `train_rail.py` (saves `rail_models.joblib`). |
| `shm/` | SHM subsystem: `shm_pipeline.py` (rainflow-style histogram, Miner's-rule proxies, fatigue-damage regression), `train_shm.py` (saves `shm_models.joblib`). |
| `docs/` | Project docs: `PS3_Specifications.md`, `Door_Subsystem_Info_Kit.md`, `Door Data Headers.md`. |
| `ACV/`, `Rail_Corrugation/`, `SHM/` | Datasets for the other three subsystems (train + test data and labels). Not committed to the repo — keep them local at the repo root. |
| `04_Example_Submission/` | Example submission files (`acv_predictions.csv`, `door_predictions.csv`, `rail_predictions.csv`, `shm_predictions.csv`). |
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
python door/train_door.py                                # train + save door/door_models.joblib
python acv/train_acv.py                                  # train + save acv/acv_models.joblib
python rail/train_rail.py                                # train + save rail/rail_models.joblib
python shm/train_shm.py                                  # train + save shm/shm_models.joblib
python predict.py --subsystem shm --input SHM/Test --output shm_predictions.csv
streamlit run RsFront.py                                 # launch the web console
```
