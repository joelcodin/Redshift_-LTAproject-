# Redshift_-LTAproject-

Train Condition Monitoring for rail vehicles. The Door subsystem implements the full
segmentation + classification pipeline for the Door Fault Diagnosis dataset (temporal segment
detection problem); other subsystems (ACV, Rail Corrugation, SHM) are placeholders in the UI.

## Files

| File | Purpose |
|---|---|
| `RsFront.py` | Streamlit frontend console (tech-noir design). Run with `streamlit run RsFront.py`. Provides an upload console that runs the Door inference pipeline and renders results. |
| `door_pipeline.py` | Shared core pipeline: parses the time format, segments a continuous door-controller stream into open/close cycles (gap-based split with a motion-flag safety net), extracts per-cycle features, and classifies cycles as Normal / Abnormal resistance. |
| `train_door.py` | Training script. Segments `Train.csv`, verifies boundaries against `Train_Segments_Answer.csv`, compares GradientBoosting vs RandomForest by holdout IoU-weighted F1, retrains the winner, and saves `door_model.joblib`. Run with `python train_door.py`. |
| `predict.py` | CLI inference script. Segments an input stream and writes per-cycle predictions with confidence. Usage: `python predict.py --input Test.csv --output door_predictions.csv`. |
| `door_dashboard.py` | Dashboard visuals for the Streamlit console — hand-built SVG cycle timeline, risk histogram, and a Monte Carlo reliability simulation. |
| `door_model.joblib` | Trained model bundle (classifier + scaler + feature names) produced by `train_door.py`. |
| `Train.csv` | Training data — one continuous, unsegmented time-series stream of door-controller readings covering many open/close cycles. |
| `Train_Segments_Answer.csv` | Ground-truth segments for `Train.csv` (start/end times, operation type, Normal/Abnormal status). |
| `Test.csv` | Test data — another continuous stream; segments are not provided for this file. |
| `door_predictions.csv` | Example output of `predict.py` on `Test.csv` (start_time, end_time, prediction, confidence per cycle). |
| `Door Data Headers.md` | Describes every column/parameter recorded in the dataset CSVs. |
| `Door_Subsystem_Info_Kit.md` | Full documentation of the Door Fault Diagnosis dataset and problem statement. |
| `__pycache__/` | Compiled Python bytecode; not needed to run the project. |
| `LICENSE` | Project license. |

## Quick start

```bash
python train_door.py                                    # train + save door_model.joblib
python predict.py --input Test.csv --output door_predictions.csv   # run inference
streamlit run RsFront.py                                # launch the web console
```
