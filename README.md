# Redshift_-LTAproject-

Train Condition Monitoring for rail vehicles. The Door subsystem implements the full
segmentation + classification pipeline for the Door Fault Diagnosis dataset (temporal segment
detection problem); other subsystems (ACV, Rail Corrugation, SHM) are placeholders in the UI.

## Files

| File | Purpose |
|---|---|
| `RsFront.py` | Streamlit frontend console (instrument-console design). Run with `streamlit run RsFront.py`. Provides an upload console with a **prediction-model picker** that runs the Door inference pipeline and renders card-based results (cycle timeline, risk histogram, Monte Carlo simulation, survival curve, predictions table + CSV download). |
| `door_pipeline.py` | Shared core pipeline: parses the time format, segments a continuous door-controller stream into open/close cycles (gap-based split with a motion-flag safety net), extracts per-cycle features, and classifies cycles as Normal / Abnormal resistance. |
| `train_door.py` | Training script. Segments `Train.csv`, verifies boundaries against `Train_Segments_Answer.csv`, evaluates RandomForest / GradientBoosting / XGBoost (if installed) by 5-fold holdout IoU-weighted F1, retrains **all** models, and saves `door_models.joblib`. Run with `python train_door.py`. |
| `predict.py` | CLI inference script. Segments an input stream and writes per-cycle predictions with confidence. Usage: `python predict.py --input Test.csv --output door_predictions.csv [--model "Gradient Boosting"]`. Default model = the best-scoring one. |
| `door_dashboard.py` | Dashboard visuals for the Streamlit console — hand-built SVG cycle timeline, risk histogram, and a Monte Carlo reliability simulation. |
| `door_models.joblib` | Trained model bundle: `{models: {name: {model, scaler}}, scores, best, feature_names}` produced by `train_door.py`. |
| `Train.csv` | Training data — one continuous, unsegmented time-series stream of door-controller readings covering many open/close cycles. |
| `Train_Segments_Answer.csv` | Ground-truth segments for `Train.csv` (start/end times, operation type, Normal/Abnormal status). |
| `Door Data Headers.md` | Describes every column/parameter recorded in the dataset CSVs. |
| `Door_Subsystem_Info_Kit.md` | Full documentation of the Door Fault Diagnosis dataset and problem statement. |
| `.gitignore` | Excludes `__pycache__/` and the generated `door_predictions.csv` from the repo. |
| `LICENSE` | Project license. |

## Quick start

```bash
python train_door.py                                    # train + save door_models.joblib
python predict.py --input <your-stream>.csv --output door_predictions.csv   # run inference
streamlit run RsFront.py                                # launch the web console
```
