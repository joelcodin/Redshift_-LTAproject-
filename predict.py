"""
Door subsystem inference script.

Segments a continuous door-controller data stream into open/close cycles and
classifies each cycle as Normal / Abnormal resistance.

Usage:
    python predict.py --input Test.csv --output door_predictions.csv
    python predict.py --input Test.csv --output preds.csv --model "Gradient Boosting"

Output columns: start_time, end_time, prediction, confidence
"""

import argparse
import os

import joblib

import door_pipeline as dp

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description="Door fault-diagnosis inference")
    parser.add_argument("--input", required=True, help="Path to the continuous CSV stream")
    parser.add_argument(
        "--output",
        default=os.path.join(BASE, "door_predictions.csv"),
        help="Path for the predictions CSV (default: door_predictions.csv)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name from door_models.joblib (default: the best one)",
    )
    args = parser.parse_args()

    bundle = joblib.load(os.path.join(BASE, "door_models.joblib"))
    model_name = args.model or bundle["best"]
    if model_name not in bundle["models"]:
        raise SystemExit(
            f"Unknown model '{model_name}'. Available: {', '.join(bundle['models'])}"
        )
    entry = bundle["models"][model_name]

    df = dp.load_stream(args.input)
    preds = dp.run_inference(df, entry["model"], entry["scaler"])
    out = preds[["start_time", "end_time", "status", "confidence"]].rename(
        columns={"status": "prediction"}
    )
    out.to_csv(args.output, index=False)

    n_abnormal = int((out["prediction"] == dp.LABEL_ABNORMAL).sum())
    print(f"model        : {model_name}")
    print(f"cycles found : {len(out)}")
    print(f"abnormal     : {n_abnormal}")
    print(f"normal       : {len(out) - n_abnormal}")
    print(f"wrote        : {args.output}")


if __name__ == "__main__":
    main()
