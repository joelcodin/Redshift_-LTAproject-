"""
Door subsystem inference script.

Segments a continuous door-controller data stream into open/close cycles and
classifies each cycle as Normal / Abnormal resistance.

Usage:
    python predict.py --input Test.csv --output door_predictions.csv

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
    args = parser.parse_args()

    bundle = joblib.load(os.path.join(BASE, "door_model.joblib"))
    model, scaler = bundle["model"], bundle["scaler"]

    df = dp.load_stream(args.input)
    preds = dp.run_inference(df, model, scaler)
    out = preds[["start_time", "end_time", "status", "confidence"]].rename(
        columns={"status": "prediction"}
    )
    out.to_csv(args.output, index=False)

    n_abnormal = int((out["prediction"] == dp.LABEL_ABNORMAL).sum())
    print(f"cycles found : {len(out)}")
    print(f"abnormal     : {n_abnormal}")
    print(f"normal       : {len(out) - n_abnormal}")
    print(f"wrote        : {args.output}")


if __name__ == "__main__":
    main()
