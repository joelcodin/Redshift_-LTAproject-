"""
Multi-subsystem inference CLI for Train Condition Monitoring.

Usage:
    python predict.py --subsystem door --input Test.csv --output door_predictions.csv
    python predict.py --subsystem shm  --input SHM/Test --output shm_predictions.csv
    python predict.py --subsystem rail --input Rail_Corrugation/Test --output rail_predictions.csv
    python predict.py --subsystem acv  --input ACV/Test/acv_test_case.xlsx --output acv_predictions.csv

--model optionally selects a named model from the subsystem's bundle
(default: the best-scoring one).
"""

import argparse
import os
import sys

import joblib
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

for _sub in ("door", "acv", "rail", "shm"):
    sys.path.insert(0, os.path.join(BASE, _sub))

SUBSYSTEMS = {
    "door": ("door/door_models.joblib", "door"),
    "shm": ("shm/shm_models.joblib", "shm"),
    "rail": ("rail/rail_models.joblib", "rail"),
    "acv": ("acv/acv_models.joblib", "acv"),
}


def main():
    parser = argparse.ArgumentParser(description="Train Condition Monitoring inference")
    parser.add_argument("--subsystem", required=True, choices=sorted(SUBSYSTEMS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    bundle_path, key = SUBSYSTEMS[args.subsystem]
    bundle = joblib.load(os.path.join(BASE, bundle_path))
    model_name = args.model or bundle["best"]
    if model_name not in bundle["models"]:
        raise SystemExit(
            f"Unknown model '{model_name}'. Available: {', '.join(bundle['models'])}"
        )
    entry = bundle["models"][model_name]

    if args.subsystem == "door":
        import door_pipeline as dp

        df = dp.load_stream(args.input)
        preds = dp.run_inference(df, entry["model"], entry["scaler"])
        out = preds[["start_time", "end_time", "status", "confidence"]].rename(
            columns={"status": "prediction"}
        )
        out.to_csv(args.output, index=False)
        n_ab = int((out["prediction"] == "Abnormal resistance").sum())
        print(f"model: {model_name} | cycles: {len(out)} | abnormal: {n_ab} | wrote: {args.output}")

    elif args.subsystem == "shm":
        import shm_pipeline as sp

        preds = sp.run_inference(args.input, entry["model"], entry["scaler"],
                                 log_target=entry.get("log_target", True))
        preds.to_csv(args.output, index=False)
        print(f"model: {model_name} | files: {len(preds)} | wrote: {args.output}")

    elif args.subsystem == "rail":
        import rail_pipeline as rp

        preds = rp.run_inference(args.input, entry["model"], entry["scaler"])
        preds[["file_id", "prediction"]].to_csv(args.output, index=False)
        print(f"model: {model_name} | files: {len(preds)} | wrote: {args.output}")
        print(preds["prediction"].value_counts().to_string())

    elif args.subsystem == "acv":
        import acv_pipeline as ap

        ranked, proba = ap.run_inference(args.input, entry["model"], entry["scaler"])
        out = pd.DataFrame({"file_id": [os.path.basename(args.input)],
                            "ranked_cars": ["|".join(ranked)]})
        out.to_csv(args.output, index=False)
        print(f"model: {model_name}")
        print("ranked_cars:", "|".join(ranked))
        print("wrote:", args.output)


if __name__ == "__main__":
    main()
