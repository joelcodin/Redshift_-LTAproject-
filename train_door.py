"""
Train the Door segmentation + classification pipeline.

1. Segment Train.csv (gap-based split with motion-flag safety net) and verify
   the segmentation against Train_Segments_Answer.csv.
2. Extract per-cycle features and train a classifier (GradientBoosting vs
   RandomForest), selecting by holdout IoU-weighted F1.
3. Retrain the winner on all labelled cycles and save door_model.joblib.

Run with:
    python train_door.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import door_pipeline as dp

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "door_model.joblib")


def main():
    train = dp.load_stream(os.path.join(BASE, "Train.csv"))
    ans = pd.read_csv(os.path.join(BASE, "Train_Segments_Answer.csv"))
    ans["s"] = ans["start_time"].apply(dp.parse_time)
    ans["e"] = ans["end_time"].apply(dp.parse_time)

    # --- segmentation check -------------------------------------------------
    segs = dp.segment_stream(train)
    print(f"segments found: {len(segs)} | true: {len(ans)}")

    exact = 0
    for (i0, i1), (_, row) in zip(sorted(segs), ans.iterrows()):
        if dp.format_time(train.loc[i0, "t"]) == dp.format_time(row["s"]) and dp.format_time(
            train.loc[i1, "t"]
        ) == dp.format_time(row["e"]):
            exact += 1
    print(f"exact boundary matches: {exact}/{len(ans)}")

    # --- features + labels ---------------------------------------------------
    X = dp.build_features_df(train, sorted(segs))
    y = ans["status"].map({dp.LABEL_NORMAL: 0, dp.LABEL_ABNORMAL: 1}).to_numpy()
    groups = ans["operation"].astype(str) + "_" + ans["status"].astype(str)

    print(f"\nfeatures: {X.shape[1]} | cycles: {len(y)} | abnormal: {y.sum()}")

    # --- holdout: classification quality on exact boundaries ------------------
    true_df = ans[["start_time", "end_time", "status"]].rename(
        columns={"start_time": "start_time", "end_time": "end_time"}
    )

    results = {}
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in models.items():
        scores = []
        for tr_idx, va_idx in skf.split(X, groups):
            scaler = StandardScaler().fit(X.iloc[tr_idx])
            Xtr, Xva = scaler.transform(X.iloc[tr_idx]), scaler.transform(X.iloc[va_idx])
            model.fit(Xtr, y[tr_idx])
            pred_labels = np.where(model.predict(Xva) == 1, dp.LABEL_ABNORMAL, dp.LABEL_NORMAL)
            pred_df = true_df.iloc[va_idx][["start_time", "end_time"]].copy()
            pred_df["prediction"] = pred_labels
            scores.append(dp.score_iou_weighted_f1(true_df.iloc[va_idx], pred_df)["score"])
        results[name] = float(np.mean(scores))
        print(f"{name}: holdout IoU-F1 (5-fold mean) = {results[name]:.4f}")

    best_name = max(results, key=results.get)
    print(f"\nbest model: {best_name}")

    # --- retrain on all data --------------------------------------------------
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = models[best_name]
    model.fit(Xs, y)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_names": dp.FEATURE_NAMES,
            "model_name": best_name,
            "version": "door-0.1",
        },
        MODEL_PATH,
    )
    print(f"saved {MODEL_PATH}")

    # --- final sanity: train-set score (expected ~1.0) -------------------------
    proba = model.predict_proba(Xs)[:, 1]
    pred_labels = np.where(model.predict(Xs) == 1, dp.LABEL_ABNORMAL, dp.LABEL_NORMAL)
    pred_df = true_df[["start_time", "end_time"]].copy()
    pred_df["prediction"] = pred_labels
    final = dp.score_iou_weighted_f1(true_df, pred_df)
    print(f"train-set IoU-F1: {final['score']:.4f}")


if __name__ == "__main__":
    main()
