"""
Train the Door segmentation + classification pipeline with MULTIPLE models.

1. Segment Train.csv (gap-based split with motion-flag safety net) and verify
   the segmentation against Train_Segments_Answer.csv.
2. Extract per-cycle features and evaluate each classifier with 5-fold
   stratified IoU-weighted F1.
3. Save ALL models (with their scalers and scores) to door_models.joblib so
   the console can let the user pick a model at prediction time.

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
MODELS_PATH = os.path.join(BASE, "door_models.joblib")

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


def build_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
    return models


def main():
    train = dp.load_stream(os.path.join(BASE, "Train.csv"))
    ans = pd.read_csv(os.path.join(BASE, "Train_Segments_Answer.csv"))
    ans["s"] = ans["start_time"].apply(dp.parse_time)
    ans["e"] = ans["end_time"].apply(dp.parse_time)

    # --- segmentation check -------------------------------------------------
    segs = sorted(dp.segment_stream(train))
    print(f"segments found: {len(segs)} | true: {len(ans)}")
    exact = sum(
        1
        for (i0, i1), (_, row) in zip(segs, ans.iterrows())
        if dp.format_time(train.loc[i0, "t"]) == dp.format_time(row["s"])
        and dp.format_time(train.loc[i1, "t"]) == dp.format_time(row["e"])
    )
    print(f"exact boundary matches: {exact}/{len(ans)}")

    # --- features + labels ---------------------------------------------------
    X = dp.build_features_df(train, segs)
    y = ans["status"].map({dp.LABEL_NORMAL: 0, dp.LABEL_ABNORMAL: 1}).to_numpy()
    groups = ans["operation"].astype(str) + "_" + ans["status"].astype(str)
    print(f"\nfeatures: {X.shape[1]} | cycles: {len(y)} | abnormal: {y.sum()}")

    true_df = ans[["start_time", "end_time", "status"]]

    # --- evaluate each model --------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model in build_models().items():
        fold_scores = []
        for tr_idx, va_idx in skf.split(X, groups):
            scaler = StandardScaler().fit(X.iloc[tr_idx])
            Xtr, Xva = scaler.transform(X.iloc[tr_idx]), scaler.transform(X.iloc[va_idx])
            model.fit(Xtr, y[tr_idx])
            pred_labels = np.where(model.predict(Xva) == 1, dp.LABEL_ABNORMAL, dp.LABEL_NORMAL)
            pred_df = true_df.iloc[va_idx][["start_time", "end_time"]].copy()
            pred_df["prediction"] = pred_labels
            fold_scores.append(dp.score_iou_weighted_f1(true_df.iloc[va_idx], pred_df)["score"])
        scores[name] = float(np.mean(fold_scores))
        print(f"{name}: holdout IoU-F1 (5-fold mean) = {scores[name]:.4f}")

    best_name = max(scores, key=scores.get)
    print(f"\nbest model: {best_name}")

    # --- retrain every model on all data ---------------------------------------
    bundle = {"models": {}, "scores": scores, "best": best_name,
              "feature_names": dp.FEATURE_NAMES, "version": "door-multi-0.1"}
    for name, model in build_models().items():
        scaler = StandardScaler().fit(X)
        model.fit(scaler.transform(X), y)
        bundle["models"][name] = {"model": model, "scaler": scaler}

    joblib.dump(bundle, MODELS_PATH)
    print(f"saved {MODELS_PATH} with models: {list(bundle['models'])}")


if __name__ == "__main__":
    main()
