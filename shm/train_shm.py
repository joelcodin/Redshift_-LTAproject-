"""
Train the SHM fatigue-damage regression models.

Evaluates RandomForest / GradientBoosting (log target) with 5-fold CV
using the official score max(0, 1 - MAPE), then saves all models to
shm_models.joblib.

Run with:
    python train_shm.py
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import shm_pipeline as sp

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
TRAIN_DIR = os.path.join(ROOT, "SHM", "Train")
LABELS_PATH = os.path.join(ROOT, "SHM", "Train_Labels.csv")
MODELS_PATH = os.path.join(BASE, "shm_models.joblib")


def main():
    labels = pd.read_csv(LABELS_PATH).set_index("filename")["damage"]
    files = sorted(os.listdir(TRAIN_DIR))
    paths = [os.path.join(TRAIN_DIR, f) for f in files]
    y = labels[files].to_numpy(dtype=float)

    t0 = time.time()
    X = sp.build_features_df(paths, verbose=True)
    print(f"features: {X.shape} | extracted in {time.time() - t0:.1f}s")
    print(f"damage range: [{y.min():.4f}, {y.max():.4f}]")

    variants = {
        "Random Forest": (
            RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1),
            True,
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, random_state=42),
            True,
        ),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, (model, use_log) in variants.items():
        target = np.log1p(y) if use_log else y
        fold_scores = []
        for tr_idx, va_idx in kf.split(X):
            scaler = StandardScaler().fit(X.iloc[tr_idx])
            model.fit(scaler.transform(X.iloc[tr_idx]), target[tr_idx])
            pred = model.predict(scaler.transform(X.iloc[va_idx]))
            if use_log:
                pred = np.expm1(pred)
            fold_scores.append(sp.mape_score(y[va_idx], pred)[0])
        scores[name] = float(np.mean(fold_scores))
        print(f"{name}: CV score (1-MAPE) = {scores[name]:.4f}")

    best_name = max(scores, key=scores.get)
    print(f"\nbest: {best_name}")

    bundle = {"models": {}, "scores": scores, "best": best_name,
              "feature_names": sp.FEATURE_NAMES, "version": "shm-0.1"}
    for name, (model, use_log) in variants.items():
        target = np.log1p(y) if use_log else y
        scaler = StandardScaler().fit(X)
        model.fit(scaler.transform(X), target)
        bundle["models"][name] = {
            "model": model, "scaler": scaler, "log_target": use_log,
        }
    joblib.dump(bundle, MODELS_PATH)
    print(f"saved {MODELS_PATH}")


if __name__ == "__main__":
    main()
