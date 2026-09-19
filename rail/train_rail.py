"""
Train the Rail Corrugation multi-class models (Normal / Side I / Side II).

Evaluates RandomForest / GradientBoosting with 5-fold stratified CV on the
official macro F1, then saves all models to rail_models.joblib.

Run with:
    python train_rail.py
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import rail_pipeline as rp

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
TRAIN_DIR = os.path.join(ROOT, "Rail_Corrugation", "Train")
LABELS_PATH = os.path.join(ROOT, "Rail_Corrugation", "Train_Labels.csv")
MODELS_PATH = os.path.join(BASE, "rail_models.joblib")


def main():
    labels = pd.read_csv(LABELS_PATH).set_index("filename")["label"]
    on_disk = sorted(
        f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".csv")
    )
    labels = labels[on_disk]
    paths = [os.path.join(TRAIN_DIR, f) for f in on_disk]
    y = labels.map({lab: i for i, lab in enumerate(rp.LABELS)}).to_numpy()

    t0 = time.time()
    X = rp.build_features_df(paths)
    print(f"features: {X.shape} | {time.time() - t0:.1f}s | labels: {dict(pd.Series(y).value_counts())}")

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42,
        ),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model in models.items():
        fold_scores = []
        for tr_idx, va_idx in skf.split(X, y):
            scaler = StandardScaler().fit(X.iloc[tr_idx])
            model.fit(scaler.transform(X.iloc[tr_idx]), y[tr_idx])
            pred = model.predict(scaler.transform(X.iloc[va_idx]))
            fold_scores.append(rp.macro_f1(y[va_idx], pred))
        scores[name] = float(np.mean(fold_scores))
        print(f"{name}: CV macro F1 = {scores[name]:.4f}")

    best_name = max(scores, key=scores.get)
    print(f"\nbest: {best_name}")

    bundle = {"models": {}, "scores": scores, "best": best_name,
              "feature_names": rp.FEATURE_NAMES, "version": "rail-0.1"}
    for name, model in models.items():
        scaler = StandardScaler().fit(X)
        model.fit(scaler.transform(X), y)
        bundle["models"][name] = {"model": model, "scaler": scaler}
    joblib.dump(bundle, MODELS_PATH)
    print(f"saved {MODELS_PATH}")


if __name__ == "__main__":
    main()
