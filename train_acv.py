"""
Train the ACV refrigerant-leak localisation model.

Leave-one-case-out: for each held-out case, train a RandomForest on the other
5 cases (per-car features, positive = faulty car), rank the held-out case's
cars by leak probability, and score with the official rank-decay metric.

Run with:
    python train_acv.py
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import acv_pipeline as ap

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE, "ACV", "Train")
LABELS_PATH = os.path.join(BASE, "ACV", "Train_Labels.csv")
MODELS_PATH = os.path.join(BASE, "acv_models.joblib")


def loo_eval(model_cls, cache):
    labels = pd.read_csv(LABELS_PATH).set_index("filename")["faulty_car"].astype(str).str.zfill(2)
    files = sorted(glob.glob(os.path.join(TRAIN_DIR, "*.xlsx")))
    scores = []
    top1 = []
    for held in files:
        name = os.path.basename(held)
        true_car = labels[name]
        train_files = [f for f in files if f != held]

        X_parts, y_parts = [], []
        for f in train_files:
            cars, X = cache[f]
            fname = os.path.basename(f)
            X_parts.append(X)
            y_parts.append(np.array([1 if c == labels[fname] else 0 for c in cars]))
        X = pd.concat(X_parts, ignore_index=True)
        y = np.concatenate(y_parts)

        scaler = StandardScaler().fit(X)
        model = model_cls()
        model.fit(scaler.transform(X), y)

        cars, Xh = cache[held]
        proba = model.predict_proba(scaler.transform(Xh))[:, 1]
        ranked = [cars[i] for i in np.argsort(-proba)]
        scores.append(ap.rank_decay_score(true_car, ranked))
        top1.append(1.0 if ranked[0] == true_car else 0.0)
        print(f"  {name}: true={true_car} ranked=[{' > '.join(ranked)}] score={scores[-1]:.3f}")

    return float(np.mean(scores)), float(np.mean(top1))


def main():
    cache = {}
    print("loading case files...")
    for f in sorted(glob.glob(os.path.join(TRAIN_DIR, "*.xlsx"))):
        cache[f] = ap.extract_file_features(f)

    models = {
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42,
        ),
    }

    scores = {}
    for name, factory in models.items():
        print(f"\n{name}:")
        s, t1 = loo_eval(factory, cache)
        scores[name] = s
        print(f"  => rank-decay {s:.3f} | top-1 {t1 * 100:.0f}%")

    best_name = max(scores, key=scores.get)
    print(f"\nbest: {best_name}")

    labels = pd.read_csv(LABELS_PATH).set_index("filename")["faulty_car"].astype(str).str.zfill(2)
    X_parts, y_parts = [], []
    for f in cache:
        cars, X = cache[f]
        fname = os.path.basename(f)
        X_parts.append(X)
        y_parts.append(np.array([1 if c == labels[fname] else 0 for c in cars]))
    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts)

    bundle = {"models": {}, "scores": scores, "best": best_name,
              "feature_names": ap.FEATURE_NAMES, "version": "acv-0.1"}
    for name, factory in models.items():
        scaler = StandardScaler().fit(X)
        model = factory()
        model.fit(scaler.transform(X), y)
        bundle["models"][name] = {"model": model, "scaler": scaler}
    joblib.dump(bundle, MODELS_PATH)
    print(f"saved {MODELS_PATH}")


if __name__ == "__main__":
    main()
