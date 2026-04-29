from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


FEATURE_COLUMNS = [
    "return_1d",
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "ma_gap_10d",
    "ma_gap_50d",
    "volume_z_20d",
]


@dataclass(frozen=True)
class TrainResult:
    model: RandomForestClassifier
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    predictions: pd.DataFrame


def train_random_forest_classifier(
    dataset: pd.DataFrame,
    train_fraction: float = 0.7,
    probability_threshold: float = 0.55,
    seed: int = 7,
) -> TrainResult:
    """Train a baseline classifier using chronological train/test split."""
    clean = dataset.dropna(subset=FEATURE_COLUMNS + ["target", "forward_return"]).copy()
    split_idx = int(len(clean) * train_fraction)
    train = clean.iloc[:split_idx]
    test = clean.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])

    train_proba = model.predict_proba(train[FEATURE_COLUMNS])[:, 1]
    test_proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    predictions = test[["close", "forward_return"]].copy()
    predictions["prob_up"] = test_proba
    predictions["signal"] = (predictions["prob_up"] >= probability_threshold).astype(int)

    return TrainResult(
        model=model,
        train_metrics=_classification_metrics(train["target"], train_proba),
        test_metrics=_classification_metrics(test["target"], test_proba),
        predictions=predictions,
    )


def _classification_metrics(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    labels = (probabilities >= 0.5).astype(int)
    metrics = {"accuracy": float(accuracy_score(y_true, labels))}

    if y_true.nunique() == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = float("nan")

    return metrics
