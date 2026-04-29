from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class WalkForwardModelResult:
    predictions: pd.DataFrame
    selected_alpha_by_date: pd.DataFrame
    validation_scores: pd.DataFrame


def build_cross_sectional_training_frame(
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    feature_columns: Iterable[str],
    score_col: str = "composite_score",
) -> pd.DataFrame:
    """Merge factor features with next-period returns for rank-model training."""
    features = list(feature_columns)
    missing = [col for col in features if col not in factor_panel.columns]
    if missing:
        raise ValueError(f"factor_panel missing feature columns: {missing}")

    keep_cols = _dedupe(
        [
        "date",
        "symbol",
        "sector",
        "eligible",
        score_col,
        "factor_coverage",
        *features,
        ]
    )
    data = factor_panel[[col for col in keep_cols if col in factor_panel.columns]].merge(
        forward_returns, on=["date", "symbol"], how="inner"
    )
    data = data[data["forward_return"].notna()].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["cross_sectional_target"] = data["forward_return"] - data.groupby("date")[
        "forward_return"
    ].transform("mean")
    return data.sort_values(["date", "symbol"]).reset_index(drop=True)


def walk_forward_ridge_ranker(
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    feature_columns: Iterable[str],
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    alpha_grid: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    min_train_months: int = 12,
    validation_months: int = 6,
) -> WalkForwardModelResult:
    """Train a monthly point-in-time Ridge ranker and score the test period."""
    features = list(feature_columns)
    training_frame = build_cross_sectional_training_frame(
        factor_panel, forward_returns, features
    )
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    rebalance_dates = pd.DatetimeIndex(
        sorted(
            date
            for date in pd.to_datetime(factor_panel["date"]).unique()
            if test_start <= pd.Timestamp(date) <= test_end
        )
    )

    predictions: list[pd.DataFrame] = []
    alpha_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []

    for date in rebalance_dates:
        train = training_frame[training_frame["date"] < date].copy()
        train_dates = pd.DatetimeIndex(sorted(train["date"].unique()))
        if len(train_dates) < min_train_months:
            continue
        active_features = [col for col in features if _has_any_observation(train, col)]
        if not active_features:
            continue

        best_alpha, alpha_scores = _select_alpha(
            train,
            active_features,
            alpha_grid=alpha_grid,
            validation_months=validation_months,
        )
        for alpha, score in alpha_scores.items():
            validation_rows.append(
                {"date": date, "alpha": alpha, "validation_ic": score}
            )

        model = _make_ridge_pipeline(best_alpha)
        model.fit(train[active_features], train["cross_sectional_target"])

        current = factor_panel[pd.to_datetime(factor_panel["date"]) == date].copy()
        if current.empty:
            continue
        current["model_score"] = model.predict(current[active_features])
        predictions.append(current)
        alpha_rows.append(
            {
                "date": date,
                "alpha": best_alpha,
                "active_features": ",".join(active_features),
                "train_months": len(train_dates),
                "train_rows": len(train),
            }
        )

    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=[*factor_panel.columns, "model_score"])
    )
    return WalkForwardModelResult(
        predictions=prediction_frame,
        selected_alpha_by_date=pd.DataFrame(alpha_rows),
        validation_scores=pd.DataFrame(validation_rows),
    )


def _select_alpha(
    train: pd.DataFrame,
    features: list[str],
    alpha_grid: Iterable[float],
    validation_months: int,
) -> tuple[float, dict[float, float]]:
    dates = pd.DatetimeIndex(sorted(train["date"].unique()))
    if len(dates) <= validation_months + 2:
        alpha = float(next(iter(alpha_grid)))
        return alpha, {alpha: np.nan}

    validation_dates = dates[-validation_months:]
    fit = train[train["date"] < validation_dates[0]]
    validate = train[train["date"].isin(validation_dates)]
    scores: dict[float, float] = {}

    for alpha in alpha_grid:
        alpha = float(alpha)
        model = _make_ridge_pipeline(alpha)
        model.fit(fit[features], fit["cross_sectional_target"])
        scored = validate[["date", "forward_return"]].copy()
        scored["prediction"] = model.predict(validate[features])
        scores[alpha] = _mean_monthly_spearman(scored)

    best_alpha = max(
        scores,
        key=lambda alpha: scores[alpha] if np.isfinite(scores[alpha]) else -np.inf,
    )
    if not np.isfinite(scores[best_alpha]):
        best_alpha = float(next(iter(alpha_grid)))
    return best_alpha, scores


def _make_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _has_any_observation(frame: pd.DataFrame, column: str) -> bool:
    values = frame[column]
    observed = values.notna()
    if isinstance(observed, pd.DataFrame):
        return bool(observed.any().any())
    return bool(observed.any())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _mean_monthly_spearman(scored: pd.DataFrame) -> float:
    ics: list[float] = []
    for _, date_frame in scored.groupby("date", sort=True):
        sample = date_frame[["prediction", "forward_return"]].dropna()
        if len(sample) < 3 or sample["prediction"].nunique() < 2:
            continue
        ic = sample["prediction"].corr(sample["forward_return"], method="spearman")
        if np.isfinite(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else np.nan
