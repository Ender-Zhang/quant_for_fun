from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd


def select_long_only_portfolio(
    factor_panel: pd.DataFrame,
    top_n: int = 30,
    sector_cap: float = 0.25,
    max_weight: float = 0.05,
    score_col: str = "composite_score",
    eligibility_col: str = "eligible",
) -> pd.DataFrame:
    """Select a monthly long-only portfolio from a cross-sectional factor panel."""
    required = {"date", "symbol", "sector", score_col}
    missing = required.difference(factor_panel.columns)
    if missing:
        raise ValueError(f"factor_panel missing required columns: {sorted(missing)}")

    rows: list[pd.DataFrame] = []
    for date, date_frame in factor_panel.groupby("date", sort=True):
        candidates = date_frame.copy()
        if eligibility_col in candidates.columns:
            candidates = candidates[candidates[eligibility_col]]
        candidates = candidates[candidates[score_col].notna()]
        candidates = candidates.sort_values(score_col, ascending=False)

        if candidates.empty:
            continue

        max_per_sector = max(1, math.floor(top_n * sector_cap))
        sector_counts: dict[str, int] = {}
        selected_indices: list[int] = []
        for idx, row in candidates.iterrows():
            sector = str(row.get("sector", "Unknown"))
            if sector_counts.get(sector, 0) >= max_per_sector:
                continue
            selected_indices.append(idx)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected_indices) >= top_n:
                break

        selected = candidates.loc[selected_indices].copy()
        if selected.empty:
            continue

        weight = min(1.0 / len(selected), max_weight)
        selected["target_weight"] = weight
        selected["cash_weight"] = 1.0 - weight * len(selected)
        selected["selected_rank"] = np.arange(1, len(selected) + 1)
        selected["rebalance_date"] = date
        rows.append(
            selected[
                [
                    "rebalance_date",
                    "date",
                    "symbol",
                    "sector",
                    score_col,
                    "target_weight",
                    "cash_weight",
                    "selected_rank",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "rebalance_date",
                "date",
                "symbol",
                "sector",
                score_col,
                "target_weight",
                "cash_weight",
                "selected_rank",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def make_forward_monthly_returns(
    prices: pd.DataFrame,
    horizon_months: int = 1,
) -> pd.DataFrame:
    """Convert daily prices to rebalance-date forward returns."""
    panel = _prepare_price_panel(prices)
    month_end_dates = _month_end_trading_dates(panel["date"])
    monthly = panel[panel["date"].isin(month_end_dates)][["date", "symbol", "close"]].copy()
    monthly = monthly.sort_values(["symbol", "date"])
    monthly["forward_return"] = monthly.groupby("symbol")["close"].shift(
        -horizon_months
    ) / monthly["close"] - 1
    return monthly[["date", "symbol", "forward_return"]].dropna()


def factor_information_coefficient(
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_columns: Iterable[str],
) -> pd.DataFrame:
    """Calculate monthly Spearman IC, ICIR, and hit rate for factor columns."""
    data = factor_panel.merge(forward_returns, on=["date", "symbol"], how="inner")
    rows: list[dict[str, float | str | int]] = []
    for factor in factor_columns:
        if factor not in data.columns:
            continue
        monthly_ics: list[float] = []
        for _, date_frame in data.groupby("date", sort=True):
            sample = date_frame[[factor, "forward_return"]].dropna()
            if len(sample) < 3:
                continue
            ic = sample[factor].corr(sample["forward_return"], method="spearman")
            if pd.notna(ic):
                monthly_ics.append(float(ic))

        if monthly_ics:
            ic_series = pd.Series(monthly_ics)
            ic_std = ic_series.std(ddof=1)
            mean_ic = ic_series.mean()
            icir = mean_ic / ic_std if ic_std and np.isfinite(ic_std) else np.nan
            hit_rate = float((ic_series > 0).mean())
        else:
            mean_ic = np.nan
            icir = np.nan
            hit_rate = np.nan

        rows.append(
            {
                "factor": factor,
                "mean_ic": float(mean_ic) if pd.notna(mean_ic) else np.nan,
                "icir": float(icir) if pd.notna(icir) else np.nan,
                "hit_rate": hit_rate,
                "observations": len(monthly_ics),
            }
        )
    return pd.DataFrame(rows)


def quantile_forward_returns(
    factor_panel: pd.DataFrame,
    forward_returns: pd.DataFrame,
    factor_col: str = "composite_score",
    quantiles: int = 5,
) -> pd.DataFrame:
    """Measure forward returns by factor bucket for monotonicity checks."""
    data = factor_panel.merge(forward_returns, on=["date", "symbol"], how="inner")
    data = data[[factor_col, "forward_return", "date"]].dropna()
    if data.empty:
        return pd.DataFrame(columns=["quantile", "mean_forward_return", "count"])

    def assign_bucket(frame: pd.DataFrame) -> pd.Series:
        if frame[factor_col].nunique() < quantiles:
            return pd.Series(np.nan, index=frame.index)
        return pd.qcut(frame[factor_col], quantiles, labels=False, duplicates="drop") + 1

    data["quantile"] = data.groupby("date", group_keys=False).apply(assign_bucket)
    return (
        data.dropna(subset=["quantile"])
        .groupby("quantile")["forward_return"]
        .agg(mean_forward_return="mean", count="count")
        .reset_index()
    )


def summarize_portfolio_exposures(
    portfolio: pd.DataFrame,
    factor_panel: pd.DataFrame,
    exposure_columns: Iterable[str] = (
        "size_control_z",
        "book_to_market_control_z",
        "momentum_12_1_control_z",
        "beta_control_z",
        "volatility_control_z",
    ),
) -> pd.DataFrame:
    """Report weighted average exposure to common controls by rebalance date."""
    if portfolio.empty:
        return pd.DataFrame(columns=["date", *exposure_columns])

    joined = portfolio.merge(
        factor_panel,
        on=["date", "symbol"],
        how="left",
        suffixes=("_portfolio", ""),
    )
    rows: list[dict[str, float | pd.Timestamp]] = []
    for date, date_frame in joined.groupby("date", sort=True):
        row: dict[str, float | pd.Timestamp] = {"date": date}
        weights = date_frame["target_weight"].fillna(0.0)
        for col in exposure_columns:
            if col in date_frame.columns:
                valid = date_frame[col].notna()
                denominator = weights[valid].sum()
                row[col] = (
                    float((date_frame.loc[valid, col] * weights[valid]).sum() / denominator)
                    if denominator > 0
                    else np.nan
                )
            else:
                row[col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    panel = prices.copy()
    if "date" not in panel.columns:
        if isinstance(panel.index, pd.MultiIndex):
            panel = panel.reset_index()
        else:
            panel = panel.reset_index().rename(columns={"index": "date"})
    if "symbol" not in panel.columns:
        panel["symbol"] = "SYNTH"
    required = {"date", "symbol", "close"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"prices missing required columns: {sorted(missing)}")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def _month_end_trading_dates(dates: pd.Series) -> pd.DatetimeIndex:
    date_series = pd.Series(pd.to_datetime(dates).sort_values().unique())
    month_period = date_series.dt.to_period("M")
    return pd.DatetimeIndex(date_series.groupby(month_period).max().to_numpy())
