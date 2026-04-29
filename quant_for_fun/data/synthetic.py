from __future__ import annotations

import numpy as np
import pandas as pd


def make_price_series(
    n_days: int = 1_000,
    start_price: float = 100.0,
    annual_drift: float = 0.08,
    annual_volatility: float = 0.25,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a synthetic daily OHLCV-like price series.

    The process is a simple geometric random walk. It is intentionally modest:
    useful for testing pipelines, not for representing a real market faithfully.
    """
    rng = np.random.default_rng(seed)
    trading_days = 252
    daily_drift = annual_drift / trading_days
    daily_volatility = annual_volatility / np.sqrt(trading_days)
    returns = rng.normal(daily_drift, daily_volatility, size=n_days)

    close = start_price * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.002, size=n_days))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, size=n_days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, size=n_days))
    volume = rng.integers(500_000, 5_000_000, size=n_days)

    index = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
