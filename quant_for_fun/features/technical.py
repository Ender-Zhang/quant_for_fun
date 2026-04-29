from __future__ import annotations

import pandas as pd


def add_technical_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a compact set of technical features from OHLCV data."""
    df = prices.copy()
    close = df["close"]

    returns = close.pct_change()
    df["return_1d"] = returns
    df["momentum_5d"] = close.pct_change(5)
    df["momentum_20d"] = close.pct_change(20)
    df["volatility_20d"] = returns.rolling(20).std()
    df["ma_gap_10d"] = close / close.rolling(10).mean() - 1
    df["ma_gap_50d"] = close / close.rolling(50).mean() - 1
    df["volume_z_20d"] = (df["volume"] - df["volume"].rolling(20).mean()) / df[
        "volume"
    ].rolling(20).std()

    return df


def add_forward_return_label(
    features: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Add a binary label: whether forward return exceeds a threshold."""
    df = features.copy()
    forward_return = df["close"].shift(-horizon) / df["close"] - 1
    df["forward_return"] = forward_return
    df["target"] = (forward_return > threshold).astype(int)
    return df
