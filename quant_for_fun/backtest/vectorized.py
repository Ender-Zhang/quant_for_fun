from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_long_flat(
    predictions: pd.DataFrame,
    trading_cost_bps: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Backtest a long-or-flat signal on the labeled forward returns.

    The signal generated at time t is applied to the forward return from t to
    t + horizon. This keeps the example compact; production research should
    model daily position paths and execution timing more explicitly.
    """
    bt = predictions.copy()
    cost = trading_cost_bps / 10_000
    turnover = bt["signal"].diff().abs().fillna(bt["signal"])

    bt["strategy_return"] = bt["signal"] * bt["forward_return"] - turnover * cost
    bt["benchmark_return"] = bt["forward_return"]
    bt["equity"] = (1 + bt["strategy_return"]).cumprod()
    bt["benchmark_equity"] = (1 + bt["benchmark_return"]).cumprod()

    return bt, performance_summary(bt["strategy_return"])


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    equity = (1 + clean).cumprod()
    total_return = equity.iloc[-1] - 1
    annual_return = equity.iloc[-1] ** (periods_per_year / len(clean)) - 1
    annual_volatility = clean.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    drawdown = equity / equity.cummax() - 1

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
