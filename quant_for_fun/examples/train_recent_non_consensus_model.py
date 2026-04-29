from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_for_fun.examples.backtest_recent_non_consensus import (
    DEFAULT_TICKERS,
    attach_market_cap_to_fundamentals,
    attach_market_cap_to_prices,
    backtest_monthly_holdings,
    fetch_cik_map,
    fetch_sec_fundamentals,
    fetch_yahoo_prices,
    make_security_master,
    summarize_returns,
    total_return,
)
from quant_for_fun.features.non_consensus import RAW_FACTOR_COLUMNS, build_non_consensus_factor_panel
from quant_for_fun.models.cross_sectional import walk_forward_ridge_ranker
from quant_for_fun.portfolio import (
    factor_information_coefficient,
    make_forward_monthly_returns,
    quantile_forward_returns,
    select_long_only_portfolio,
    summarize_portfolio_exposures,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a walk-forward cross-sectional model on non-consensus factors."
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-04-29")
    parser.add_argument("--lookback-start", default="2020-01-01")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--min-factor-coverage", type=float, default=0.30)
    parser.add_argument("--min-train-months", type=int, default=24)
    parser.add_argument("--validation-months", type=int, default=6)
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--output-dir", default="reports/recent_non_consensus_model")
    parser.add_argument("--cache-dir", default="reports/cache/recent_non_consensus")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_model_backtest(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        lookback_start=args.lookback_start,
        top_n=args.top_n,
        max_weight=args.max_weight,
        min_factor_coverage=args.min_factor_coverage,
        min_train_months=args.min_train_months,
        validation_months=args.validation_months,
        cache_dir=cache_dir,
    )

    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(output_dir / f"{name}.csv", index=False)

    print("Walk-forward Ridge non-consensus model")
    print(f"period: {args.start} to {args.end}")
    print(f"tickers requested: {len(args.tickers)}")
    print("\nModel summary")
    for key, value in result["model_summary"].iloc[0].items():
        print(f"{key:>28}: {value: .4f}")
    print("\nComposite summary")
    for key, value in result["composite_summary"].iloc[0].items():
        print(f"{key:>28}: {value: .4f}")
    print("\nModel IC")
    print(result["model_ic"].to_string(index=False))
    print("\nModel quantiles")
    print(result["model_quantiles"].to_string(index=False))
    print("\nSelected alpha by date")
    print(result["selected_alpha_by_date"].tail(8).to_string(index=False))
    print("\nLatest model holdings")
    latest = result["model_holdings"][
        result["model_holdings"]["date"] == result["model_holdings"]["date"].max()
    ]
    print(
        latest[
            ["date", "symbol", "sector", "model_score", "target_weight", "selected_rank"]
        ].to_string(index=False)
    )
    print(f"\nWrote CSV outputs to {output_dir}")


def run_model_backtest(
    tickers: list[str],
    start: str,
    end: str,
    lookback_start: str,
    top_n: int,
    max_weight: float,
    min_factor_coverage: float,
    min_train_months: int,
    validation_months: int,
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    lookback_ts = pd.Timestamp(lookback_start)

    prices = fetch_yahoo_prices(tickers, lookback_ts, end_ts, cache_dir=cache_dir)
    cik_map = fetch_cik_map(cache_dir=cache_dir)
    fundamentals = fetch_sec_fundamentals(
        tickers, cik_map, start_year=lookback_ts.year - 2, cache_dir=cache_dir
    )
    fundamentals = attach_market_cap_to_fundamentals(fundamentals, prices)
    prices = attach_market_cap_to_prices(prices, fundamentals)
    securities = make_security_master(tickers)
    factor_panel = build_non_consensus_factor_panel(
        prices,
        fundamentals,
        securities=securities,
        min_factor_coverage=min_factor_coverage,
    )
    forward_returns = make_forward_monthly_returns(prices)

    model_features = [
        col
        for col in [f"{factor}_z" for factor in RAW_FACTOR_COLUMNS] + ["factor_coverage"]
        if col in factor_panel.columns
    ]
    model_result = walk_forward_ridge_ranker(
        factor_panel,
        forward_returns,
        feature_columns=model_features,
        test_start=start_ts,
        test_end=end_ts,
        min_train_months=min_train_months,
        validation_months=validation_months,
    )
    model_panel = model_result.predictions.copy()
    model_panel = model_panel[(model_panel["date"] >= start_ts) & (model_panel["date"] <= end_ts)]
    composite_panel = factor_panel[
        (factor_panel["date"] >= start_ts) & (factor_panel["date"] <= end_ts)
    ].copy()
    forward_test = forward_returns[
        (forward_returns["date"] >= start_ts) & (forward_returns["date"] <= end_ts)
    ].copy()

    model_holdings = select_long_only_portfolio(
        model_panel,
        top_n=top_n,
        sector_cap=0.25,
        max_weight=max_weight,
        score_col="model_score",
    )
    composite_holdings = select_long_only_portfolio(
        composite_panel,
        top_n=top_n,
        sector_cap=0.25,
        max_weight=max_weight,
        score_col="composite_score",
    )
    model_returns = backtest_monthly_holdings(model_holdings, forward_test)
    composite_returns = backtest_monthly_holdings(composite_holdings, forward_test)

    benchmark_total = total_return(model_returns["benchmark_return"])
    model_summary = summarize_returns(model_returns["portfolio_return"])
    model_summary.update(
        {
            "benchmark_total_return": benchmark_total,
            "excess_total_return": total_return(
                model_returns["portfolio_return"] - model_returns["benchmark_return"]
            ),
            "average_holdings": float(model_holdings.groupby("date")["symbol"].nunique().mean()),
        }
    )
    composite_summary = summarize_returns(composite_returns["portfolio_return"])
    composite_summary.update(
        {
            "benchmark_total_return": total_return(composite_returns["benchmark_return"]),
            "excess_total_return": total_return(
                composite_returns["portfolio_return"]
                - composite_returns["benchmark_return"]
            ),
            "average_holdings": float(
                composite_holdings.groupby("date")["symbol"].nunique().mean()
            ),
        }
    )

    model_ic = factor_information_coefficient(
        model_panel,
        forward_test,
        ["model_score", "composite_score", *model_features],
    )
    model_quantiles = quantile_forward_returns(
        model_panel,
        forward_test,
        factor_col="model_score",
    )
    model_exposures = summarize_portfolio_exposures(model_holdings, model_panel)

    return {
        "model_summary": pd.DataFrame([model_summary]),
        "composite_summary": pd.DataFrame([composite_summary]),
        "model_monthly_returns": model_returns,
        "composite_monthly_returns": composite_returns,
        "model_holdings": model_holdings,
        "composite_holdings": composite_holdings,
        "model_ic": model_ic,
        "model_quantiles": model_quantiles,
        "model_exposures": model_exposures,
        "selected_alpha_by_date": model_result.selected_alpha_by_date,
        "validation_scores": model_result.validation_scores,
        "model_factor_panel": model_panel,
    }


if __name__ == "__main__":
    main()
