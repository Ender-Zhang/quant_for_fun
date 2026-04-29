from __future__ import annotations

import argparse
import gzip
import http.client
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from quant_for_fun.features.non_consensus import (
    RAW_FACTOR_COLUMNS,
    build_non_consensus_factor_panel,
)
from quant_for_fun.portfolio import (
    factor_information_coefficient,
    make_forward_monthly_returns,
    quantile_forward_returns,
    select_long_only_portfolio,
    summarize_portfolio_exposures,
)


DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AVGO",
    "ORCL",
    "CRM",
    "ADBE",
    "AMD",
    "QCOM",
    "CSCO",
    "TXN",
    "IBM",
    "INTU",
    "AMAT",
    "LRCX",
    "MU",
    "KLAC",
    "ANET",
    "PANW",
    "NOW",
    "JNJ",
    "LLY",
    "MRK",
    "ABBV",
    "TMO",
    "ABT",
    "ISRG",
    "DHR",
    "COST",
    "WMT",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "CAT",
    "GE",
    "HON",
    "UPS",
    "DE",
    "BA",
    "XOM",
    "CVX",
    "COP",
    "DIS",
    "NFLX",
    "CMCSA",
    "LIN",
    "SHW",
    "APD",
]


SECTORS = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AMZN": "Consumer Discretionary",
    "META": "Communication Services",
    "GOOGL": "Communication Services",
    "TSLA": "Consumer Discretionary",
    "AVGO": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    "AMD": "Technology",
    "QCOM": "Technology",
    "CSCO": "Technology",
    "TXN": "Technology",
    "IBM": "Technology",
    "INTU": "Technology",
    "AMAT": "Technology",
    "LRCX": "Technology",
    "MU": "Technology",
    "KLAC": "Technology",
    "ANET": "Technology",
    "PANW": "Technology",
    "NOW": "Technology",
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "MRK": "Health Care",
    "ABBV": "Health Care",
    "TMO": "Health Care",
    "ABT": "Health Care",
    "ISRG": "Health Care",
    "DHR": "Health Care",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "CAT": "Industrials",
    "GE": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "DE": "Industrials",
    "BA": "Industrials",
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "DIS": "Communication Services",
    "NFLX": "Communication Services",
    "CMCSA": "Communication Services",
    "LIN": "Materials",
    "SHW": "Materials",
    "APD": "Materials",
}


CONCEPTS = {
    "eps_diluted": ("us-gaap", ["EarningsPerShareDiluted"]),
    "net_income": ("us-gaap", ["NetIncomeLoss", "ProfitLoss"]),
    "operating_cash_flow": (
        "us-gaap",
        ["NetCashProvidedByUsedInOperatingActivities"],
    ),
    "total_assets": ("us-gaap", ["Assets"]),
    "total_liabilities": ("us-gaap", ["Liabilities"]),
    "cash_and_equivalents": (
        "us-gaap",
        [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ],
    ),
    "short_term_debt": (
        "us-gaap",
        ["ShortTermBorrowings", "ShortTermDebt", "ShortTermDebtCurrent"],
    ),
    "long_term_debt": ("us-gaap", ["LongTermDebt", "LongTermDebtNoncurrent"]),
    "common_shares_outstanding": (
        "dei",
        ["EntityCommonStockSharesOutstanding"],
    ),
    "repurchase_cash": (
        "us-gaap",
        ["PaymentsForRepurchaseOfCommonStock", "PaymentsForRepurchaseOfEquity"],
    ),
    "sga_expense": ("us-gaap", ["SellingGeneralAndAdministrativeExpense"]),
    "research_and_development": (
        "us-gaap",
        [
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
        ],
    ),
    "revenue": (
        "us-gaap",
        ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    ),
    "shareholder_equity": (
        "us-gaap",
        ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    ),
}

YTD_FLOW_COLUMNS = {"operating_cash_flow", "repurchase_cash"}


@dataclass(frozen=True)
class BacktestResult:
    monthly_returns: pd.DataFrame
    summary: dict[str, float]
    ic: pd.DataFrame
    quantiles: pd.DataFrame
    holdings: pd.DataFrame
    exposures: pd.DataFrame
    factor_panel: pd.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest the non-consensus factor prototype on recent public data."
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-04-29")
    parser.add_argument("--lookback-start", default="2023-01-01")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--min-factor-coverage", type=float, default=0.30)
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--output-dir", default="reports/recent_non_consensus")
    parser.add_argument("--cache-dir", default="reports/cache/recent_non_consensus")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_recent_backtest(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        lookback_start=args.lookback_start,
        top_n=args.top_n,
        min_factor_coverage=args.min_factor_coverage,
        cache_dir=Path(args.cache_dir),
    )

    result.monthly_returns.to_csv(output_dir / "monthly_returns.csv", index=False)
    result.ic.to_csv(output_dir / "factor_ic.csv", index=False)
    result.quantiles.to_csv(output_dir / "quantile_returns.csv", index=False)
    result.holdings.to_csv(output_dir / "holdings.csv", index=False)
    result.exposures.to_csv(output_dir / "exposures.csv", index=False)
    result.factor_panel.to_csv(output_dir / "factor_panel.csv", index=False)

    print("Recent non-consensus factor backtest")
    print(f"period: {args.start} to {args.end}")
    print(f"tickers requested: {len(args.tickers)}")
    print(f"monthly observations: {len(result.monthly_returns)}")
    print("\nSummary")
    for key, value in result.summary.items():
        print(f"{key:>24}: {value: .4f}")
    print("\nIC")
    print(result.ic.to_string(index=False))
    print("\nComposite quantiles")
    print(result.quantiles.to_string(index=False))
    print("\nLatest holdings")
    latest = result.holdings[result.holdings["date"] == result.holdings["date"].max()]
    print(
        latest[
            ["date", "symbol", "sector", "composite_score", "target_weight", "selected_rank"]
        ].to_string(index=False)
    )
    print(f"\nWrote CSV outputs to {output_dir}")


def run_recent_backtest(
    tickers: list[str],
    start: str,
    end: str,
    lookback_start: str,
    top_n: int,
    min_factor_coverage: float,
    cache_dir: Path | None = None,
) -> BacktestResult:
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
    factor_panel = factor_panel[
        (factor_panel["date"] >= start_ts) & (factor_panel["date"] <= end_ts)
    ].copy()

    holdings = select_long_only_portfolio(
        factor_panel,
        top_n=top_n,
        sector_cap=0.25,
        max_weight=0.05,
    )
    forward_returns = make_forward_monthly_returns(prices)
    forward_returns = forward_returns[
        (forward_returns["date"] >= start_ts) & (forward_returns["date"] <= end_ts)
    ].copy()
    monthly_returns = backtest_monthly_holdings(holdings, forward_returns)
    summary = summarize_returns(monthly_returns["portfolio_return"])
    summary.update(
        {
            "benchmark_total_return": total_return(monthly_returns["benchmark_return"]),
            "excess_total_return": total_return(
                monthly_returns["portfolio_return"] - monthly_returns["benchmark_return"]
            ),
            "average_holdings": float(holdings.groupby("date")["symbol"].nunique().mean()),
            "eligible_symbols_last_month": float(
                factor_panel[factor_panel["date"] == factor_panel["date"].max()][
                    "eligible"
                ].sum()
            ),
        }
    )

    factor_cols = ["composite_score"] + [f"{col}_z" for col in RAW_FACTOR_COLUMNS]
    ic = factor_information_coefficient(factor_panel, forward_returns, factor_cols)
    quantiles = quantile_forward_returns(factor_panel, forward_returns)
    exposures = summarize_portfolio_exposures(holdings, factor_panel)
    return BacktestResult(
        monthly_returns=monthly_returns,
        summary=summary,
        ic=ic,
        quantiles=quantiles,
        holdings=holdings,
        exposures=exposures,
        factor_panel=factor_panel,
    )


def fetch_yahoo_prices(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    start_epoch = int(start.replace(tzinfo=timezone.utc).timestamp())
    end_epoch = int((end + pd.Timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp())
    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        print(f"fetching price {ticker}", flush=True)
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?period1={start_epoch}&period2={end_epoch}"
            "&interval=1d&events=history&includeAdjustedClose=true"
        )
        cache_path = (
            cache_dir / "yahoo" / f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}.json"
            if cache_dir
            else None
        )
        data = fetch_json(
            url,
            user_agent="Mozilla/5.0 quant_for_fun research",
            cache_path=cache_path,
        )
        result = data.get("chart", {}).get("result") or []
        if not result:
            print(f"warning: no Yahoo data for {ticker}")
            continue
        result = result[0]
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        adjclose = (result.get("indicators", {}).get("adjclose") or [{}])[0].get(
            "adjclose", []
        )
        if not timestamps:
            continue
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(timestamps, unit="s").date,
                "symbol": ticker,
                "open": quote.get("open", []),
                "high": quote.get("high", []),
                "low": quote.get("low", []),
                "close": adjclose or quote.get("close", []),
                "volume": quote.get("volume", []),
            }
        )
        frame["date"] = pd.to_datetime(frame["date"])
        rows.append(frame.dropna(subset=["close", "volume"]))
        time.sleep(0.05)

    if not rows:
        raise RuntimeError("No price data downloaded.")
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"])


def fetch_cik_map(cache_dir: Path | None = None) -> dict[str, int]:
    data = fetch_json(
        "https://www.sec.gov/files/company_tickers.json",
        user_agent="quant_for_fun research contact@example.com",
        cache_path=cache_dir / "sec" / "company_tickers.json" if cache_dir else None,
    )
    return {record["ticker"].upper(): int(record["cik_str"]) for record in data.values()}


def fetch_sec_fundamentals(
    tickers: list[str],
    cik_map: dict[str, int],
    start_year: int,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        print(f"fetching SEC facts {ticker}", flush=True)
        cik = cik_map.get(ticker.upper())
        if not cik:
            print(f"warning: missing CIK for {ticker}")
            continue
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
        cache_path = cache_dir / "sec" / f"companyfacts_{cik:010d}.json" if cache_dir else None
        try:
            data = fetch_json(
                url,
                user_agent="quant_for_fun research contact@example.com",
                cache_path=cache_path,
            )
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"warning: SEC facts failed for {ticker}: {exc}")
            continue
        frame = extract_fundamental_frame(ticker, data, start_year=start_year)
        if not frame.empty:
            frames.append(frame)
        time.sleep(0.11)

    if not frames:
        raise RuntimeError("No SEC fundamentals downloaded.")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "fiscal_period_end"])


def extract_fundamental_frame(
    ticker: str,
    data: dict,
    start_year: int,
) -> pd.DataFrame:
    concept_frames = {
        output_col: extract_concept(
            data,
            namespace,
            concepts,
            start_year,
            allow_ytd=output_col in YTD_FLOW_COLUMNS,
        )
        for output_col, (namespace, concepts) in CONCEPTS.items()
    }
    for output_col in YTD_FLOW_COLUMNS:
        if output_col in concept_frames and not concept_frames[output_col].empty:
            concept_frames[output_col] = quarterize_ytd_flow(concept_frames[output_col])
    base = concept_frames["net_income"].copy()
    if base.empty:
        base = concept_frames["eps_diluted"].copy()
    if base.empty:
        return pd.DataFrame()

    base = base.rename(columns={"value": "net_income"}) if "value" in base.columns else base
    base = base[["accn", "fiscal_period_end", "filing_date", "form", "fp", "net_income"]]

    for output_col, frame in concept_frames.items():
        if output_col == "net_income" or frame.empty:
            continue
        values = frame[["accn", "value"]].rename(columns={"value": output_col})
        values = values.drop_duplicates("accn", keep="last")
        base = base.merge(values, on="accn", how="left")

    base["symbol"] = ticker
    return base


def extract_concept(
    data: dict,
    namespace: str,
    concepts: list[str],
    start_year: int,
    allow_ytd: bool = False,
) -> pd.DataFrame:
    for concept in concepts:
        fact = data.get("facts", {}).get(namespace, {}).get(concept)
        if not fact:
            continue
        rows = []
        for values in fact.get("units", {}).values():
            for row in values:
                filed = row.get("filed")
                end = row.get("end")
                accn = row.get("accn")
                value = row.get("val")
                if not filed or not end or not accn or value is None:
                    continue
                form = row.get("form", "")
                if form not in {"10-Q", "10-K"}:
                    continue
                end_ts = pd.Timestamp(end)
                if end_ts.year < start_year:
                    continue
                frame = str(row.get("frame", ""))
                is_instant = "start" not in row
                if namespace == "dei" or is_instant:
                    if frame and not frame.endswith("I"):
                        continue
                else:
                    if frame and frame.endswith("I"):
                        continue
                    start = pd.Timestamp(row["start"])
                    if not allow_ytd and (end_ts - start).days > 120:
                        continue
                rows.append(
                    {
                        "accn": accn,
                        "fiscal_period_end": end_ts,
                        "filing_date": pd.Timestamp(filed),
                        "form": form,
                        "fy": row.get("fy"),
                        "fp": row.get("fp", ""),
                        "value": float(value),
                    }
                )
        if rows:
            frame = pd.DataFrame(rows)
            return frame.sort_values(["filing_date", "fiscal_period_end"]).drop_duplicates(
                "accn", keep="last"
            )
    return pd.DataFrame()


def quarterize_ytd_flow(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    period_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 4}
    result["period_order"] = result["fp"].map(period_order)
    result = result[result["period_order"].notna()].copy()
    rows = []
    for fy, fy_frame in result.groupby("fy", sort=True):
        fy_frame = fy_frame.sort_values(["period_order", "filing_date"]).copy()
        previous_value = np.nan
        previous_order = np.nan
        for _, row in fy_frame.iterrows():
            value = row["value"]
            order = row["period_order"]
            if np.isfinite(previous_value) and order > previous_order:
                row = row.copy()
                row["value"] = value - previous_value
            rows.append(row.drop(labels=["period_order"]))
            if order >= previous_order or not np.isfinite(previous_order):
                previous_value = value
                previous_order = order
    if not rows:
        return result.drop(columns=["period_order"])
    return pd.DataFrame(rows)


def attach_market_cap_to_fundamentals(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    result = fundamentals.copy()
    result["filing_date"] = pd.to_datetime(result["filing_date"]).astype("datetime64[ns]")
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    price_lookup = prices[["date", "symbol", "close"]].sort_values(["symbol", "date"])
    parts = []
    for symbol, frame in result.groupby("symbol", sort=False):
        sf = frame.sort_values("filing_date").copy()
        px = price_lookup[price_lookup["symbol"] == symbol]
        if px.empty:
            sf["market_cap"] = np.nan
            parts.append(sf)
            continue
        aligned = pd.merge_asof(
            sf,
            px,
            left_on="filing_date",
            right_on="date",
            by="symbol",
            direction="backward",
        )
        aligned["market_cap"] = aligned["close"] * aligned["common_shares_outstanding"]
        aligned = aligned.drop(columns=["date", "close"], errors="ignore")
        parts.append(aligned)
    return pd.concat(parts, ignore_index=True)


def attach_market_cap_to_prices(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    parts = []
    prices = prices.copy()
    fundamentals = fundamentals.copy()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    fundamentals["filing_date"] = pd.to_datetime(fundamentals["filing_date"]).astype(
        "datetime64[ns]"
    )
    shares = fundamentals[
        ["symbol", "filing_date", "common_shares_outstanding"]
    ].dropna()
    for symbol, frame in prices.groupby("symbol", sort=False):
        sf = frame.sort_values("date").copy()
        symbol_shares = shares[shares["symbol"] == symbol].sort_values("filing_date")
        if symbol_shares.empty:
            sf["market_cap"] = np.nan
            parts.append(sf)
            continue
        aligned = pd.merge_asof(
            sf,
            symbol_shares,
            left_on="date",
            right_on="filing_date",
            by="symbol",
            direction="backward",
        )
        aligned["common_shares_outstanding"] = aligned[
            "common_shares_outstanding"
        ].fillna(symbol_shares["common_shares_outstanding"].iloc[0])
        aligned["market_cap"] = aligned["close"] * aligned["common_shares_outstanding"]
        aligned = aligned.drop(columns=["filing_date", "common_shares_outstanding"])
        parts.append(aligned)
    return pd.concat(parts, ignore_index=True)


def make_security_master(tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": tickers,
            "security_type": "Common Stock",
            "sector": [SECTORS.get(ticker, "Unknown") for ticker in tickers],
            "is_etf": False,
            "is_adr": False,
            "is_preferred": False,
            "is_otc": False,
        }
    )


def backtest_monthly_holdings(
    holdings: pd.DataFrame,
    forward_returns: pd.DataFrame,
    trading_cost_bps: float = 25.0,
) -> pd.DataFrame:
    if holdings.empty:
        return pd.DataFrame(columns=["date", "portfolio_return", "benchmark_return"])

    joined = holdings.merge(forward_returns, on=["date", "symbol"], how="left")
    joined["weighted_return"] = joined["target_weight"] * joined["forward_return"].fillna(0.0)
    portfolio_returns = joined.groupby("date")["weighted_return"].sum().rename(
        "gross_portfolio_return"
    )
    benchmark_returns = forward_returns.groupby("date")["forward_return"].mean().rename(
        "benchmark_return"
    )
    turnover = estimate_turnover(holdings).rename("turnover")
    monthly = pd.concat([portfolio_returns, benchmark_returns, turnover], axis=1).dropna(
        subset=["gross_portfolio_return"]
    )
    monthly["cost"] = monthly["turnover"].fillna(0.0) * trading_cost_bps / 10_000
    monthly["portfolio_return"] = monthly["gross_portfolio_return"] - monthly["cost"]
    monthly.index.name = "date"
    return monthly.reset_index()


def estimate_turnover(holdings: pd.DataFrame) -> pd.Series:
    rows = {}
    previous = pd.Series(dtype=float)
    for date, date_frame in holdings.groupby("date", sort=True):
        current = date_frame.set_index("symbol")["target_weight"]
        all_symbols = previous.index.union(current.index)
        rows[date] = 0.5 * (
            current.reindex(all_symbols, fill_value=0.0)
            - previous.reindex(all_symbols, fill_value=0.0)
        ).abs().sum()
        previous = current
    return pd.Series(rows)


def summarize_returns(returns: pd.Series, periods_per_year: int = 12) -> dict[str, float]:
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
    annual_return = equity.iloc[-1] ** (periods_per_year / len(clean)) - 1
    annual_volatility = clean.std(ddof=0) * np.sqrt(periods_per_year)
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(annual_return / annual_volatility) if annual_volatility else 0.0,
        "max_drawdown": float(drawdown.min()),
    }


def total_return(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float((1 + clean).prod() - 1)


def fetch_json(url: str, user_agent: str, cache_path: Path | None = None) -> dict:
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text())

    last_error: Exception | None = None
    for attempt in range(4):
        request = Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"})
        try:
            with urlopen(request, timeout=30) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                text = body.decode("utf-8")
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(text)
                return json.loads(text)
        except (HTTPError, URLError, TimeoutError, http.client.IncompleteRead) as exc:
            last_error = exc
            time.sleep(0.75 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError("unreachable fetch_json retry state")


if __name__ == "__main__":
    main()
