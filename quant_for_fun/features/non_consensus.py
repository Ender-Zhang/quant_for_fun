from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


MAIN_FACTOR_WEIGHTS: dict[str, float] = {
    "pead_sue": 0.15,
    "cash_quality_noa": 0.20,
    "issuance_buyback": 0.15,
    "insider_buy_pressure": 0.10,
    "institutional_breadth": 0.15,
    "text_improvement": 0.10,
    "organization_intangibles": 0.15,
}

RAW_FACTOR_COLUMNS = list(MAIN_FACTOR_WEIGHTS)
SHADOW_FACTOR_COLUMNS = ["customer_momentum_shadow"]
CONTROL_FACTOR_COLUMNS = [
    "size_control",
    "book_to_market_control",
    "momentum_12_1_control",
    "beta_control",
    "volatility_control",
]


def add_sec_reporting_lag(
    data: pd.DataFrame,
    filing_date_col: str = "filing_date",
    available_date_col: str = "available_date",
    lag_trading_days: int = 2,
) -> pd.DataFrame:
    """Add a point-in-time availability date using a conservative SEC lag."""
    if filing_date_col not in data.columns:
        raise ValueError(f"missing required column: {filing_date_col}")

    result = data.copy()
    filing_dates = pd.to_datetime(result[filing_date_col])
    result[available_date_col] = filing_dates + pd.offsets.BDay(lag_trading_days)
    return result


def make_us_large_mid_universe(
    prices: pd.DataFrame,
    securities: pd.DataFrame | None = None,
    min_market_cap: float = 2_000_000_000,
    min_average_dollar_volume: float = 20_000_000,
    adv_window: int = 60,
    min_listing_days: int = 252,
    excluded_sectors: Iterable[str] = ("Financials", "Real Estate"),
) -> pd.DataFrame:
    """Build the monthly eligible universe for liquid US large/mid-cap stocks."""
    panel = _prepare_price_panel(prices)
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    panel["average_dollar_volume_60d"] = _group_rolling_mean(
        panel, "symbol", "dollar_volume", adv_window
    )
    panel["listing_days"] = panel.groupby("symbol").cumcount() + 1

    monthly_dates = _month_end_trading_dates(panel["date"])
    universe = panel[panel["date"].isin(monthly_dates)].copy()

    if securities is not None and not securities.empty:
        static = securities.copy()
        static["symbol"] = static["symbol"].astype(str)
        universe = universe.merge(static, on="symbol", how="left")

    if "sector" not in universe.columns:
        universe["sector"] = "Unknown"
    if "security_type" not in universe.columns:
        universe["security_type"] = "Common Stock"
    for col in ("is_etf", "is_adr", "is_preferred", "is_otc"):
        if col not in universe.columns:
            universe[col] = False

    common_stock = universe["security_type"].fillna("Common Stock").str.lower().str.contains(
        "common"
    )
    excluded_sector = universe["sector"].isin(set(excluded_sectors))
    universe["eligible"] = (
        common_stock
        & ~universe["is_etf"].fillna(False)
        & ~universe["is_adr"].fillna(False)
        & ~universe["is_preferred"].fillna(False)
        & ~universe["is_otc"].fillna(False)
        & ~excluded_sector
        & (universe["market_cap"] >= min_market_cap)
        & (universe["average_dollar_volume_60d"] >= min_average_dollar_volume)
        & (universe["listing_days"] >= min_listing_days)
    )

    columns = [
        "date",
        "symbol",
        "close",
        "volume",
        "market_cap",
        "average_dollar_volume_60d",
        "listing_days",
        "sector",
        "eligible",
    ]
    optional = [c for c in ("industry", "sic") if c in universe.columns]
    return universe[columns + optional].sort_values(["date", "symbol"]).reset_index(drop=True)


def build_non_consensus_factor_panel(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    securities: pd.DataFrame | None = None,
    insider_trades: pd.DataFrame | None = None,
    institutional_holdings: pd.DataFrame | None = None,
    filing_text_metrics: pd.DataFrame | None = None,
    supply_chain_links: pd.DataFrame | None = None,
    min_factor_coverage: float = 0.50,
) -> pd.DataFrame:
    """Create monthly non-consensus alpha factors and conventional risk controls.

    The returned panel is keyed by ``date`` and ``symbol``. Main alpha columns
    are z-scored within sector when enough peers are available, otherwise within
    the full rebalance date. Conventional controls are reported separately and
    do not enter ``composite_score``.
    """
    prices = _prepare_price_panel(prices)
    rebalance_dates = _month_end_trading_dates(prices["date"])
    universe = make_us_large_mid_universe(prices, securities)
    panel = universe.copy()

    factor_frames = [
        _make_pead_sue(fundamentals, rebalance_dates),
        _make_cash_quality_noa(fundamentals, rebalance_dates),
        _make_issuance_buyback(fundamentals, rebalance_dates),
        _make_organization_intangibles(fundamentals, rebalance_dates),
        _make_insider_buy_pressure(insider_trades, rebalance_dates),
        _make_institutional_breadth(institutional_holdings, rebalance_dates),
        _make_text_improvement(filing_text_metrics, rebalance_dates),
        _make_customer_momentum_shadow(supply_chain_links, prices, rebalance_dates),
        make_conventional_risk_controls(prices, fundamentals, rebalance_dates),
    ]

    for frame in factor_frames:
        if not frame.empty:
            panel = panel.merge(frame, on=["date", "symbol"], how="left")

    if "insider_buy_value_90d" in panel.columns:
        panel["insider_buy_pressure"] = panel["insider_buy_value_90d"] / panel[
            "market_cap"
        ].replace(0, np.nan)
        panel = panel.drop(columns=["insider_buy_value_90d"])

    panel = add_factor_zscores(panel, RAW_FACTOR_COLUMNS + SHADOW_FACTOR_COLUMNS)
    panel = add_factor_zscores(panel, CONTROL_FACTOR_COLUMNS, suffix="_z")
    panel = add_composite_score(panel, min_factor_coverage=min_factor_coverage)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def add_factor_zscores(
    panel: pd.DataFrame,
    columns: Iterable[str],
    suffix: str = "_z",
    min_sector_count: int = 5,
) -> pd.DataFrame:
    """Winsorize and z-score factor columns by date, then by sector when possible."""
    result = panel.copy()
    columns = [col for col in columns if col in result.columns]
    if not columns:
        return result

    for col in columns:
        result[f"{col}{suffix}"] = np.nan

    for _, date_frame in result.groupby("date", sort=False):
        date_index = date_frame.index
        date_level_scores = {
            col: _winsorized_zscore(date_frame[col]) for col in columns
        }

        for col in columns:
            result.loc[date_index, f"{col}{suffix}"] = date_level_scores[col]

        if "sector" not in date_frame.columns:
            continue

        for _, sector_frame in date_frame.groupby("sector", sort=False):
            sector_index = sector_frame.index
            for col in columns:
                if sector_frame[col].notna().sum() >= min_sector_count:
                    result.loc[sector_index, f"{col}{suffix}"] = _winsorized_zscore(
                        sector_frame[col]
                    )

    return result


def add_composite_score(
    panel: pd.DataFrame,
    weights: dict[str, float] | None = None,
    min_factor_coverage: float = 0.50,
) -> pd.DataFrame:
    """Combine available main factors while reporting data coverage."""
    result = panel.copy()
    weights = weights or MAIN_FACTOR_WEIGHTS
    total_weight = float(sum(weights.values()))
    weighted_sum = pd.Series(0.0, index=result.index)
    observed_weight = pd.Series(0.0, index=result.index)

    for factor, weight in weights.items():
        z_col = f"{factor}_z"
        if z_col not in result.columns:
            continue
        observed = result[z_col].notna()
        weighted_sum = weighted_sum.add(result[z_col].fillna(0.0) * weight)
        observed_weight = observed_weight.add(observed.astype(float) * weight)

    result["factor_coverage"] = observed_weight / total_weight if total_weight else 0.0
    result["composite_score"] = weighted_sum / observed_weight.replace(0, np.nan)
    result.loc[result["factor_coverage"] < min_factor_coverage, "composite_score"] = np.nan
    return result


def make_conventional_risk_controls(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None = None,
    rebalance_dates: Iterable[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Report common controls without adding them to the alpha score."""
    panel = _prepare_price_panel(prices)
    if rebalance_dates is None:
        rebalance_dates = _month_end_trading_dates(panel["date"])
    rebalance_dates = pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates)))

    panel["return_1d"] = panel.groupby("symbol")["close"].pct_change()
    panel["momentum_12_1_control"] = panel.groupby("symbol")["close"].transform(
        lambda s: s.shift(21) / s.shift(252) - 1
    )
    panel["volatility_control"] = panel.groupby("symbol")["return_1d"].transform(
        lambda s: s.rolling(126, min_periods=60).std() * np.sqrt(252)
    )
    panel["market_return"] = panel.groupby("date")["return_1d"].transform("mean")

    beta_parts: list[pd.DataFrame] = []
    for symbol, symbol_frame in panel.groupby("symbol", sort=False):
        cov = symbol_frame["return_1d"].rolling(252, min_periods=120).cov(
            symbol_frame["market_return"]
        )
        var = symbol_frame["market_return"].rolling(252, min_periods=120).var()
        beta = cov / var.replace(0, np.nan)
        beta_parts.append(
            pd.DataFrame(
                {
                    "date": symbol_frame["date"].to_numpy(),
                    "symbol": symbol,
                    "beta_control": beta.to_numpy(),
                }
            )
        )
    beta_panel = pd.concat(beta_parts, ignore_index=True) if beta_parts else pd.DataFrame()

    controls = panel[panel["date"].isin(rebalance_dates)][
        [
            "date",
            "symbol",
            "market_cap",
            "momentum_12_1_control",
            "volatility_control",
        ]
    ].copy()
    controls["size_control"] = np.log(controls["market_cap"].replace(0, np.nan))
    if not beta_panel.empty:
        controls = controls.merge(beta_panel, on=["date", "symbol"], how="left")

    book_to_market = _make_book_to_market(fundamentals, rebalance_dates)
    if not book_to_market.empty:
        controls = controls.merge(book_to_market, on=["date", "symbol"], how="left")

    return controls.drop(columns=["market_cap"]).sort_values(["date", "symbol"])


def _make_pead_sue(fundamentals: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = _prepare_fundamentals(fundamentals)
    if f.empty:
        return pd.DataFrame(columns=["date", "symbol", "pead_sue"])

    earnings = _first_existing(f, ["eps_diluted", "net_income"])
    f["_earnings_signal"] = earnings
    f["seasonal_surprise"] = f.groupby("symbol")["_earnings_signal"].diff(4)
    surprise_std = f.groupby("symbol")["seasonal_surprise"].transform(
        lambda s: s.rolling(8, min_periods=4).std()
    )
    f["pead_sue"] = f["seasonal_surprise"] / surprise_std.replace(0, np.nan)
    f["expires_date"] = f["available_date"] + pd.offsets.BDay(60)
    return _align_events_to_dates(f, dates, ["pead_sue"], expires_col="expires_date")


def _make_cash_quality_noa(fundamentals: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = _prepare_fundamentals(fundamentals)
    needed = {"net_income", "operating_cash_flow", "total_assets"}
    if f.empty or not needed.issubset(f.columns):
        return pd.DataFrame(columns=["date", "symbol", "cash_quality_noa"])

    f["lag_total_assets"] = f.groupby("symbol")["total_assets"].shift(1)
    f["average_assets"] = (f["total_assets"] + f["lag_total_assets"]) / 2
    accruals = (f["net_income"] - f["operating_cash_flow"]) / f[
        "average_assets"
    ].replace(0, np.nan)

    total_liabilities = _optional_number(f, "total_liabilities", 0.0)
    cash = _optional_number(f, "cash_and_equivalents", 0.0)
    short_debt = _optional_number(f, "short_term_debt", 0.0)
    long_debt = _optional_number(f, "long_term_debt", 0.0)
    operating_assets = f["total_assets"] - cash
    operating_liabilities = total_liabilities - short_debt - long_debt
    net_operating_assets = (operating_assets - operating_liabilities) / f[
        "total_assets"
    ].replace(0, np.nan)

    f["cash_quality_noa"] = -0.6 * accruals - 0.4 * net_operating_assets
    return _align_events_to_dates(f, dates, ["cash_quality_noa"])


def _make_issuance_buyback(fundamentals: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    f = _prepare_fundamentals(fundamentals)
    share_col = "common_shares_outstanding"
    if f.empty or share_col not in f.columns:
        return pd.DataFrame(columns=["date", "symbol", "issuance_buyback"])

    f["shares_12m_ago"] = f.groupby("symbol")[share_col].shift(4)
    share_change = f[share_col] / f["shares_12m_ago"].replace(0, np.nan) - 1

    if "repurchase_cash" in f.columns and "market_cap" in f.columns:
        f["repurchase_cash_ttm"] = f.groupby("symbol")["repurchase_cash"].transform(
            lambda s: s.rolling(4, min_periods=1).sum()
        )
        buyback_yield = f["repurchase_cash_ttm"] / f["market_cap"].replace(0, np.nan)
    else:
        buyback_yield = 0.0

    f["issuance_buyback"] = -share_change + buyback_yield
    return _align_events_to_dates(f, dates, ["issuance_buyback"])


def _make_insider_buy_pressure(
    insider_trades: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
    window_days: int = 90,
) -> pd.DataFrame:
    if insider_trades is None or insider_trades.empty:
        return pd.DataFrame(columns=["date", "symbol", "insider_buy_value_90d"])

    trades = add_sec_reporting_lag(insider_trades)
    trades["symbol"] = trades["symbol"].astype(str)
    if "transaction_code" not in trades.columns:
        trades["transaction_code"] = ""
    if "acquisition_disposition" not in trades.columns:
        trades["acquisition_disposition"] = "A"
    trades["transaction_code"] = trades["transaction_code"].astype(str).str.upper()
    trades["acquisition_disposition"] = trades["acquisition_disposition"].astype(str).str.upper()
    if "is_derivative" not in trades.columns:
        trades["is_derivative"] = False
    if "is_10b5_1" not in trades.columns:
        trades["is_10b5_1"] = False

    open_market_buy = (
        (trades["transaction_code"] == "P")
        & (trades["acquisition_disposition"] == "A")
        & ~trades["is_derivative"].fillna(False)
        & ~trades["is_10b5_1"].fillna(False)
    )
    trades = trades[open_market_buy].copy()
    if trades.empty:
        return pd.DataFrame(columns=["date", "symbol", "insider_buy_value_90d"])

    if "owner_role" not in trades.columns:
        trades["owner_role"] = ""
    role = trades["owner_role"].astype(str).str.lower()
    role_weight = pd.Series(0.75, index=trades.index)
    role_weight.loc[role.str.contains("ceo|chief executive|cfo|chief financial")] = 1.5
    role_weight.loc[role.str.contains("director|chair")] = 1.0
    trades["weighted_buy_value"] = trades["shares"] * trades["price"] * role_weight
    return _rolling_event_sum(
        trades,
        dates,
        value_col="weighted_buy_value",
        output_col="insider_buy_value_90d",
        window_days=window_days,
    )


def _make_institutional_breadth(
    holdings: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if holdings is None or holdings.empty:
        return pd.DataFrame(columns=["date", "symbol", "institutional_breadth"])

    h = add_sec_reporting_lag(holdings)
    h["symbol"] = h["symbol"].astype(str)
    counts = (
        h.groupby(["symbol", "report_date", "available_date"])["manager_id"]
        .nunique()
        .reset_index(name="holder_count")
    )
    counts = counts.sort_values(["symbol", "report_date", "available_date"])
    counts["previous_holder_count"] = counts.groupby("symbol")["holder_count"].shift(1)
    counts["institutional_breadth"] = (
        counts["holder_count"] / counts["previous_holder_count"].replace(0, np.nan) - 1
    )
    return _align_events_to_dates(counts, dates, ["institutional_breadth"])


def _make_text_improvement(
    filing_text_metrics: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if filing_text_metrics is None or filing_text_metrics.empty:
        return pd.DataFrame(columns=["date", "symbol", "text_improvement"])

    text = _prepare_events(filing_text_metrics)
    if "total_words" not in text.columns:
        return pd.DataFrame(columns=["date", "symbol", "text_improvement"])

    total_words = text["total_words"].replace(0, np.nan)
    text["negative_ratio"] = _optional_number(text, "negative_word_count", 0.0) / total_words
    text["uncertainty_ratio"] = (
        _optional_number(text, "uncertainty_word_count", 0.0) / total_words
    )
    text["litigious_ratio"] = _optional_number(text, "litigious_word_count", 0.0) / total_words
    text["log_total_words"] = np.log(total_words)

    components = [
        "negative_ratio",
        "uncertainty_ratio",
        "litigious_ratio",
        "log_total_words",
    ]
    for col in components:
        text[f"{col}_change"] = text.groupby("symbol")[col].diff()

    deterioration = (
        text["negative_ratio_change"]
        + text["uncertainty_ratio_change"]
        + text["litigious_ratio_change"]
        + 0.25 * text["log_total_words_change"]
    )
    text["text_improvement"] = -deterioration
    return _align_events_to_dates(text, dates, ["text_improvement"])


def _make_organization_intangibles(
    fundamentals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    sga_depreciation: float = 0.20,
    rd_depreciation: float = 0.15,
) -> pd.DataFrame:
    f = _prepare_fundamentals(fundamentals)
    if f.empty or "total_assets" not in f.columns:
        return pd.DataFrame(columns=["date", "symbol", "organization_intangibles"])

    parts: list[pd.DataFrame] = []
    for symbol, symbol_frame in f.groupby("symbol", sort=False):
        sf = symbol_frame.sort_values(["fiscal_period_end", "available_date"]).copy()
        sga_values = _optional_number(sf, "sga_expense", 0.0).fillna(0.0)
        rd_values = _optional_number(sf, "research_and_development", 0.0).fillna(0.0)

        org_capital: list[float] = []
        rd_capital: list[float] = []
        org_prev = 0.0
        rd_prev = 0.0
        for sga, rd in zip(sga_values, rd_values, strict=False):
            org_prev = (1 - sga_depreciation) * org_prev + float(sga)
            rd_prev = (1 - rd_depreciation) * rd_prev + float(rd)
            org_capital.append(org_prev)
            rd_capital.append(rd_prev)

        sf["organization_capital"] = org_capital
        sf["rd_capital"] = rd_capital
        if "revenue" in sf.columns:
            sf["revenue_growth_1y"] = (
                sf["revenue"] / sf["revenue"].shift(4).replace(0, np.nan) - 1
            )
        else:
            sf["revenue_growth_1y"] = np.nan
        rd_intensity = sf["rd_capital"] / sf["total_assets"].replace(0, np.nan)
        rd_productivity = sf["revenue_growth_1y"] / rd_intensity.replace(0, np.nan)
        sf["organization_intangibles"] = (
            0.7 * sf["organization_capital"] / sf["total_assets"].replace(0, np.nan)
            + 0.3 * rd_productivity
        )
        sf["symbol"] = symbol
        parts.append(sf)

    events = pd.concat(parts, ignore_index=True) if parts else f
    return _align_events_to_dates(events, dates, ["organization_intangibles"])


def _make_customer_momentum_shadow(
    supply_chain_links: pd.DataFrame | None,
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if supply_chain_links is None or supply_chain_links.empty:
        return pd.DataFrame(columns=["date", "symbol", "customer_momentum_shadow"])

    links = supply_chain_links.copy()
    required = {"supplier_symbol", "customer_symbol", "effective_date"}
    if not required.issubset(links.columns):
        return pd.DataFrame(columns=["date", "symbol", "customer_momentum_shadow"])

    px = _prepare_price_panel(prices)
    px["return_21d"] = px.groupby("symbol")["close"].pct_change(21)
    customer_returns = px[px["date"].isin(dates)][["date", "symbol", "return_21d"]].rename(
        columns={"symbol": "customer_symbol"}
    )
    links["effective_date"] = pd.to_datetime(links["effective_date"])

    rows: list[dict[str, object]] = []
    for date, date_returns in customer_returns.groupby("date", sort=False):
        active_links = links[links["effective_date"] <= date]
        if active_links.empty:
            continue
        linked = active_links.merge(date_returns, on="customer_symbol", how="left")
        supplier_signal = linked.groupby("supplier_symbol")["return_21d"].mean()
        for supplier, value in supplier_signal.items():
            rows.append(
                {
                    "date": date,
                    "symbol": supplier,
                    "customer_momentum_shadow": value,
                }
            )
    return pd.DataFrame(rows)


def _make_book_to_market(
    fundamentals: pd.DataFrame | None,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame(columns=["date", "symbol", "book_to_market_control"])

    f = _prepare_fundamentals(fundamentals)
    if "book_equity" in f.columns:
        book_equity = f["book_equity"]
    elif "shareholder_equity" in f.columns:
        book_equity = f["shareholder_equity"]
    elif {"total_assets", "total_liabilities"}.issubset(f.columns):
        book_equity = f["total_assets"] - f["total_liabilities"]
    else:
        return pd.DataFrame(columns=["date", "symbol", "book_to_market_control"])

    if "market_cap" not in f.columns:
        return pd.DataFrame(columns=["date", "symbol", "book_to_market_control"])

    f["book_to_market_control"] = book_equity / f["market_cap"].replace(0, np.nan)
    return _align_events_to_dates(f, dates, ["book_to_market_control"])


def _prepare_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    panel = prices.copy()
    if "date" not in panel.columns:
        if isinstance(panel.index, pd.MultiIndex):
            panel = panel.reset_index()
        else:
            panel = panel.reset_index().rename(columns={"index": "date"})
    if "symbol" not in panel.columns:
        panel["symbol"] = "SYNTH"
    required = {"date", "symbol", "close", "volume"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"prices missing required columns: {sorted(missing)}")
    if "market_cap" not in panel.columns:
        panel["market_cap"] = np.nan
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def _prepare_fundamentals(fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame()
    f = add_sec_reporting_lag(fundamentals)
    f["symbol"] = f["symbol"].astype(str)
    if "fiscal_period_end" in f.columns:
        f["fiscal_period_end"] = pd.to_datetime(f["fiscal_period_end"])
    else:
        f["fiscal_period_end"] = pd.to_datetime(f["filing_date"])
    return f.sort_values(["symbol", "fiscal_period_end", "available_date"]).reset_index(
        drop=True
    )


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    result = add_sec_reporting_lag(events)
    result["symbol"] = result["symbol"].astype(str)
    return result.sort_values(["symbol", "available_date"]).reset_index(drop=True)


def _align_events_to_dates(
    events: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    value_cols: list[str],
    expires_col: str | None = None,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["date", "symbol", *value_cols])

    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()
    rows: list[pd.DataFrame] = []
    for symbol, symbol_events in events.groupby("symbol", sort=False):
        source = symbol_events.sort_values("available_date").drop_duplicates(
            "available_date", keep="last"
        )
        source = source.set_index("available_date")
        keep_cols = value_cols + ([expires_col] if expires_col else [])
        aligned = source[keep_cols].reindex(dates, method="ffill")
        aligned["date"] = dates
        aligned["symbol"] = symbol
        if expires_col is not None:
            active = aligned[expires_col].notna() & (aligned["date"] <= aligned[expires_col])
            aligned.loc[~active, value_cols] = np.nan
            aligned = aligned.drop(columns=[expires_col])
        rows.append(aligned.reset_index(drop=True))

    output = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return output[["date", "symbol", *value_cols]]


def _rolling_event_sum(
    events: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    value_col: str,
    output_col: str,
    window_days: int,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).sort_values()
    rows: list[dict[str, object]] = []
    for symbol, symbol_events in events.groupby("symbol", sort=False):
        symbol_events = symbol_events.sort_values("available_date")
        for date in dates:
            start = date - pd.Timedelta(days=window_days)
            in_window = symbol_events[
                (symbol_events["available_date"] <= date)
                & (symbol_events["available_date"] > start)
            ]
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    output_col: in_window[value_col].sum() if not in_window.empty else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _month_end_trading_dates(dates: pd.Series) -> pd.DatetimeIndex:
    date_series = pd.Series(pd.to_datetime(dates).sort_values().unique())
    month_period = date_series.dt.to_period("M")
    return pd.DatetimeIndex(date_series.groupby(month_period).max().to_numpy())


def _group_rolling_mean(
    panel: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
) -> pd.Series:
    return (
        panel.groupby(group_col, group_keys=False)[value_col]
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _winsorized_zscore(values: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    if clean.notna().sum() < 2:
        return pd.Series(np.nan, index=values.index)
    lo = clean.quantile(lower)
    hi = clean.quantile(upper)
    clipped = clean.clip(lo, hi)
    std = clipped.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index).where(clean.notna(), np.nan)
    return (clipped - clipped.mean()) / std


def _first_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for col in columns:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _optional_number(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)
