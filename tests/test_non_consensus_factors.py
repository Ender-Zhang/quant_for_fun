import numpy as np
import pandas as pd

from quant_for_fun.features.non_consensus import (
    RAW_FACTOR_COLUMNS,
    add_sec_reporting_lag,
    build_non_consensus_factor_panel,
)
from quant_for_fun.portfolio import (
    factor_information_coefficient,
    make_forward_monthly_returns,
    select_long_only_portfolio,
    summarize_portfolio_exposures,
)


def test_sec_reporting_lag_uses_two_business_days() -> None:
    filings = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "filing_date": [pd.Timestamp("2024-01-05")],
        }
    )

    lagged = add_sec_reporting_lag(filings)

    assert lagged.loc[0, "available_date"] == pd.Timestamp("2024-01-09")


def test_non_consensus_panel_selection_and_diagnostics() -> None:
    prices = _make_price_panel()
    fundamentals = _make_fundamentals()
    securities = _make_security_master()
    insiders = _make_insider_trades()
    holdings = _make_13f_holdings()
    text = _make_text_metrics()
    links = pd.DataFrame(
        {
            "supplier_symbol": ["T00", "T01"],
            "customer_symbol": ["I05", "I06"],
            "effective_date": [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-01")],
        }
    )

    panel = build_non_consensus_factor_panel(
        prices,
        fundamentals,
        securities=securities,
        insider_trades=insiders,
        institutional_holdings=holdings,
        filing_text_metrics=text,
        supply_chain_links=links,
    )

    assert set(RAW_FACTOR_COLUMNS).issubset(panel.columns)
    assert "customer_momentum_shadow" in panel.columns
    assert "composite_score" in panel.columns
    assert "size_control_z" in panel.columns
    assert panel.loc[panel["symbol"] == "F10", "eligible"].eq(False).all()
    assert panel["composite_score"].notna().any()

    portfolio = select_long_only_portfolio(
        panel,
        top_n=4,
        sector_cap=0.50,
        max_weight=0.30,
    )

    assert not portfolio.empty
    assert portfolio.groupby("date").size().max() <= 4
    assert portfolio.groupby(["date", "sector"]).size().max() <= 2
    assert portfolio["target_weight"].max() <= 0.30

    forward_returns = make_forward_monthly_returns(prices)
    ic = factor_information_coefficient(
        panel,
        forward_returns,
        ["composite_score", "pead_sue_z", "cash_quality_noa_z"],
    )
    exposures = summarize_portfolio_exposures(portfolio, panel)

    assert set(ic["factor"]) == {"composite_score", "pead_sue_z", "cash_quality_noa_z"}
    assert "size_control_z" in exposures.columns
    assert not exposures.empty


def _symbols() -> list[str]:
    return [f"T{i:02d}" for i in range(5)] + [f"I{i:02d}" for i in range(5, 10)] + ["F10"]


def _make_price_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", periods=760)
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for i, symbol in enumerate(_symbols()):
        base_price = 40 + i * 3
        drift = 0.00015 + i * 0.00002
        returns = drift + 0.005 * np.sin(np.arange(len(dates)) / (18 + i))
        close = base_price * np.cumprod(1 + returns)
        volume = 1_200_000 + i * 80_000
        shares = 80_000_000 + i * 3_000_000
        for date, price in zip(dates, close, strict=False):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": volume,
                    "market_cap": price * shares,
                }
            )
    return pd.DataFrame(rows)


def _make_security_master() -> pd.DataFrame:
    rows = []
    for symbol in _symbols():
        if symbol == "F10":
            sector = "Financials"
        elif symbol.startswith("T"):
            sector = "Technology"
        else:
            sector = "Industrials"
        rows.append(
            {
                "symbol": symbol,
                "security_type": "Common Stock",
                "sector": sector,
                "industry": "Testing",
                "is_etf": False,
                "is_adr": False,
                "is_preferred": False,
                "is_otc": False,
            }
        )
    return pd.DataFrame(rows)


def _make_fundamentals() -> pd.DataFrame:
    fiscal_dates = pd.date_range("2017-03-31", periods=14, freq="QE")
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for i, symbol in enumerate(_symbols()):
        for q, fiscal_date in enumerate(fiscal_dates):
            total_assets = 1_000_000_000 + i * 80_000_000 + q * 20_000_000
            net_income = 35_000_000 + i * 2_000_000 + q * (1_000_000 + i * 150_000)
            operating_cash_flow = net_income + (8_000_000 - i * 250_000)
            shares = 90_000_000 + i * 2_000_000 - q * (120_000 + i * 15_000)
            market_cap = 3_500_000_000 + i * 250_000_000 + q * 30_000_000
            rows.append(
                {
                    "symbol": symbol,
                    "fiscal_period_end": fiscal_date,
                    "filing_date": fiscal_date + pd.offsets.BDay(32),
                    "eps_diluted": 0.40 + i * 0.03 + q * (0.015 + i * 0.001),
                    "net_income": net_income,
                    "operating_cash_flow": operating_cash_flow,
                    "total_assets": total_assets,
                    "total_liabilities": total_assets * 0.45,
                    "cash_and_equivalents": total_assets * 0.08,
                    "short_term_debt": total_assets * 0.03,
                    "long_term_debt": total_assets * 0.16,
                    "common_shares_outstanding": shares,
                    "repurchase_cash": max(0.0, i - 2) * 3_000_000,
                    "market_cap": market_cap,
                    "sga_expense": 55_000_000 + i * 1_500_000,
                    "research_and_development": 12_000_000 + i * 1_000_000,
                    "revenue": 240_000_000 + i * 12_000_000 + q * (6_000_000 + i * 100_000),
                    "shareholder_equity": total_assets * 0.55,
                }
            )
    return pd.DataFrame(rows)


def _make_insider_trades() -> pd.DataFrame:
    rows = []
    for i, symbol in enumerate(_symbols()[:10]):
        rows.append(
            {
                "symbol": symbol,
                "filing_date": pd.Timestamp("2020-06-15") + pd.offsets.BDay(i),
                "transaction_code": "P",
                "acquisition_disposition": "A",
                "is_derivative": False,
                "is_10b5_1": False,
                "owner_role": "CEO" if i % 2 == 0 else "Director",
                "shares": 8_000 + i * 500,
                "price": 40 + i,
            }
        )
    return pd.DataFrame(rows)


def _make_13f_holdings() -> pd.DataFrame:
    report_dates = pd.date_range("2018-03-31", periods=12, freq="QE")
    rows = []
    for i, symbol in enumerate(_symbols()[:10]):
        for q, report_date in enumerate(report_dates):
            holder_count = 4 + (i % 4) + q // 3
            for manager in range(holder_count):
                rows.append(
                    {
                        "symbol": symbol,
                        "report_date": report_date,
                        "filing_date": report_date + pd.offsets.BDay(45),
                        "manager_id": f"M{manager:03d}",
                        "shares": 10_000 + manager * 100,
                    }
                )
    return pd.DataFrame(rows)


def _make_text_metrics() -> pd.DataFrame:
    filing_dates = pd.date_range("2018-03-31", periods=12, freq="QE")
    rows = []
    for i, symbol in enumerate(_symbols()[:10]):
        for q, filing_date in enumerate(filing_dates):
            total_words = 12_000 + i * 100 + q * 80
            rows.append(
                {
                    "symbol": symbol,
                    "filing_date": filing_date + pd.offsets.BDay(34),
                    "total_words": total_words,
                    "negative_word_count": 360 + i * 8 - q * 2,
                    "uncertainty_word_count": 210 + i * 5 - q,
                    "litigious_word_count": 90 + i * 3,
                }
            )
    return pd.DataFrame(rows)
