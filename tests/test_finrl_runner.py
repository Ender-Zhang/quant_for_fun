from pathlib import Path

import pytest
import pandas as pd

from quant_for_fun.examples.run_finrl_2026 import PRESETS
from quant_for_fun.examples.run_finrl_2026 import align_full_ticker_history
from quant_for_fun.examples.run_finrl_2026 import build_parser
from quant_for_fun.examples.run_finrl_2026 import normalize_akshare_us_daily
from quant_for_fun.examples.run_finrl_2026 import resolve_data_source
from quant_for_fun.examples.run_finrl_2026 import resolve_tickers
from quant_for_fun.examples.run_finrl_2026 import parse_agent_list
from quant_for_fun.examples.run_finrl_2026 import resolve_path
from quant_for_fun.examples.run_finrl_2026 import validate_ticker_coverage


def test_parse_agent_list_normalizes_supported_agents() -> None:
    assert parse_agent_list("PPO, sac,td3,recurrent_ppo") == [
        "ppo",
        "sac",
        "td3",
        "recurrent_ppo",
    ]


def test_parse_agent_list_rejects_unknown_agents() -> None:
    with pytest.raises(ValueError, match="Unsupported agents"):
        parse_agent_list("ppo,not_real")


def test_quick_preset_is_small_enough_for_first_run() -> None:
    assert PRESETS["quick"].timesteps < PRESETS["release"].timesteps
    assert PRESETS["quick"].universe == "quick"


def test_popular100_universe_includes_isrg_once() -> None:
    tickers = resolve_tickers(None, "popular100")
    assert "ISRG" in tickers
    assert tickers.count("ISRG") == 1
    assert len(tickers) == 100


def test_auto_data_source_uses_akshare_for_popular100() -> None:
    assert resolve_data_source("auto", "popular100") == "akshare"
    assert resolve_data_source("auto", "quick") == "yahoo"
    assert resolve_data_source("yahoo", "popular100") == "yahoo"


def test_normalize_akshare_us_daily_matches_finrl_schema() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2019-12-31", "2020-01-02", "2020-01-03", "2026-03-20"],
            "open": [1, 2, 3, 4],
            "high": [2, 3, 4, 5],
            "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10, 20, 30, 40],
        }
    )
    normalized = normalize_akshare_us_daily(
        frame, "AAPL", pd.Timestamp("2020-01-01"), pd.Timestamp("2026-03-20")
    )
    assert normalized.columns.tolist() == [
        "date",
        "close",
        "high",
        "low",
        "open",
        "volume",
        "tic",
        "day",
    ]
    assert normalized["date"].tolist() == ["2020-01-02", "2020-01-03"]
    assert normalized["tic"].tolist() == ["AAPL", "AAPL"]
    assert normalized["day"].tolist() == [3, 4]


def test_align_full_ticker_history_uses_latest_first_date() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
            "tic": ["OLD", "OLD", "NEW", "OLD", "NEW"],
            "close": [1, 2, 3, 4, 5],
        }
    )
    aligned = align_full_ticker_history(frame, ["OLD", "NEW"])
    assert aligned["date"].tolist() == ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]
    assert aligned["tic"].tolist() == ["NEW", "OLD", "NEW", "OLD"]


def test_validate_ticker_coverage_rejects_missing_tickers() -> None:
    frame = pd.DataFrame({"tic": ["AAPL", "MSFT"]})
    with pytest.raises(ValueError, match="missing 1 expected tickers"):
        validate_ticker_coverage(frame, ["AAPL", "MSFT", "ISRG"], "raw Yahoo data")


def test_seed_argument_defaults_to_reproducible_value() -> None:
    args = build_parser().parse_args([])
    assert args.seed == 42


def test_resolve_path_keeps_external_projects_isolated() -> None:
    project_root = Path("/tmp/quant_for_fun")
    assert resolve_path("external_projects/FinRL", project_root) == (
        project_root / "external_projects/FinRL"
    )
