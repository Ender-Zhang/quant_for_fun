from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_AGENTS = ("a2c", "ddpg", "ppo", "td3", "sac")
CONTRIB_AGENTS = ("trpo", "tqc", "crossq", "recurrent_ppo")
RECURRENT_AGENTS = ("recurrent_ppo",)
SUPPORTED_AGENTS = DEFAULT_AGENTS + CONTRIB_AGENTS
QUICK_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "JPM", "UNH", "HD", "CAT")
POPULAR_100_TICKERS = (
    "NVDA",
    "GOOGL",
    "GOOG",
    "AAPL",
    "MSFT",
    "AMZN",
    "AVGO",
    "META",
    "TSLA",
    "WMT",
    "MU",
    "AMD",
    "ASML",
    "INTC",
    "COST",
    "NFLX",
    "CSCO",
    "PLTR",
    "LRCX",
    "AMAT",
    "TXN",
    "LIN",
    "KLAC",
    "ARM",
    "PEP",
    "TMUS",
    "ADI",
    "QCOM",
    "AMGN",
    "SHOP",
    "GILD",
    "STX",
    "ISRG",
    "APP",
    "PANW",
    "WDC",
    "MRVL",
    "PDD",
    "HON",
    "BKNG",
    "SBUX",
    "CRWD",
    "CEG",
    "INTU",
    "VRTX",
    "ADBE",
    "CMCSA",
    "CDNS",
    "MAR",
    "MELI",
    "SNPS",
    "ADP",
    "ABNB",
    "CSX",
    "ORLY",
    "MDLZ",
    "MPWR",
    "DASH",
    "MNST",
    "NXPI",
    "AEP",
    "ROST",
    "REGN",
    "BKR",
    "CTAS",
    "WBD",
    "FTNT",
    "MSTR",
    "PCAR",
    "FANG",
    "FAST",
    "ADSK",
    "XEL",
    "MCHP",
    "EA",
    "DDOG",
    "FER",
    "EXC",
    "PYPL",
    "IDXX",
    "ODFL",
    "CCEP",
    "TRI",
    "TTWO",
    "KDP",
    "ALNY",
    "ROP",
    "PAYX",
    "WDAY",
    "AXON",
    "CPRT",
    "INSM",
    "GEHC",
    "KHC",
    "CTSH",
    "CHTR",
    "VRSK",
    "DXCM",
    "TEAM",
    "ZS",
)
POPULAR_EXTRA_TICKERS = ("ISRG",)
FINRL_REPO_URL = "https://github.com/AI4Finance-Foundation/FinRL.git"


@dataclass(frozen=True)
class ExperimentConfig:
    preset: str
    train_start: str
    train_end: str
    trade_start: str
    trade_end: str
    timesteps: int
    universe: str


PRESETS = {
    "quick": ExperimentConfig(
        preset="quick",
        train_start="2020-01-01",
        train_end="2025-01-01",
        trade_start="2025-01-01",
        trade_end="2026-03-20",
        timesteps=1_000,
        universe="quick",
    ),
    "release": ExperimentConfig(
        preset="release",
        train_start="2014-01-06",
        train_end="2025-12-31",
        trade_start="2026-01-01",
        trade_end="2026-03-20",
        timesteps=20_000,
        universe="dow30",
    ),
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preset = PRESETS[args.preset]

    project_root = Path(__file__).resolve().parents[2]
    finrl_root = resolve_path(args.finrl_root, project_root)
    data_dir = resolve_path(args.data_dir, project_root)
    report_dir = resolve_path(args.report_dir, project_root)
    model_dir = report_dir / "trained_models"
    log_dir = report_dir / "sb3_logs"

    ensure_finrl_checkout(finrl_root, clone_if_missing=not args.no_clone)
    sys.path.insert(0, str(finrl_root))

    agents = parse_agent_list(args.agents)
    universe = args.universe or preset.universe
    data_source = resolve_data_source(args.data_source, universe)
    tickers = resolve_tickers(args.tickers, universe)
    config = {
        **asdict(preset),
        "train_start": args.train_start or preset.train_start,
        "train_end": args.train_end or preset.train_end,
        "trade_start": args.trade_start or preset.trade_start,
        "trade_end": args.trade_end or preset.trade_end,
        "timesteps": args.timesteps or preset.timesteps,
        "universe": universe,
        "data_source": data_source,
        "agents": agents,
        "tickers": tickers,
        "use_vix": False if data_source == "akshare" else not args.no_vix,
        "use_turbulence": args.use_turbulence,
        "initial_amount": args.initial_amount,
        "trading_cost_pct": args.trading_cost_pct,
        "seed": args.seed,
        "finrl_root": str(finrl_root),
        "data_dir": str(data_dir),
        "report_dir": str(report_dir),
    }

    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = project_root / "reports" / "_cache"
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_dir / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    (report_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    if args.stage in {"all", "data"}:
        run_data_stage(config, data_dir)

    if args.stage in {"all", "train"}:
        run_train_stage(config, data_dir, model_dir, log_dir)

    if args.stage in {"all", "backtest"}:
        run_backtest_stage(config, data_dir, model_dir, report_dir)

    print(f"\nFinRL experiment finished. Artifacts: {report_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an isolated FinRL 2026 stock-trading experiment."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="quick",
        help="quick is faster; release mirrors the FinRL 2026 tutorial defaults.",
    )
    parser.add_argument(
        "--stage", choices=("all", "data", "train", "backtest"), default="all"
    )
    parser.add_argument("--agents", default=",".join(DEFAULT_AGENTS))
    parser.add_argument("--universe", choices=("quick", "dow30", "popular100"), default=None)
    parser.add_argument("--data-source", choices=("auto", "yahoo", "akshare"), default="auto")
    parser.add_argument("--tickers", default=None, help="Comma-separated custom tickers.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--trade-start", default=None)
    parser.add_argument("--trade-end", default=None)
    parser.add_argument("--initial-amount", type=float, default=1_000_000)
    parser.add_argument("--trading-cost-pct", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-vix", action="store_true")
    parser.add_argument("--use-turbulence", action="store_true")
    parser.add_argument("--no-clone", action="store_true")
    parser.add_argument("--finrl-root", default="external_projects/FinRL")
    parser.add_argument("--data-dir", default="data/processed/finrl_2026")
    parser.add_argument("--report-dir", default="reports/finrl_2026")
    return parser


def parse_agent_list(value: str) -> list[str]:
    agents = [agent.strip().lower() for agent in value.split(",") if agent.strip()]
    unknown = sorted(set(agents) - set(SUPPORTED_AGENTS))
    if unknown:
        raise ValueError(f"Unsupported agents: {', '.join(unknown)}")
    return agents


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def ensure_finrl_checkout(finrl_root: Path, clone_if_missing: bool) -> None:
    if (finrl_root / "finrl").is_dir():
        return
    if not clone_if_missing:
        raise FileNotFoundError(
            f"FinRL checkout not found at {finrl_root}. Remove --no-clone or clone it."
        )
    finrl_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", FINRL_REPO_URL, str(finrl_root)],
        check=True,
    )


def resolve_tickers(custom_tickers: str | None, universe: str) -> list[str]:
    if custom_tickers:
        return [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
    if universe == "quick":
        return list(QUICK_TICKERS)
    if universe == "popular100":
        return dedupe_tickers([*POPULAR_100_TICKERS, *POPULAR_EXTRA_TICKERS])
    from finrl import config_tickers

    return list(config_tickers.DOW_30_TICKER)


def dedupe_tickers(tickers: list[str] | tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(ticker.upper() for ticker in tickers))


def resolve_data_source(value: str, universe: str) -> str:
    if value != "auto":
        return value
    return "akshare" if universe == "popular100" else "yahoo"


def run_data_stage(config: dict, data_dir: Path) -> None:
    import itertools

    from finrl.config import INDICATORS
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer
    from finrl.meta.preprocessor.preprocessors import data_split

    print("\n=== FinRL data stage ===")
    if config["data_source"] == "akshare":
        raw = fetch_akshare_us_data(
            tickers=config["tickers"],
            start_date=config["train_start"],
            end_date=config["trade_end"],
        )
    else:
        from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

        raw = YahooDownloader(
            start_date=config["train_start"],
            end_date=config["trade_end"],
            ticker_list=config["tickers"],
        ).fetch_data()
    validate_ticker_coverage(raw, config["tickers"], f"raw {config['data_source']} data")
    if config["data_source"] == "akshare":
        raw = align_full_ticker_history(raw, config["tickers"])

    feature_engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=config["use_vix"],
        use_turbulence=config["use_turbulence"],
        user_defined_feature=False,
    )
    processed = feature_engineer.preprocess_data(raw)
    validate_ticker_coverage(processed, config["tickers"], "processed feature data")

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"]).fillna(0)

    train = data_split(processed_full, config["train_start"], config["train_end"])
    trade = data_split(processed_full, config["trade_start"], config["trade_end"])

    raw.to_csv(data_dir / "raw_data.csv", index=False)
    processed_full.to_csv(data_dir / "processed_full.csv", index=False)
    train.to_csv(data_dir / "train_data.csv")
    trade.to_csv(data_dir / "trade_data.csv")
    print(f"Saved train/trade data to {data_dir}")


def fetch_akshare_us_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    import akshare as ak

    frames = []
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    for ticker in tickers:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                frame = ak.stock_us_daily(symbol=ticker, adjust="qfq")
                normalized = normalize_akshare_us_daily(frame, ticker, start, end)
                if not normalized.empty:
                    frames.append(normalized)
                    print(f"AkShare downloaded {ticker}: {len(normalized)} rows")
                    break
            except Exception as error:  # noqa: BLE001 - surface provider failures with ticker context
                last_error = error
                time.sleep(min(attempt, 3))
        else:
            try:
                frame = ak.stock_us_hist(
                    symbol=f"105.{ticker}",
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="qfq",
                )
                normalized = normalize_akshare_us_hist(frame, ticker, start, end)
                if not normalized.empty:
                    frames.append(normalized)
                    print(f"AkShare fallback downloaded {ticker}: {len(normalized)} rows")
                    continue
            except Exception as error:  # noqa: BLE001
                last_error = error
            raise ValueError(f"AkShare failed to download {ticker}: {last_error}") from last_error

    if not frames:
        raise ValueError("AkShare returned no US stock data.")
    return pd.concat(frames, ignore_index=True).sort_values(["date", "tic"])


def normalize_akshare_us_daily(
    frame: pd.DataFrame, ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    expected = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"AkShare daily frame for {ticker} missing columns: {missing}")
    normalized = frame[expected].copy()
    return normalize_ohlcv_frame(normalized, ticker, start, end)


def normalize_akshare_us_hist(
    frame: pd.DataFrame, ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    rename = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
    }
    missing = [column for column in rename if column not in frame.columns]
    if missing:
        raise ValueError(f"AkShare hist frame for {ticker} missing columns: {missing}")
    normalized = frame.rename(columns=rename)[list(rename.values())].copy()
    return normalize_ohlcv_frame(normalized, ticker, start, end)


def normalize_ohlcv_frame(
    frame: pd.DataFrame, ticker: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized = normalized[(normalized["date"] >= start) & (normalized["date"] < end)]
    normalized = normalized.dropna(subset=["open", "high", "low", "close", "volume"])
    normalized = normalized.sort_values("date")
    normalized["date"] = normalized["date"].dt.strftime("%Y-%m-%d")
    normalized["tic"] = ticker
    normalized["day"] = pd.to_datetime(normalized["date"]).dt.dayofweek
    return normalized[["date", "close", "high", "low", "open", "volume", "tic", "day"]]


def align_full_ticker_history(frame: pd.DataFrame, expected_tickers: list[str]) -> pd.DataFrame:
    first_dates = frame.groupby("tic")["date"].min()
    missing = sorted(set(expected_tickers) - set(first_dates.index))
    if missing:
        raise ValueError(f"Cannot align history; missing tickers: {', '.join(missing)}")
    common_start = first_dates.max()
    aligned = frame[frame["date"] >= common_start].copy()
    counts_by_date = aligned.groupby("date")["tic"].nunique()
    full_dates = counts_by_date[counts_by_date == len(expected_tickers)].index
    aligned = aligned[aligned["date"].isin(full_dates)].sort_values(["date", "tic"])
    if aligned.empty:
        raise ValueError("Cannot align history; no dates contain all expected tickers.")
    print(
        "Aligned AkShare data to common ticker history: "
        f"{aligned['date'].min()} -> {aligned['date'].max()} "
        f"({len(full_dates)} trading days, {len(expected_tickers)} tickers)"
    )
    return aligned


def validate_ticker_coverage(frame: pd.DataFrame, expected_tickers: list[str], label: str) -> None:
    observed = set(frame["tic"].dropna().unique())
    expected = set(expected_tickers)
    missing = sorted(expected - observed)
    if missing:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(
            f"{label} is missing {len(missing)} expected tickers: {preview}{suffix}"
        )


def run_train_stage(config: dict, data_dir: Path, model_dir: Path, log_dir: Path) -> None:
    from stable_baselines3.common.logger import configure

    from finrl.config import INDICATORS
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    print("\n=== FinRL train stage ===")
    train = read_finrl_csv(data_dir / "train_data.csv")
    env_kwargs = make_env_kwargs(train, INDICATORS, config)
    train_env = StockTradingEnv(df=train, **env_kwargs)
    sb3_env, _ = train_env.get_sb_env()

    for agent_name in config["agents"]:
        print(f"\n--- Training {agent_name.upper()} for {config['timesteps']} timesteps ---")
        model = make_model(
            agent_name,
            env=sb3_env,
            timesteps=config["timesteps"],
            seed=config["seed"],
        )
        model.set_logger(configure(str(log_dir / agent_name), ["stdout", "csv"]))
        trained = model.learn(
            total_timesteps=config["timesteps"],
            tb_log_name=agent_name,
        )
        trained.save(str(model_dir / f"agent_{agent_name}"))
        print(f"Saved {agent_name.upper()} model to {model_dir}")


def run_backtest_stage(
    config: dict, data_dir: Path, model_dir: Path, report_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.config import INDICATORS
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    print("\n=== FinRL backtest stage ===")
    train = read_finrl_csv(data_dir / "train_data.csv")
    trade = read_finrl_csv(data_dir / "trade_data.csv")
    env_kwargs = make_env_kwargs(trade, INDICATORS, config)
    model_classes = load_model_classes(config["agents"])
    result = pd.DataFrame()

    for agent_name in config["agents"]:
        model_path = model_dir / f"agent_{agent_name}"
        if not model_path.with_suffix(".zip").exists():
            print(f"Skipping {agent_name.upper()}: {model_path}.zip not found")
            continue
        model = model_classes[agent_name].load(str(model_path))
        trade_env = StockTradingEnv(df=trade, **env_kwargs)
        if agent_name in RECURRENT_AGENTS:
            account_value, actions = recurrent_drl_prediction(
                model=model,
                environment=trade_env,
            )
        else:
            account_value, actions = DRLAgent.DRL_prediction(
                model=model,
                environment=trade_env,
            )
        account_value.to_csv(report_dir / f"account_value_{agent_name}.csv", index=False)
        actions.to_csv(report_dir / f"actions_{agent_name}.csv")
        result[agent_name] = account_value.set_index("date")["account_value"]

    result["mvo"] = simple_mvo_curve(train, trade, config["initial_amount"])
    dji = download_dji_curve(config)
    if dji is not None:
        result["dji"] = dji

    result = result.dropna(how="all")
    result.to_csv(report_dir / "backtest_values.csv")

    summary = pd.DataFrame(
        [{"strategy": column, **performance_summary(result[column])} for column in result]
    )
    summary.to_csv(report_dir / "backtest_summary.csv", index=False)
    (report_dir / "backtest_summary.json").write_text(
        summary.to_json(orient="records", indent=2) + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(12, 5))
    result.plot(ax=plt.gca())
    plt.title("FinRL 2026 Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.tight_layout()
    plt.savefig(report_dir / "backtest_result.png", dpi=150)
    plt.close()

    print("\n=== Backtest summary ===")
    print(summary.to_string(index=False))
    print(f"Saved backtest report to {report_dir}")


def read_finrl_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.set_index(frame.columns[0])
    frame.index.names = [""]
    return frame


def make_env_kwargs(train_or_trade: pd.DataFrame, indicators: list[str], config: dict) -> dict:
    stock_dimension = len(train_or_trade.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(indicators) * stock_dimension
    cost = [config["trading_cost_pct"]] * stock_dimension
    return {
        "hmax": 100,
        "initial_amount": config["initial_amount"],
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": cost,
        "sell_cost_pct": cost,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": indicators,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }


def make_model(agent_name: str, env, timesteps: int, seed: int):
    if agent_name in DEFAULT_AGENTS:
        from finrl.agents.stablebaselines3.models import DRLAgent

        agent = DRLAgent(env=env)
        return agent.get_model(
            agent_name,
            policy_kwargs=policy_kwargs(agent_name),
            model_kwargs=model_kwargs(agent_name, timesteps),
            verbose=1,
            seed=seed,
        )

    model_class = load_model_classes([agent_name])[agent_name]
    kwargs = model_kwargs(agent_name, timesteps)
    print(kwargs)
    return model_class(
        policy=policy_name(agent_name),
        env=env,
        verbose=1,
        seed=seed,
        policy_kwargs=policy_kwargs(agent_name),
        **kwargs,
    )


def load_model_classes(agent_names: list[str]) -> dict:
    from stable_baselines3 import A2C
    from stable_baselines3 import DDPG
    from stable_baselines3 import PPO
    from stable_baselines3 import SAC
    from stable_baselines3 import TD3

    classes = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}
    if any(agent_name in CONTRIB_AGENTS for agent_name in agent_names):
        try:
            from sb3_contrib import CrossQ
            from sb3_contrib import RecurrentPPO
            from sb3_contrib import TQC
            from sb3_contrib import TRPO
        except ImportError as exc:
            raise ImportError(
                "Agents trpo, tqc, crossq, and recurrent_ppo require sb3-contrib. "
                'Install with: .venv/bin/python -m pip install -e ".[dev,finrl]"'
            ) from exc
        classes.update(
            {
                "trpo": TRPO,
                "tqc": TQC,
                "crossq": CrossQ,
                "recurrent_ppo": RecurrentPPO,
            }
        )
    return classes


def recurrent_drl_prediction(model, environment, deterministic: bool = True):
    """Run SB3 recurrent policies while preserving LSTM state across trading days."""
    test_env, test_obs = environment.get_sb_env()
    test_obs = test_env.reset()
    lstm_states = None
    episode_starts = np.ones((test_env.num_envs,), dtype=bool)
    account_memory = None
    actions_memory = None
    max_steps = len(environment.df.index.unique()) - 1

    for step in range(len(environment.df.index.unique())):
        action, lstm_states = model.predict(
            test_obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        test_obs, rewards, dones, info = test_env.step(action)
        episode_starts = dones

        if step == max_steps - 1:
            account_memory = test_env.env_method(method_name="save_asset_memory")
            actions_memory = test_env.env_method(method_name="save_action_memory")
        if dones[0]:
            print("hit end!")
            break

    return account_memory[0], actions_memory[0]


def model_kwargs(agent_name: str, timesteps: int) -> dict:
    if agent_name == "a2c":
        return {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
    if agent_name == "ppo":
        n_steps = min(512, max(128, timesteps))
        return {
            "n_steps": n_steps,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 64,
        }
    if agent_name == "ddpg":
        return {"batch_size": 64, "buffer_size": 50_000, "learning_rate": 0.001}
    if agent_name == "td3":
        return {"batch_size": 64, "buffer_size": 50_000, "learning_rate": 0.001}
    if agent_name == "sac":
        return {
            "batch_size": 64,
            "buffer_size": 50_000,
            "learning_rate": 0.0001,
            "learning_starts": min(100, max(10, timesteps // 10)),
            "ent_coef": "auto_0.1",
        }
    if agent_name == "trpo":
        n_steps = min(512, max(128, timesteps))
        return {
            "n_steps": n_steps,
            "batch_size": 64,
            "learning_rate": 0.001,
            "target_kl": 0.01,
        }
    if agent_name == "tqc":
        return {
            "batch_size": 64,
            "buffer_size": 50_000,
            "learning_rate": 0.0003,
            "learning_starts": min(100, max(10, timesteps // 10)),
        }
    if agent_name == "crossq":
        return {
            "batch_size": 64,
            "buffer_size": 50_000,
            "learning_rate": 0.0003,
            "learning_starts": min(100, max(10, timesteps // 10)),
        }
    if agent_name == "recurrent_ppo":
        n_steps = min(512, max(128, timesteps))
        return {
            "n_steps": n_steps,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 64,
        }
    raise ValueError(f"Unsupported agent: {agent_name}")


def policy_kwargs(agent_name: str) -> dict | None:
    if agent_name == "crossq":
        return {"net_arch": [256, 256]}
    return None


def policy_name(agent_name: str) -> str:
    if agent_name == "recurrent_ppo":
        return "MlpLstmPolicy"
    return "MlpPolicy"


def simple_mvo_curve(
    train: pd.DataFrame,
    trade: pd.DataFrame,
    initial_amount: float,
    max_weight: float = 0.5,
) -> pd.Series:
    train_prices = train.pivot(index="date", columns="tic", values="close")
    trade_prices = trade.pivot(index="date", columns="tic", values="close")
    returns = train_prices.pct_change().dropna()
    mu = returns.mean().to_numpy()
    cov = returns.cov().to_numpy()

    raw_weights = np.linalg.pinv(cov).dot(mu)
    raw_weights = np.clip(raw_weights, 0, None)
    if raw_weights.sum() <= 0:
        weights = np.repeat(1 / len(train_prices.columns), len(train_prices.columns))
    else:
        weights = raw_weights / raw_weights.sum()
        weights = np.clip(weights, 0, max_weight)
        weights = weights / weights.sum()

    last_prices = train_prices.tail(1).to_numpy()[0]
    shares = initial_amount * weights / last_prices
    return pd.Series(trade_prices.to_numpy().dot(shares), index=trade_prices.index)


def download_dji_curve(config: dict) -> pd.Series | None:
    try:
        import yfinance as yf

        dji = yf.download(
            "^DJI",
            start=config["trade_start"],
            end=config["trade_end"],
            progress=False,
            auto_adjust=False,
        )
    except Exception as exc:
        print(f"Skipping DJIA baseline: {exc}")
        return None

    if dji.empty or "Close" not in dji:
        print("Skipping DJIA baseline: no data returned")
        return None

    close = dji["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = close.index.astype(str)
    return close.div(close.iloc[0]).mul(config["initial_amount"])


def performance_summary(values: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    clean = values.dropna()
    if len(clean) < 2:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    returns = clean.pct_change().dropna()
    total_return = clean.iloc[-1] / clean.iloc[0] - 1
    annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(periods_per_year)
    drawdown = clean / clean.cummax() - 1
    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe": float(
            annual_return / annual_volatility if annual_volatility > 0 else 0.0
        ),
        "max_drawdown": float(drawdown.min()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
