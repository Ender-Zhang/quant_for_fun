from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_AGENTS = ("a2c", "ddpg", "ppo", "td3", "sac")
QUICK_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "JPM", "UNH", "HD", "CAT")
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
    tickers = resolve_tickers(args.tickers, args.universe or preset.universe)
    config = {
        **asdict(preset),
        "train_start": args.train_start or preset.train_start,
        "train_end": args.train_end or preset.train_end,
        "trade_start": args.trade_start or preset.trade_start,
        "trade_end": args.trade_end or preset.trade_end,
        "timesteps": args.timesteps or preset.timesteps,
        "universe": args.universe or preset.universe,
        "agents": agents,
        "tickers": tickers,
        "use_vix": not args.no_vix,
        "use_turbulence": args.use_turbulence,
        "initial_amount": args.initial_amount,
        "trading_cost_pct": args.trading_cost_pct,
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
    parser.add_argument("--agents", default=",".join(SUPPORTED_AGENTS))
    parser.add_argument("--universe", choices=("quick", "dow30"), default=None)
    parser.add_argument("--tickers", default=None, help="Comma-separated custom tickers.")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--trade-start", default=None)
    parser.add_argument("--trade-end", default=None)
    parser.add_argument("--initial-amount", type=float, default=1_000_000)
    parser.add_argument("--trading-cost-pct", type=float, default=0.001)
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
    from finrl import config_tickers

    return list(config_tickers.DOW_30_TICKER)


def run_data_stage(config: dict, data_dir: Path) -> None:
    import itertools

    from finrl.config import INDICATORS
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer
    from finrl.meta.preprocessor.preprocessors import data_split
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

    print("\n=== FinRL data stage ===")
    raw = YahooDownloader(
        start_date=config["train_start"],
        end_date=config["trade_end"],
        ticker_list=config["tickers"],
    ).fetch_data()

    feature_engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=config["use_vix"],
        use_turbulence=config["use_turbulence"],
        user_defined_feature=False,
    )
    processed = feature_engineer.preprocess_data(raw)

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


def run_train_stage(config: dict, data_dir: Path, model_dir: Path, log_dir: Path) -> None:
    from stable_baselines3.common.logger import configure

    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.config import INDICATORS
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    print("\n=== FinRL train stage ===")
    train = read_finrl_csv(data_dir / "train_data.csv")
    env_kwargs = make_env_kwargs(train, INDICATORS, config)
    train_env = StockTradingEnv(df=train, **env_kwargs)
    sb3_env, _ = train_env.get_sb_env()

    for agent_name in config["agents"]:
        print(f"\n--- Training {agent_name.upper()} for {config['timesteps']} timesteps ---")
        agent = DRLAgent(env=sb3_env)
        model = agent.get_model(
            agent_name,
            model_kwargs=model_kwargs(agent_name, config["timesteps"]),
            verbose=1,
            seed=42,
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
    from stable_baselines3 import A2C
    from stable_baselines3 import DDPG
    from stable_baselines3 import PPO
    from stable_baselines3 import SAC
    from stable_baselines3 import TD3

    from finrl.agents.stablebaselines3.models import DRLAgent
    from finrl.config import INDICATORS
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

    print("\n=== FinRL backtest stage ===")
    train = read_finrl_csv(data_dir / "train_data.csv")
    trade = read_finrl_csv(data_dir / "trade_data.csv")
    env_kwargs = make_env_kwargs(trade, INDICATORS, config)
    model_classes = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}
    result = pd.DataFrame()

    for agent_name in config["agents"]:
        model_path = model_dir / f"agent_{agent_name}"
        if not model_path.with_suffix(".zip").exists():
            print(f"Skipping {agent_name.upper()}: {model_path}.zip not found")
            continue
        model = model_classes[agent_name].load(str(model_path))
        trade_env = StockTradingEnv(df=trade, **env_kwargs)
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
    raise ValueError(f"Unsupported agent: {agent_name}")


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
