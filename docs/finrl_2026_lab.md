# FinRL 2026 Lab

这个实验把现成量化/RL 项目隔离在 `external_projects/` 里：

- `external_projects/FinRL`
- `external_projects/FinRL-Trading`
- `external_projects/qlib`
- `external_projects/TradeMaster`

`external_projects/`、`data/processed/` 和 `reports/` 都在 `.gitignore` 中，第三方源码、下载数据、训练模型和回测图不会进入本项目的 git 历史。

每日实战学习安排见 `docs/finrl_daily_practice_plan.md`。

## 环境

推荐使用 Python 3.11：

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,finrl]"
```

## 快速成就感版本

默认 `quick` preset 会使用 8 只高流动性美股，训练 A2C、DDPG、PPO、TD3、SAC，每个 agent 训练 1,000 timesteps：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --preset quick
```

也可以手动加入 `sb3-contrib` 里的扩展 agent：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 \
  --preset quick \
  --agents trpo,tqc,crossq,recurrent_ppo \
  --report-dir reports/finrl_2026_extra_agents
```

这些不是默认运行项，因为 Day 0 目标是先跑通主流程；扩展对比建议单独放到新的
`--report-dir`，避免覆盖默认 5 个 agent 的结果。

输出位置：

- `data/processed/finrl_2026/train_data.csv`
- `data/processed/finrl_2026/trade_data.csv`
- `reports/finrl_2026/trained_models/`
- `reports/finrl_2026/backtest_summary.csv`
- `reports/finrl_2026/backtest_result.png`

## FinRL 2026 Release 版本

更接近 FinRL v0.3.8 教程默认配置，使用 Dow 30，训练 20,000 timesteps：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --preset release
```

如果只想重跑某一步：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --stage data
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --stage train
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --stage backtest
```

## 改模型时优先改这些

- `--agents ppo,sac`：只训练指定 agent。
- `--agents trpo,tqc,crossq,recurrent_ppo`：尝试 SB3 contrib 里的扩展 agent。
- `--timesteps 5000`：增加训练步数。
- `--tickers SPY,QQQ,TLT,GLD`：换成 ETF/多资产版本。
- `--universe popular100`：使用 Nasdaq-100 风格的高热度大盘股票池，并确保包含 ISRG。
- `--data-source akshare`：使用 AkShare 美股行情接口；`--data-source auto` 下 popular100 会默认用 AkShare。
- `--no-vix`：跳过 VIX 下载，减少网络依赖。

这些实验只用于学习和研究，不构成投资建议。真正接近实盘前，至少要额外检查样本外年份、手续费、滑点、换手率、最大回撤和 paper trading 表现。
