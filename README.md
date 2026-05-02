# quant_for_fun

一个用来学习、练习和训练量化金融模型的小型实验项目。

目标不是一开始就追求复杂策略，而是先建立一个稳定闭环：

1. 获取或生成价格数据
2. 构造特征和预测标签
3. 训练一个基础模型
4. 把模型信号转换成仓位
5. 做简单回测和指标评估
6. 逐步替换为真实数据、更多因子、更严格的验证方式

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m quant_for_fun.examples.train_baseline
```

如果你暂时不想安装依赖，也可以先阅读 `docs/learning_path.md` 和源码结构。

## 项目结构

```text
quant_for_fun/
  data/          数据加载与模拟数据
  features/      因子和标签构造
  models/        模型训练与预测
  backtest/      回测与绩效指标
  examples/      可直接运行的实验脚本
docs/            学习路线、实验记录模板
tests/           基础测试
```

## 当前内置实验

`python -m quant_for_fun.examples.train_baseline`

这个脚本会：

- 生成一条模拟价格序列
- 构造动量、波动率、均线距离等特征
- 训练一个 `RandomForestClassifier` 预测未来收益方向
- 用预测概率生成 long / flat 策略信号
- 输出累计收益、年化收益、最大回撤、Sharpe 等指标

## FinRL 2026 外部实验

如果你想先复用现成的金融 RL 项目，项目支持把第三方仓库隔离放在
`external_projects/`，再通过一个轻量入口跑 FinRL v0.3.8 的 2026 股票交易实验：

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,finrl]"
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --preset quick
```

默认 quick 模式会训练 A2C、DDPG、PPO、TD3、SAC，并把数据、模型和回测图写到
`data/processed/finrl_2026/` 与 `reports/finrl_2026/`。这两个目录和
`external_projects/` 都不会进入 git。更多说明见 `docs/finrl_2026_lab.md`。

## 非大众化美股选股因子

项目现在也包含一条面向美股大中盘、多头月度选股的因子研究链路：

- `quant_for_fun.features.non_consensus.build_non_consensus_factor_panel`
  从价格、SEC point-in-time 基本面、Form 4、13F、10-K/10-Q 文本指标和供应链链接中构造因子。
- 主因子包括 PEAD/SUE、应计项与 NOA、净股本发行/回购执行、内部人真实买入、机构持有人广度、文本恶化、组织资本/R&D。
- 客户动量作为 `customer_momentum_shadow` 输出，默认不进入综合分。
- 常规 `size / book-to-market / 12-1 momentum / beta / volatility` 只作为风险暴露诊断，不进入 alpha 打分。
- `quant_for_fun.portfolio.select_long_only_portfolio` 可按综合分选出月度多头组合，并控制行业上限和单票上限。
- `python -m quant_for_fun.examples.backtest_recent_non_consensus`
  会下载 Yahoo Finance 价格和 SEC companyfacts，跑一个最近年份的快速公开数据回测。
- `python -m quant_for_fun.examples.train_recent_non_consensus_model`
  用历史月份做 walk-forward Ridge 横截面排序模型，并把模型组合与手工综合分组合对比。

这些函数默认对 SEC 披露使用 `filing_date + 2` 个交易日的可用日期，避免把尚未公开的数据放进历史信号。

## 建议练习顺序

先把 baseline 跑通，然后按这个顺序改：

1. 修改 `quant_for_fun/data/synthetic.py` 里的市场过程参数
2. 在 `quant_for_fun/features/technical.py` 添加新特征
3. 调整 `quant_for_fun/models/baseline.py` 的模型和训练窗口
4. 修改 `quant_for_fun/backtest/vectorized.py` 的交易成本或仓位规则
5. 接入真实行情数据，并把 train/test 切分改成 walk-forward

注意：这里的代码用于学习和研究，不构成投资建议。

## 每日学习计划

如果你希望每天按固定节奏学习，可以从 `docs/daily_study_plan.md` 开始。它按 8 周安排了金融概念、策略练习、因子研究、机器学习建模、walk-forward 验证和真实数据接入。
