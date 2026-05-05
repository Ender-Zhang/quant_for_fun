# FinRL 每日实战计划

目标：用 FinRL 和本项目现有入口，快速跑通金融 RL 交易流程，然后每天围绕真实实验迭代。最终目标不是相信某条漂亮曲线，而是建立一套有资格 paper trading、再小额实盘的研究流程。

这不是投资建议，也不保证赚钱。这里说的“能挣钱”，指的是你逐步达到一组可验证门槛：样本外不过度崩、paper trading 稳定、小资金实盘风险可控。

## 当前项目入口

快速跑通：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --preset quick
```

只跑指定 agent：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --preset quick --agents ppo,sac
```

换成 ETF 股票池：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 \
  --preset quick \
  --tickers SPY,QQQ,TLT,GLD,IWM,EFA,EEM,SHY \
  --agents ppo,sac \
  --timesteps 5000
```

换成 popular100 股票池：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 \
  --preset quick \
  --universe popular100 \
  --data-source akshare \
  --agents recurrent_ppo \
  --timesteps 5000
```

说明：`popular100` 使用 Nasdaq-100 风格的大盘高流动性股票池，并确保包含 ISRG；
`--data-source auto` 下 popular100 会默认使用 AkShare，减少 Yahoo 批量下载不稳定的问题。

输出重点看：

- `reports/finrl_2026/backtest_summary.csv`
- `reports/finrl_2026/backtest_result.png`
- `reports/finrl_2026/actions_*.csv`
- `reports/finrl_2026/account_value_*.csv`
- `reports/finrl_2026/experiment_config.json`

## 近期待办

按这个顺序推进，先保证每一步都有可复现记录，再进入下一步：

- 跑一遍默认 quick preset，确认 A2C、DDPG、PPO、TD3、SAC 的主流程仍能完整生成 summary 和曲线。
- 用 `--universe popular100 --data-source akshare --agents recurrent_ppo` 跑一次小步数实验，确认 100 只股票的数据覆盖、日期对齐和 ISRG 纳入结果。
- 单独跑 `trpo,tqc,crossq,recurrent_ppo` 扩展 agent，并把输出放到 `reports/finrl_2026_extra_agents`，不要覆盖默认结果。
- 对比默认 5 个 agent、扩展 agent、MVO 和 DJI 的 total return、Sharpe、max drawdown，记录到 `docs/journal/YYYY-MM-DD.md`。
- 把表现最好的组合拆成年份回测，至少覆盖 2021、2022、2023、2024、2025。
- 检查 `actions_*.csv`，记录是否存在过度交易、单资产集中或回撤期间继续加仓。
- 根据回测结果决定下一轮只改一个变量：股票池、seed、timesteps、交易成本、reward 或风控。

## 每日固定节奏

每天 60-120 分钟即可。不要每天都大改，重点是留下可比较的实验结果。

1. 复盘昨天结果：看收益、Sharpe、最大回撤、是否跑赢基准。
2. 只改一个变量：股票池、时间窗口、agent、timesteps、成本、reward 或风控。
3. 跑实验：保留命令和输出路径。
4. 写记录：结论要能指导明天做什么。
5. 不做真钱交易：直到 paper trading 阶段达标。

建议每天记录到 `docs/journal/YYYY-MM-DD.md`：

```text
日期：
问题：
改动：
命令：
数据区间：
agent：
关键指标：
是否跑赢基准：
最大回撤：
观察：
明天只改什么：
```

## 第 0 天：环境和第一条曲线

目标：不理解也先跑通，获得第一张图。

任务：

- 确认环境安装：

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,finrl]"
```

- 跑 quick preset：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --preset quick
```

- 打开 `reports/finrl_2026/backtest_summary.csv` 和 `backtest_result.png`。

产出：

- 记录哪个 agent 表现最好。
- 记录它是否跑赢 `mvo` 和 `dji`。
- 写一句话解释：为什么这还不能说明能实盘赚钱。

过关标准：

- 本地能完整生成数据、模型、summary 和图。

## 第 1 周：把 FinRL 流程看懂

### Day 1：数据从哪里来

学习：

- Yahoo Finance 日线数据字段。
- FinRL 数据格式：`date`, `tic`, `open`, `high`, `low`, `close`, `volume`。

实战：

- 查看 `data/processed/finrl_2026/train_data.csv`。
- 数一数 ticker、日期范围、每个 ticker 是否行数一致。

产出：

- 写下训练集和交易集的起止日期。
- 写下每个 ticker 的数据行数。

### Day 2：技术指标是什么

学习：

- `macd`, `rsi_30`, `cci_30`, `dx_30`, `boll_ub`, `boll_lb`。
- 技术指标只是状态特征，不是魔法信号。

实战：

- 在 `train_data.csv` 中挑一个 ticker，看这些指标前 20 行。
- 对比 `close` 和 `rsi_30` 的变化。

产出：

- 写下你觉得最容易理解的 3 个指标。
- 写下哪些指标可能滞后。

### Day 3：环境 state/action/reward

学习：

- state：现金、价格、持仓、技术指标。
- action：每个资产买入/卖出的强度。
- reward：组合资产变化。

实战：

- 阅读 `external_projects/FinRL/finrl/meta/env_stock_trading/env_stocktrading.py`。
- 找到 `step`, `_buy_stock`, `_sell_stock`, `_initiate_state`。

产出：

- 用自己的话写出：agent 每天到底在决定什么。

### Day 4：5 个 agent 先别玄学

学习：

- A2C/PPO：on-policy，训练更直接。
- DDPG/TD3/SAC：off-policy，连续动作常用。
- SAC 通常探索能力更强，但不代表金融上一定更好。

实战：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --preset quick --agents a2c,ppo,sac
```

可选扩展：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 \
  --preset quick \
  --agents trpo,tqc,crossq,recurrent_ppo \
  --report-dir reports/finrl_2026_extra_agents
```

说明：TRPO、TQC、CrossQ 可以复用当前连续动作交易环境；RecurrentPPO 用 LSTM 给 PPO 增加
时间记忆；GRPO 主要来自大语言模型推理训练，并不是当前 FinRL 股票交易环境的即插即用 agent。

产出：

- 对比 A2C/PPO/SAC 的 total return、Sharpe、max drawdown。

### Day 5：换 ETF 池

目标：从个股短期噪声切到更适合实战的多资产组合。

实战：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 \
  --preset quick \
  --tickers SPY,QQQ,TLT,GLD,IWM,EFA,EEM,SHY \
  --agents ppo,sac \
  --timesteps 2000
```

产出：

- 对比 ETF 池和原 quick 股票池的回撤。
- 记录哪个更稳。

### Day 6：重复实验

目标：判断结果是不是运气。

实战：

- 先重复跑同一命令。
- 观察结果是否完全一致。
- 如果未来加了 seed 参数，要跑 5 个 seed。

产出：

- 写下“单次回测为什么不可靠”。

### Day 7：周复盘

产出一份短报告：

- 本周跑了哪些命令。
- 哪个组合表现最好。
- 哪个基准最难打败。
- 最大回撤是否可接受。
- 下周最该补的验证是什么。

过关标准：

- 你能解释 `data -> env -> agent -> backtest`。
- 你知道当前最好结果还没有实盘意义。

## 第 2 周：让回测更可信

### Day 8：只跑 backtest

学习：

- 训练和回测要分离。
- 以后改评估时不应反复重新训练。

实战：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 --stage backtest --preset quick
```

产出：

- 确认不重新训练也能生成 summary。

### Day 9：提高交易成本

目标：看策略是否靠频繁交易假象盈利。

实战：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 \
  --preset quick \
  --agents ppo,sac \
  --trading-cost-pct 0.002
```

产出：

- 对比 0.1% 和 0.2% 成本下收益变化。

### Day 10：观察 actions

学习：

- action 文件能告诉你模型是否过度交易。

实战：

- 打开 `reports/finrl_2026/actions_sac.csv`。
- 观察每天各资产买卖数量。
- 找出交易最频繁的资产。

产出：

- 写下是否存在频繁反向交易。

### Day 11：年度拆分

目标：不要只看一段总曲线。

实战：

- 用不同 `--trade-start` 和 `--trade-end` 跑 2021、2022、2023、2024、2025。
- 每次保留不同 `--report-dir`，例如 `reports/finrl_2026_2022`。

示例：

```bash
.venv/bin/python -m quant_for_fun.examples.run_finrl_2026 \
  --preset quick \
  --train-start 2016-01-01 \
  --train-end 2022-01-01 \
  --trade-start 2022-01-01 \
  --trade-end 2023-01-01 \
  --agents ppo,sac \
  --report-dir reports/finrl_2026_2022 \
  --data-dir data/processed/finrl_2026_2022
```

产出：

- 建一个表：年份、最佳 agent、收益、Sharpe、最大回撤、是否跑赢基准。

### Day 12：基准意识

学习：

- 打赢现金不够。
- 打赢指数不够。
- 至少要和 MVO、等权、简单动量比较。

实战：

- 记录 `mvo`、`dji` 和最优 RL agent 的差距。

产出：

- 写下当前 RL 到底赢在哪里，输在哪里。

### Day 13：失败样本

目标：主动找策略表现最差的时候。

实战：

- 找回测曲线中最大回撤区间。
- 查看那段时间市场发生了什么。
- 看 actions 文件是否还在加仓。

产出：

- 写出一个风险假设：例如 VIX 高时降仓、单资产仓位上限、回撤熔断。

### Day 14：周复盘

过关标准：

- 你有至少 3 个不同时间窗口的实验结果。
- 你知道哪个年份策略最容易失效。
- 你开始关注回撤和交易行为，而不只是收益。

## 第 3 周：walk-forward

目标：把训练方式改成更接近真实研究。

### Day 15：理解 walk-forward

学习：

- 不能用未来训练过去。
- 训练窗口向前滚动，测试下一年。

实战：

- 设计窗口：
  - train 2016-2020, trade 2021
  - train 2017-2021, trade 2022
  - train 2018-2022, trade 2023
  - train 2019-2023, trade 2024
  - train 2020-2024, trade 2025

产出：

- 写出你准备跑的窗口表。

### Day 16-18：手动跑 3 个窗口

每天跑一个窗口，每个窗口只用 `ppo,sac`，先别贪多。

产出：

- 每天补一行结果表。

### Day 19：整理 walk-forward 表

产出：

- 统计有几年的 RL 跑赢 `mvo`。
- 统计有几年的 RL 跑赢 `dji`。
- 记录最差年份最大回撤。

### Day 20：决定是否继续 RL

判断：

- 如果多数年份输给简单基准，先别调大模型。
- 如果收益不错但回撤太大，优先改风控。
- 如果某一年特别差，先解释失败原因。

产出：

- 写一句结论：继续调模型、改 reward，还是先做基准策略。

### Day 21：周复盘

过关标准：

- 至少 3 个 walk-forward 测试窗口。
- 有一张汇总表。
- 有一个清晰的下一步假设。

## 第 4 周：reward 和风控

目标：从“赚更多”转为“亏得少、活得久”。

### Day 22：reward 设计

学习：

- 纯收益 reward 容易鼓励高换手和高风险。
- 更实用的 reward：收益 - 成本 - 换手惩罚 - 回撤惩罚。

实战：

- 阅读 FinRL env 的 reward 位置。
- 暂时不改源码，先写出 reward 设计方案。

产出：

- 一段 reward 公式和解释。

### Day 23：仓位上限

学习：

- 单资产过度集中会造成大回撤。

实战：

- 检查 actions 和 account value。
- 估算是否某些资产长期主导组合。

产出：

- 写出单资产最大权重建议，例如 25% 或 35%。

### Day 24：VIX 风控

学习：

- 高波动环境下，策略可以降仓而不是硬预测。

实战：

- 跑一次默认 VIX。
- 再跑一次 `--no-vix`。
- 对比回撤。

产出：

- 判断 VIX 对当前结果是帮助还是噪声。

### Day 25：成本敏感性表

实战：

- 分别跑 `--trading-cost-pct 0.0005`, `0.001`, `0.002`, `0.005`。

产出：

- 成本敏感性表。
- 如果成本稍微增加策略就失效，暂时不考虑实盘。

### Day 26：降低换手的想法

实战：

- 从 actions 文件估算换手。
- 记录交易最频繁的日期。

产出：

- 写出两个降低换手方案：action 阈值、调仓频率、换手惩罚等。

### Day 27-28：实现或请 Codex 实现一个小风控改动

候选改动：

- 增加换手率统计。
- 增加 yearly summary。
- 增加 walk-forward runner。
- 增加 `--seed` 参数。

过关标准：

- 不是只调模型，而是增强实验框架。

## 第 5 周：模型训练

目标：开始认真训练，但每次只改一个维度。

### Day 29：timesteps 曲线

实战：

- 跑 `1000`, `5000`, `10000`, `20000` timesteps。
- 先只跑 `sac` 或 `ppo`。

产出：

- 观察训练更久是否真的更好。

### Day 30：PPO 参数

实战：

- 调整 `n_steps`, `batch_size`, `learning_rate`。
- 每次只改一个参数。

产出：

- 一张参数和指标表。

### Day 31：SAC 参数

实战：

- 调整 `learning_rate`, `buffer_size`, `learning_starts`, `ent_coef`。

产出：

- 找出最稳定的一组参数，不只看最高收益。

### Day 32：seed 实验

目标：确认不是随机初始化带来的幻觉。

实战：

- 如果还没有 `--seed`，当天优先加这个参数。
- 同一配置跑 5 个 seed。

产出：

- 平均收益、中位数收益、最差回撤。

### Day 33：模型选择规则

写一个固定选择规则：

- 先看最大回撤。
- 再看 Sharpe。
- 再看总收益。
- 必须和基准比较。

产出：

- 明确以后不再用“最高收益”单独选模型。

### Day 34-35：周复盘

过关标准：

- 有一个候选配置。
- 有 seed 稳定性结果。
- 有失败案例解释。

## 第 6 周：paper trading 准备

目标：从回测脚本过渡到每日信号流程。

### Day 36：每日信号格式

设计一个每日输出：

```text
date:
model:
target_weights:
orders:
risk_flags:
cash_weight:
```

产出：

- 写出信号文件格式。

### Day 37：手动生成明日持仓

实战：

- 用最新模型对最近交易日生成 action。
- 不下单，只写目标仓位。

产出：

- `reports/paper_signals/YYYY-MM-DD.md`。

### Day 38：模拟成交

学习：

- open price、close price、next day execution 的差异。

实战：

- 假设第二天开盘成交。
- 手动记录模拟成交价和滑点。

产出：

- 第一条 paper trade 记录。

### Day 39：风控清单

必须包括：

- 单日最大亏损。
- 单周最大亏损。
- 最大仓位。
- 最大换手。
- 网络或数据失败时不交易。
- 模型输出异常时不交易。

产出：

- `docs/paper_trading_rules.md` 草案。

### Day 40-42：连续 3 天 paper 信号

每天做：

- 生成信号。
- 记录模拟成交。
- 更新组合净值。
- 和 SPY/现金对比。

过关标准：

- 流程能手动稳定执行，不靠临场发挥。

## 第 7-8 周：paper trading 正式期

目标：至少 20 个交易日 paper trading。

每天任务：

1. 更新数据。
2. 生成目标仓位。
3. 记录模拟订单。
4. 检查风控。
5. 写当天日志。

每周任务：

- 汇总收益、回撤、换手。
- 检查是否跑赢基准。
- 复盘最大亏损日。
- 不因为一周表现好就加钱。

20 个交易日过关标准：

- 没有数据或脚本事故。
- 最大回撤低于你愿意实盘承受的一半。
- 至少没有显著输给简单基准。
- 你能解释主要盈亏来源。

## 第 3 个月：小额实盘资格

只有满足这些条件才进入小额实盘：

- 至少 3-5 个 walk-forward 年份测试。
- 至少 20 个交易日 paper trading。
- 最差情形回撤你能接受。
- 策略不是靠一次极端行情赚钱。
- 有明确停机规则。
- API key 不允许提现。
- 不使用杠杆。

小额实盘设置：

- 初始资金：亏完不影响生活的钱，例如 500-2000 美元。
- 单日最大亏损：1%。
- 单周最大亏损：3%。
- 单月最大亏损：5%，触发后停止交易并复盘。
- 单资产最大权重：25%-35%。
- 每天只交易一次，避免盘中冲动操作。

每天实盘后记录：

```text
今日净值：
今日收益：
基准收益：
最大持仓：
成交偏差：
是否触发风控：
我是否手动干预：
明天是否继续：
```

## 什么时候才算“能挣钱”

第一阶段：能跑通。

- 你能稳定复现实验。
- 你知道结果文件在哪里。

第二阶段：能研究。

- 你能做 walk-forward。
- 你能解释失败年份。
- 你有固定模型选择规则。

第三阶段：能 paper trading。

- 连续 20-60 个交易日流程稳定。
- 没有因为数据、脚本、冲动手动改仓造成事故。

第四阶段：能小额实盘。

- 小资金连续 3 个月执行稳定。
- 最大回撤可控。
- 不因为赚钱就随意放大仓位。

第五阶段：才谈零花钱。

- 如果目标是每月 100 美元，年化 10% 大约需要 12,000 美元本金，年化 20% 也大约需要 6,000 美元本金，而且波动会很明显。
- 所以真正的目标是先把策略做稳，再考虑本金规模。

## 停止规则

出现这些情况，暂停交易或暂停优化：

- 回测只在一个年份赚钱。
- 稍微提高交易成本就失效。
- paper trading 连续出现执行错误。
- 最大回撤超过预设限制。
- 你开始想手动追涨杀跌。
- 策略解释不清楚但收益很好。

## 每周复盘问题

每周日回答：

- 本周最好的实验是什么？
- 最差的实验是什么？
- 哪个假设被证伪了？
- 哪个结果可能是过拟合？
- 下周只推进哪一个核心问题？

## 近期最推荐的 3 个项目改动

优先级从高到低：

1. 增加 walk-forward runner，自动跑多个训练/测试年份。
2. 增加 seed 参数和多 seed 汇总。
3. 增加换手率、年度收益、月度收益和成本敏感性报表。

把这 3 个补上之后，再去大幅改模型结构会更有意义。
