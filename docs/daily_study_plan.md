# 8 周量化学习与实战计划

这个计划假设你已经会 Python，重点放在金融概念、量化研究流程、数据练习和训练自己的模型。节奏是每天 60-120 分钟。每一天都要留下一个小产出：一段实验代码、一张图、一段实验记录，或者一个指标对比。

建议主线参考：

- Georgia Tech Machine Learning for Trading
- Coursera Machine Learning for Trading Specialization
- BigQuant 量化入门资料
- Quantra factor investing / trading strategy 相关课程

## 每日固定流程

每天按这个顺序做：

1. 概念：理解当天主题，不追求背公式，先知道它解决什么问题。
2. 数据：找一组价格、收益率、因子或标签数据。
3. 实验：在项目里写一个脚本或 notebook 验证概念。
4. 记录：写下假设、指标、观察和下一步。

实验记录模板放在 `docs/learning_path.md` 里。你也可以每天新建一条 markdown 记录，例如 `docs/journal/2026-04-27.md`。

## 第 1 周：市场、收益率和风险直觉

目标：知道量化里最基础的对象是什么：价格、收益率、波动率、回撤、交易成本。

### Day 1：市场数据长什么样

学习：

- OHLCV：open, high, low, close, volume
- 日线、分钟线、复权、停牌、缺失数据
- 为什么量化研究一般不直接预测价格，而是预测收益率或方向

练习：

- 跑通 `python -m quant_for_fun.examples.train_baseline`
- 阅读 `quant_for_fun/data/synthetic.py`
- 观察生成出来的数据字段

产出：

- 写一段记录：价格数据里哪些字段能做特征，哪些不能乱用。

### Day 2：收益率

学习：

- 简单收益率：`P_t / P_{t-1} - 1`
- 对数收益率：`log(P_t / P_{t-1})`
- 累计收益率

练习：

- 在 `features/technical.py` 里对比 1 日、5 日、20 日收益率
- 画出累计收益曲线，或者打印前后若干行检查

产出：

- 解释为什么模型通常预测 future return，而不是直接预测未来价格。

### Day 3：波动率

学习：

- rolling volatility
- 年化波动率
- 波动率和风险的关系

练习：

- 修改 `volatility_20d` 为 10 日、60 日版本
- 比较不同窗口下信号变化

产出：

- 一张或一段表格：不同窗口 volatility 的差异。

### Day 4：最大回撤

学习：

- equity curve
- drawdown
- max drawdown
- 为什么只看收益率会误导

练习：

- 阅读 `quant_for_fun/backtest/vectorized.py`
- 手动构造一组收益率，验证最大回撤计算

产出：

- 用自己的话解释最大回撤和 Sharpe 的区别。

### Day 5：交易成本和换手

学习：

- 手续费、滑点、冲击成本
- turnover
- 高频换仓为什么容易被成本吃掉

练习：

- 把 `trading_cost_bps` 从 0、5、20、50 分别跑一遍
- 对比 total return 和 Sharpe

产出：

- 一张成本敏感性表。

### Day 6：训练集和测试集

学习：

- 为什么时间序列不能随机切分
- in-sample vs out-of-sample
- 数据泄漏

练习：

- 阅读 `quant_for_fun/models/baseline.py`
- 修改 `train_fraction`，观察测试表现变化

产出：

- 列出 3 种容易造成金融模型数据泄漏的情况。

### Day 7：周复盘

学习：

- 回顾收益、风险、成本、验证方式

练习：

- 重新跑 baseline
- 整理一份第 1 周实验总结

产出：

- 写出你目前理解的“一个量化策略闭环”：数据 -> 特征 -> 标签 -> 模型 -> 信号 -> 回测 -> 评估。

## 第 2 周：经典策略直觉

目标：理解最常见的策略思想：动量、均值回归、趋势跟随。

### Day 8：动量

学习：

- 过去涨的资产未来继续涨
- time-series momentum vs cross-sectional momentum

练习：

- 修改 `momentum_5d`、`momentum_20d` 的窗口
- 观察模型特征重要性

产出：

- 找出哪个 momentum 特征在当前模拟数据里更有效。

### Day 9：均值回归

学习：

- 价格偏离均值后回归
- z-score
- 均值回归适合什么市场

练习：

- 新增 `zscore_20d = (close - ma20) / std20`
- 用 z-score 设计一个简单 long/flat 规则

产出：

- 对比均值回归规则和 baseline ML 策略。

### Day 10：均线策略

学习：

- short moving average
- long moving average
- golden cross / death cross 的局限

练习：

- 实现 10 日均线上穿 50 日均线策略
- 加入交易成本

产出：

- 一份均线策略指标摘要。

### Day 11：突破策略

学习：

- Donchian channel
- 价格突破 N 日高点
- 趋势策略的胜率和盈亏比

练习：

- 构造 `breakout_20d`
- 测试突破后持有 5 日或 20 日

产出：

- 记录突破策略的胜率、平均收益和最大回撤。

### Day 12：止损与仓位

学习：

- stop-loss
- position sizing
- 固定仓位 vs 波动率调整仓位

练习：

- 给策略加一个简单止损规则
- 尝试把信号从 0/1 改成 0/0.5/1

产出：

- 比较满仓和半仓策略的波动率、回撤。

### Day 13：策略失效

学习：

- regime change
- overfitting
- 为什么单个策略不会一直有效

练习：

- 把 synthetic 数据分成前后两段，分别评估策略
- 改变漂移率和波动率重新生成数据

产出：

- 写下“策略在哪种市场环境下更容易失效”。

### Day 14：周复盘

练习：

- 整理动量、均值回归、均线、突破策略的优缺点

产出：

- 一张策略对比表：逻辑、适合环境、风险、可训练特征。

## 第 3 周：投资组合和风险

目标：从单资产策略过渡到多资产组合。

### Day 15：组合收益

学习：

- 权重
- portfolio return
- rebalancing

练习：

- 生成 3-5 条 synthetic price series
- 计算等权组合收益

产出：

- 等权组合和单资产表现对比。

### Day 16：相关性

学习：

- correlation
- diversification
- 相关性在危机时可能上升

练习：

- 计算多个资产收益率相关矩阵

产出：

- 找出最高相关和最低相关的一对资产。

### Day 17：Markowitz 组合

学习：

- expected return
- covariance matrix
- efficient frontier

练习：

- 手动枚举随机权重组合
- 找 Sharpe 较高的组合

产出：

- 一组候选权重和指标。

### Day 18：风险平价直觉

学习：

- 每个资产贡献相近风险
- volatility weighting

练习：

- 用 `1 / volatility` 生成权重
- 对比等权组合

产出：

- 记录哪个组合回撤更小。

### Day 19：VaR 和 Expected Shortfall

学习：

- Value at Risk
- Expected Shortfall
- 历史模拟法

练习：

- 用组合日收益率计算 95% VaR 和 ES

产出：

- 解释 VaR 的一个缺点。

### Day 20：组合再平衡

学习：

- monthly rebalance
- drift
- transaction cost

练习：

- 比较不再平衡、月度再平衡、周度再平衡

产出：

- 再平衡频率与成本的对比。

### Day 21：周复盘

产出：

- 写出你自己的组合研究 checklist。

## 第 4 周：因子模型

目标：理解“模型训练”的输入不只是技术指标，也可以是价值、质量、动量、波动等因子。

### Day 22：什么是因子

学习：

- alpha factor
- risk factor
- value, momentum, quality, size, volatility

练习：

- 把当前技术特征按因子类别分类

产出：

- 写一张因子字典：名称、含义、可能方向、风险。

### Day 23：因子 IC

学习：

- Information Coefficient
- Spearman rank correlation
- 因子和未来收益的关系

练习：

- 计算每个特征和 `forward_return` 的相关性

产出：

- 排名前 3 的因子和你的解释。

### Day 24：因子分层

学习：

- quantile
- top bucket / bottom bucket
- 单调性

练习：

- 把某个因子分成 5 组
- 比较每组未来收益

产出：

- 判断这个因子是否有单调性。

### Day 25：多因子合成

学习：

- rank
- z-score 标准化
- composite score

练习：

- 把 momentum、volatility、ma_gap 做成一个综合分数

产出：

- 对比单因子和多因子策略。

### Day 26：因子中性化直觉

学习：

- 行业、市值、风格暴露
- 为什么好因子可能只是押中了某个风险

练习：

- 先用 synthetic 数据理解概念
- 记录真实数据接入后需要哪些字段：行业、市值、估值等

产出：

- 写下未来真实数据表需要包含哪些字段。

### Day 27：因子衰减

学习：

- 因子预测力随 horizon 变化
- 1 日、5 日、20 日 forward return

练习：

- 修改 `horizon=1, 5, 20`
- 分别计算因子相关性

产出：

- 找出哪个 horizon 下特征更稳定。

### Day 28：周复盘

产出：

- 设计你的第一个多因子模型草案。

## 第 5 周：监督学习模型

目标：从策略规则过渡到可以训练和评估的模型。

### Day 29：标签设计

学习：

- binary classification
- regression target
- threshold label
- label horizon

练习：

- 修改 `add_forward_return_label`
- 尝试上涨/下跌二分类和收益率回归两种目标

产出：

- 说明你更想训练分类模型还是回归模型。

### Day 30：基准模型

学习：

- baseline model
- 为什么简单模型很重要

练习：

- 跑当前 RandomForest baseline
- 记录 accuracy、AUC、Sharpe、max drawdown

产出：

- baseline 实验记录。

### Day 31：模型指标

学习：

- accuracy
- precision / recall
- ROC AUC
- finance metric vs ML metric

练习：

- 找一次 AUC 提升但回测收益下降的情况

产出：

- 写下为什么 ML 指标不等于交易指标。

### Day 32：概率转仓位

学习：

- probability threshold
- confidence
- position sizing

练习：

- 比较 threshold=0.50, 0.55, 0.60, 0.65

产出：

- 阈值敏感性表。

### Day 33：特征重要性

学习：

- feature importance
- permutation importance
- 相关不等于可交易 alpha

练习：

- 输出 RandomForest feature importance

产出：

- 解释最重要的 3 个特征。

### Day 34：过拟合

学习：

- 模型复杂度
- max_depth
- min_samples_leaf

练习：

- 改 RandomForest 参数，对比 train/test AUC

产出：

- 一张过拟合观察表。

### Day 35：周复盘

产出：

- 写出你的第一版“模型训练协议”：数据、特征、标签、切分、指标、回测规则。

## 第 6 周：时间序列验证和 walk-forward

目标：让模型评估更接近真实交易。

### Day 36：为什么不能随机 K-fold

学习：

- temporal ordering
- look-ahead bias
- leakage

练习：

- 对比随机切分和时间切分的结果

产出：

- 记录随机切分为什么可能虚高。

### Day 37：walk-forward

学习：

- rolling train window
- expanding train window
- test window

练习：

- 设计一个 walk-forward 函数接口

产出：

- 写出伪代码。

### Day 38：滚动训练

学习：

- 每隔一段时间重新训练模型
- 模型漂移

练习：

- 实现最简单的 rolling retrain

产出：

- 多个测试窗口的指标表。

### Day 39：窗口长度

学习：

- 数据太少 vs 数据太旧
- market regime

练习：

- 比较 252、504、756 日训练窗口

产出：

- 找出当前数据下较稳定的窗口。

### Day 40：样本外稳定性

学习：

- mean performance
- distribution of performance
- worst period

练习：

- 汇总每个 walk-forward 窗口的收益和回撤

产出：

- 找出最差窗口并分析原因。

### Day 41：模型选择

学习：

- 不只选收益最高
- 稳定性、回撤、换手、解释性

练习：

- 对比 2-3 个模型或参数组

产出：

- 写出选择某个模型的理由。

### Day 42：周复盘

产出：

- 把 walk-forward 作为以后所有模型实验的默认验证方式。

## 第 7 周：真实数据接入与个人数据集

目标：开始用真实数据训练自己的模型。

### Day 43：确定研究市场

选择一个方向：

- 美股日线
- A 股日线
- ETF
- 加密货币
- 期货

产出：

- 写清楚你要研究的市场、标的范围、频率、数据来源。

### Day 44：数据 schema

学习：

- symbol
- timestamp
- OHLCV
- adjusted close

练习：

- 设计 `data/raw/` 和 `data/processed/` 文件格式

产出：

- 一份数据字段说明。

### Day 45：数据质量检查

学习：

- missing values
- duplicated rows
- abnormal returns
- split / dividend adjustment

练习：

- 写一个数据检查 checklist

产出：

- 数据质量报告模板。

### Day 46：真实数据特征

练习：

- 把真实数据跑进 `add_technical_features`
- 检查是否有大量空值或异常值

产出：

- 第一份真实数据特征表。

### Day 47：真实数据 baseline

练习：

- 用真实数据训练 baseline 模型
- 输出 ML 指标和回测指标

产出：

- 第一份真实数据 baseline 实验记录。

### Day 48：多标的训练

学习：

- panel data
- cross-sectional features
- symbol-level split

练习：

- 思考单标的模型和多标的模型的区别

产出：

- 写出多标的模型的数据结构草案。

### Day 49：周复盘

产出：

- 确定以后长期积累的数据源和更新频率。

## 第 8 周：形成自己的研究工作流

目标：把学习转成长期可复用的模型训练流程。

### Day 50：实验管理

学习：

- 每次实验必须记录参数、数据范围、结果

练习：

- 新建 `docs/journal/`
- 写第一篇正式实验记录

产出：

- 实验记录规范。

### Day 51：模型版本

学习：

- model version
- feature version
- dataset version

练习：

- 给 baseline 命名为 `baseline_rf_v1`

产出：

- 模型命名规则。

### Day 52：策略报告

学习：

- 策略逻辑
- 数据区间
- 样本外表现
- 风险指标
- 失败场景

练习：

- 写一份 baseline 策略报告

产出：

- `reports/baseline_rf_v1.md`

### Day 53：压力测试

学习：

- 更高成本
- 不同市场环境
- 参数扰动

练习：

- 对当前模型做成本和参数压力测试

产出：

- 压力测试表。

### Day 54：组合模型

学习：

- ensemble
- 多策略组合
- 信号平均

练习：

- 把规则策略和 ML 策略信号做简单合成

产出：

- 对比单模型和组合模型。

### Day 55：个人研究方向

选择一个长期方向：

- 多因子选股
- ETF 轮动
- 趋势跟随
- pairs trading
- 加密货币动量
- 波动率策略

产出：

- 写一页个人研究方向说明。

### Day 56：总复盘

产出：

- 总结 8 周收获
- 列出 3 个下一阶段实验
- 确定第一个长期训练模型

## 推荐执行方式

每周只维护一个主实验分支，例如：

```bash
git checkout -b week-01-risk-basics
```

每完成一个小实验就提交一次：

```bash
git add -A
git commit -m "Add volatility and drawdown experiments"
```

## 第一阶段最重要的判断标准

不要急着追高收益。前 8 周真正要练出来的是：

- 看懂收益、风险、回撤和成本
- 知道什么是数据泄漏
- 能设计一个标签
- 能把概率变成仓位
- 能做样本外验证
- 能判断模型是否只是过拟合
- 能把一次实验记录清楚

这些能力比任何单个模型都更重要。
