# 第 4.6 节：代表性因子的分组净值

已有 `external/AlphaBench/backtest/factor_metrics/metrics.py::FL_QuantileReturn`
只返回分位数组的平均收益，未保存每日分组净值，也不绘制参考图中的双轴曲线。
新增实现位于 `src/alpha_research/reporting/group_nav.py`，命令入口为
`scripts/plot_representative_factor_nav.py`。

## 本地复现

在 AlphaResearch 仓库中运行：

```bash
conda activate AlphaResearch
python scripts/plot_representative_factor_nav.py
```

默认从 `figures/representative_factor_nav/` 的每日 CSV 和 `metadata.json`
重画四张图，不加载 Qlib 或原始市场数据，也不调用 LLM 或 FFO。
先核对每日行数、分期 RankIC 和终值与原 metadata 一致，再输出图像；
原 CSV 和 metadata 保持不变。可用 `--source-dir PATH` 指定保存的数据，
用 `--output-dir PATH` 指定图像输出位置。

需要从原始市场数据重新计算因子时，显式运行：

```bash
python scripts/plot_representative_factor_nav.py --recompute --provider /path/to/cn_data
```

此模式默认使用仓库同级 `quantaalpha_qlib_csi300/cn_data`；
可以使用 `--workers 4` 和 `--config PATH`。
默认配置 `configs/representative_factor_nav.json` 保存两个公式、出处和论文分期。
重新计算会写入每日数据、净值 CSV、summary、metadata 和图像，仍不调用 LLM 或 FFO。
每张图默认只保存 PDF；如需用于演示，可显式指定
`--format png` 或 `--format svg`，并通过 `--output-dir` 将额外导出放在仓库外。

## 图像和数据

原始数据重新计算模式将下列图像和数据写入 `figures/representative_factor_nav/`；
默认保存数据重画模式只更新其中四张 PDF：

- 每个因子各一张 `*_full_nav.pdf`：2016—2025 连续全区间，虚线标出 Validation 和 Test 起点。
- 每个因子各一张 `*_test_nav.pdf`：2022—2025 测试期，重新以 1 为初始净值。
- `*_daily.csv`：信号日期、实际收益实现日期、每日各组收益、每日 RankIC、组内股票数和缺失/并列记录数。
- `*_nav.csv`：与图完全一致的逐日净值，包含初始值 1。
- `summary.csv`：训练、验证、测试分期的 RankIC 和平均多空日收益。
- `metadata.json`：公式、运行口径、数据位置、日历/成分文件摘要、跳过日期、核对指标和终值。

## 计算口径

使用本地 Qlib 的 CSI 300 历史成分区间，直接计算原始公式，不做截面中性化、
缩尾、缺失值填充或因子符号翻转。Qlib 自动读取滚动窗口所需的历史数据。
收益标签与论文一致：`Ref($close, -1)/$close - 1`。

每天只保留因子及标签均有限的股票，因子值由低到高分为十组，G1 最低、G10 最高。
并列因子值按股票代码排序，同一并列值可能跨组；这种处理不使用收益来打破并列。
各组股票数最多相差 1，组内等权。有效样本不足十只或全截面因子相同时跳过该日，
并记录原因。缺失标签的剔除遵循论文因子评估口径，因此属于事后诊断。

组收益为组内标签均值；组净值为 `cumprod(1 + group_return)`。
多空收益为 `G10_return - G1_return`，净值为 `cumprod(1 + long_short_return)`。
多空采用相对净资本 100% 多头、100% 空头，即总敞口 200%；它不是两条净值相减，
也不是 G10/G1 的比值。每天再平衡，费用设为零，没有模拟滑点、停牌成交限制、
涨跌停成交限制、借券成本、融资和容量。

图的横轴使用收益实现日 t+1，初始值放在第一天的信号日 t。
最后信号日为 2025-12-26，其对应收益在下一交易日实现，因此曲线会延伸至下一交易日。
全区间图连续包含论文 Train/Validation/Test 间的日历空档；分期 RankIC 严格按论文
各区间独立计算。脚本会核对保存的六个 RankIC，偏差超过 0.000051 时明确报错。

## 论文使用建议

这两例为事后选择的案例。图用于说明因子分组的收益差异，不能作为事前选因子的
样本外策略业绩。因子用到 t 日收盘信息，同时采用 t 到 t+1 收盘收益，未模拟
信号形成后的可执行成交价格，因此建议称为 **factor-sorted cumulative return diagnostics**，
避免称为可交易策略回测或成本后收益。

全区间图用于展示长期分组形态；正文讨论测试期时，可用 `*_test_nav.pdf`。
不要把“fee rate = 0”解释为已有真实费用模型。当前模块刻意只实现零费用诊断。

VOL_ADJ_UPPER_SHADOW 对应 seed 123 的第 38 轮。
INTRADAY_DRAWDOWN_VOLUME 对应早期 seed 42 的第 34 轮，与正文三次 100 轮实验中的
seed 42 区别保留在配置中。

验证计算逻辑：

```bash
python -m pytest tests/test_group_nav.py -q
```
