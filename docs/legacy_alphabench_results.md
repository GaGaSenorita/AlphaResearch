# CoE (CoT) baseline — 三次独立重复的结果

方法：official AlphaBench `searcher.algo.cot.CoTAlgo`（代码里叫 CoT，论文里叫 CoE，同一个算法）。

## 协议

| | |
|---|---|
| Universe | CSI300 |
| Train (search) | 2010-01-01 → 2018-12-31 |
| Validation (selection) | 2019-01-01 → 2020-12-31 |
| Test (report once) | 2021-01-01 → 2024-12-31 |
| Label | `close_return`, next-day, `forward_n=1` |
| Seeds | Alpha158, `exclude_var=vwap` → 42 |
| Chains | 30（一个 seed 一条），10 轮 |
| LLM | Kimi-K2.6, temperature 0.75, `enable_reason=true` |
| Selection | validation `rank_ic` top-k，相关性过滤 0.7，budget 30 |
| Pool scoring | 每日截面 z-score，等权平均 |

## 三次重复的来源

| run | 目录 | config | 跑的时间 |
|---|---|---|---|
| rep1 | `runs/alphabench_cot/static_csi300_final` | 由 `finalize.py` 从 `static_csi300_v2` 的 297 个候选档案重选 | 2026-08-06 03:37（搜索） |
| rep2 | `runs/alphabench_cot/rep2` | `configs/alphabench_cot/static_baseline.yaml` | 2026-08-06 12:21 |
| rep3 | `runs/alphabench_cot/rep3` | `configs/alphabench_cot/static_baseline.yaml` | 2026-08-06 14:28 |

**关于 rep1 的等价性**：`static_v2.yaml` 与 `static_baseline.yaml` 只差 4 行——`name`、
`objective`（ic → rank_ic）、`factor-budget`（50 → 30）、`out-dir`。搜索阶段参数
（workers=30、temperature=0.75、search-rounds=10、max-correlation=0.7、协议、seed 池）
完全一致，且 `CoTAlgo._is_better` 硬编码 IC、搜索不受 `objective` 影响，因此从 v2 的档案
按 baseline 的选择参数重选，等价于端到端跑一遍 `static_baseline.yaml`。

**依赖**：`static_csi300_final/` 里只有 `summary.json`（68K），其档案在
`static_csi300_v2/`（73M）。删掉 v2，rep1 就没有原始数据支撑。

## Test pool 结果

| run | IC | RankIC | ICIR | RankICIR | turnover | daily_count |
|---|---|---|---|---|---|---|
| rep1 | 0.00279 | 0.01769 | 0.01565 | 0.09266 | 0.27304 | 968 |
| rep2 | 0.00360 | 0.02035 | 0.02114 | 0.11243 | 0.30220 | 968 |
| rep3 | 0.00464 | 0.02048 | 0.02590 | 0.10664 | 0.26621 | 968 |
| **mean** | **0.00368** | **0.01951** | **0.02090** | **0.10391** | **0.28048** | 968 |
| **std** | **0.00092** | **0.00157** | **0.00513** | **0.01016** | **0.01911** | 0 |

**Test RankIC = 0.01951 ± 0.00157**（n=3，样本标准差）

rep1 的 validation pool（另两次的 validation 在 `validation_selection.json`，不在 summary 里）：
IC 0.00823 / RankIC 0.04330 / ICIR 0.04989 / RankICIR 0.26474 / turnover 0.27286

## 怎么读这个数

BASELINE.md 里的说明照抄：可比的已发表数字用的是 10 日或 30 日 forward return、100 因子池、
LightGBM/MLP 组合。本 baseline 是次日、30 因子、等权 rank。三个因素都把数字压低，所以
0.0177 的 test RankIC 不能直接和 MCTS 论文里 CSI300 的 ~0.04 对比。

## 复现

```bash
# rep2 / rep3 的方式（端到端）
python -u scripts/run_experiment.py configs/alphabench_cot/static_baseline.yaml \
    --set "args.out-dir=runs/alphabench_cot/repN"

# rep1 的方式（从已完成的搜索重选）
python finalize.py    # 从 static_csi300_v2 的档案按 rank_ic/budget=30 重选
```

需要先在另一个终端起官方评测后端：`ppo start backend`。

详细的设计决策与三个反直觉发现见 `BASELINE.md`。
