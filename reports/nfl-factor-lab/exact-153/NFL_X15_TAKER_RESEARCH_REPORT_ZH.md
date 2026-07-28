# NFL X-15：153 场多观察点 Taker 信号研究

状态：`PRELIMINARY_NON_EXECUTABLE`

数据分区：153 场 development

Final holdout：81 场，`SEALED_UNREAD`

最终审核结论：`NO_QUALIFIED_DEVELOPMENT_SHORTLIST`

发布日期：2026-07-28

## 1. 研究问题

X-15 研究的不是“事件后价格均值”，而是：

> 在事件发生后 1、2、3、5、10 秒分别观察时，结合事件、比赛状态、nflfastR reference gap 和当时实际成交价，能否预测之后 5–60 秒尚未完成的价格变化？

时间网格：

```text
landmark = 1 / 2 / 3 / 5 / 10 秒
endpoint = 5 / 10 / 15 / ... / 60 秒
```

每行数据的粒度：

```text
episode × venue × logical market × outcome × landmark × endpoint
```

目标：

```text
target_delta
= endpoint actual trade mark
- landmark actual trade mark
```

actual trade mark 是历史成交，不是当时可保证成交的 ask/bid。因此本研究只能发现历史 signal candidate，不能作为真实 taker fill backtest。

## 2. 关键正确性修复

旧 publication `sha256:069b…a32a4` 与后续中间版本已保留作审计，但被最终版本取代。最终实现修复了四个会导致误判的问题：

1. **未来可观测性泄漏**：信号选择只使用 landmark 当时已知的 `decision_eligible`；不能因为未来 endpoint 恰好有成交才允许产生信号。
2. **Censoring**：当时可产生、但未来 endpoint 无法评估的信号保留为 `CENSORED_EVALUATION`，不进入 hit rate、markout、分布或统计门。
3. **统计门**：Polymarket 与 Kalshi 必须分别通过 support、cluster CI、BH q-value、LOO 和单场贡献门，再检查同方向；不能只因方向相同就通过。
4. **概率与 shadow curve**：只有 multinomial logistic 报告真正的 probabilistic log loss；回归模型不伪装成概率模型。Shadow curve 按 exit time 记账，并纳入初始 NAV peak。

## 3. 数据漏斗

| 项目 | 数量 |
|---|---:|
| Development games | 153 |
| 全部 landmark-grid rows | 1,525,434 |
| Decision-eligible landmark rows | 529,362 |
| Target-eligible landmark rows | 304,846 |
| Decision episodes | 12,197 |
| Target-eligible episodes | 11,940 |
| Walk-forward OOF predictions | 1,750,516 |
| Policy-selected signals | 8,374 |
| 可评估 signals | 5,206 |
| Censored signals | 3,168 |
| Holdout reaction reads | 0 |

`decision_eligible` 只依赖当时已知信息；`target_eligible` 还要求未来 endpoint 有实际成交且没有污染、顺序歧义或方向错误。

## 4. Walk-forward 模型结果

五个 chronological folds：

```text
train W1–2  → validate W3–4
train W1–4  → validate W5–6
train W1–6  → validate W7–8
train W1–8  → validate W9–10
train W1–10 → validate W11–12
```

Polymarket 与 Kalshi 分开训练。效果指标仅使用 `EVALUATED` signals。

| Venue | Model | Macro-F1 | Log loss | MAE | Total / evaluated / censored | Hit rate | Mean markout |
|---|---|---:|---:|---:|---:|---:|---:|
| Kalshi | Huber | 0.537 | N/A | 1.58pp | 207 / 151 / 56 | 43.7% | 1.60pp |
| Kalshi | Logistic | 0.606 | 0.142 | 1.59pp | 343 / 239 / 104 | 89.1% | 7.84pp |
| Kalshi | XGBoost | 0.522 | N/A | 1.51pp | 171 / 132 / 39 | 81.8% | 7.56pp |
| Kalshi | Reference baseline | 0.265 | N/A | 10.06pp | 5,693 / 4,135 / 1,558 | 24.4% | 0.12pp |
| Polymarket | Huber | 0.467 | N/A | 1.93pp | 64 / 15 / 49 | 26.7% | 0.23pp |
| Polymarket | Logistic | 0.570 | 0.226 | 1.81pp | 136 / 21 / 115 | 95.2% | 9.66pp |
| Polymarket | XGBoost | 0.524 | N/A | 1.82pp | 77 / 19 / 58 | 84.2% | 9.50pp |
| Polymarket | Reference baseline | 0.296 | N/A | 10.20pp | 1,683 / 494 / 1,189 | 21.1% | 0.13pp |

这些高 hit rate 不能单独解释为 alpha。尤其 Polymarket learned-model 的可评估信号只有 15–21 条，大量信号被 censor；这是严重的 coverage/selection uncertainty。

## 5. Statistical gate

正式门：

1. 至少 30 场、20 个可评估信号；
2. 10,000 次 game-cluster bootstrap 的 mean CI 排除 0；
3. factor-family 内 BH `q ≤ 0.05`；
4. leave-one-game-out 同号率 `≥ 0.80`；
5. 最大单场绝对贡献 `≤ 25%`；
6. Polymarket 与 Kalshi 各自满足上述条件且同方向。

最终结果：

| Gate | 通过数量 |
|---|---:|
| 单 venue individual statistical gate | 8 |
| 双 venue complete development gate | 0 |

8 个 individual gates 全部来自 Kalshi：

- Interception：4 个模型；
- Lost fumble：1 个模型；
- Ordinary pass：3 个模型。

Polymarket 没有任何 factor×model 独立通过完整统计门，因此没有 cross-venue pass。

## 6. 体育语义审核

| Factor | 审核 | 结论 |
|---|---|---|
| Interception | `REDEFINE` | 当前桶仍需明确分离 pick-six 与四档抄截；虽有 4 个 Kalshi individual gates，但 Polymarket 仅覆盖 1–3 场。 |
| Lost fumble | `REDEFINE` | 需要拆分 scrimmage/sack fumble、punt/muff 和 scoring return；仅 Kalshi logistic 通过 individual gate。 |
| Turnover on downs | `ACCEPT` 体育语义 / `DATA_GAP` 统计 | 定义清楚，但 Kalshi 最多 28 场，低于 30 场门；Polymarket 仅 2–6 场。 |
| Missed field goal | `REDEFINE / DATA_GAP` | 需锁定 beneficiary、clock/end-half、blocked/return 和非正常档位场景；覆盖不足。 |
| Ordinary pass | `REJECT` | 这是含成功/失败、sack、turnover、penalty 的异质残余桶；统计显著不能修复定义无效。 |

所以当前最重要的发现不是“已经找到一个可交易 factor”，而是：

> Turnover/value-shock family 值得继续，但在体育定义和跨 venue 覆盖同时过门之前，不能进入正式 holdout。

## 7. Polymarket 与 Kalshi 的差异

Kalshi 在这批历史数据中有更多可评估 signals；Polymarket censor rate 显著更高。当前只能说明：

- 两个 venue 的实际成交 availability 和 sampling 不同；
- Polymarket learned-model 估计由极少数可评估 signals 支撑；
- trades-only 数据不能判断谁反应更快；
- 没有历史 BBO/L2、相同主机 local receive timestamp、spread、fees 和可用 size，不能解释为执行优势或跨 venue hedge。

完整统计门要求两个 venue 各自通过，正是为了避免把 coverage 差异误判成市场机制。

## 8. Shadow curve 的边界

当前：

```text
gross_markout
= predicted direction
× historical actual-trade price change
```

固定 1 contract，以 exit time 累加到初始 NAV 1,000。它只能诊断 signal 集中度、连续不利段和 policy overlap；不能代表真实 P&L，因为缺少 entry ask、exit bid、spread、fees、slippage、size 和 fill。

因此 terminal NAV 与 drawdown 不作为 shortlist 依据，也不计算正式 Sharpe、Sortino、Calmar 或 executable MDD。

## 9. 最终结论和下一步

本轮正式冻结：

```text
NO_QUALIFIED_DEVELOPMENT_SHORTLIST
```

81 场 holdout 继续封存，原因不是系统没跑完，而是没有任何体育语义有效的 factor 同时通过两个 venue 的 development gate。

下一步最短路径：

1. 将 interception、lost fumble、missed FG 按上述体育语义重新注册为新版本；
2. 不再使用 ordinary pass；
3. 扩充 Polymarket 可评估覆盖，或在预注册规则中明确单 venue 研究，但不能事后降低当前双 venue gate；
4. 新定义重新跑 development，只有产生合格 shortlist 后才一次性读取 81 场 holdout；
5. 真正 taker backtest 等同场 NFL L2/quotes 可用后再做。

## 10. Artifact 与入口

- 最终 X-15 manifest：`sha256:bbbdbfece9c1092cfa2e0c967d3a56766d3f0a920651950d4f0c30851e08b2ab`
- 离线工作台：`reports/nfl-factor-lab/exact-153/x15-taker-workbench-v2/sha256/f6/f6a7ed7d2fe1005302efc507d30da30d2e007a2b184360dac13bdfde8be179ff.html`
- Master Notebook：`notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb`
- Local Jupyter：`http://127.0.0.1:8890/lab/tree/notebooks/nfl-factor-lab/NFL_Factor_Lab_Master.ipynb`

Claim boundary：

```text
DEVELOPMENT_ONLY_HISTORICAL_ACTUAL_TRADES
no executable quote, fill, live latency, causality, tradeability, or holdout claim
```
