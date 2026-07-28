# NFL X-13：153 场完整报告

## 当前 State

- 比赛：153
- Factors：101
- Factor × Horizon：707
- 实际市场观察：4,773,299
- Reaction paths：610,932
- Factor observations：496,996
- Support PASS：509
- 机械候选：113
- 机械候选覆盖 Factors：52
- Holdout：CLOSED

当前所有结果仍是 historical source-time development association；不是因果、实时延迟、执行性或可交易 alpha。

## 这份报告究竟计算了什么

- 每个 factor 先在每场比赛内计算，再跨 153 场聚合。
- Horizon 是事件后 1/2/5/10/20/30/60 秒；没有实际成交时不 forward-fill。
- 数值是按 focal outcome 定向后的实际成交概率变化，单位为百分点（pp）。
- 当前表是未匹配 development 均值；matched-control 结果必须单独发布，不能混写。
- 机械候选要求样本支持、q≤0.05、LOO 同号率≥0.80、最大单场贡献≤25% 且双 venue 同方向。

## 当前可以读出的规律

- 正向最强的是 explosive play、first down、fourth-down conversion。
- 负向最强的是 red-zone empty drive、turnover on downs、missed field goal、red-zone turnover 和 lost fumble。
- 大变化主要集中在 20–60 秒。它可能表示渐进调整，也可能来自 source-time 不确定性、成交稀疏或样本构成，历史数据不能区分。
- Quarter / game-time 单独分桶的均值整体较小；OT 的部分大幅度单元支持不足。因此比赛相对时间必须与事件、比分、球权和赛前概率组合审核。
- 双 venue 虽常同方向，但机械候选中 Polymarket 的绝对变化中位数约为 Kalshi 的 7.4×；当前幅度不是双 venue 等强度复现。
- 以上都仍是待审核 pattern，不是 signal、因果或 alpha。

## 主要上升方向（未匹配）

| Factor | Horizon | Change | 95% CI | q | Games |
|---|---:|---:|---:|---:|---:|
| NFL.EVENT.EXPLOSIVE_PLAY | 60s | +2.54pp | [+2.06, +3.05]pp | 0.0011 | 130 |
| NFL.EVENT.EXPLOSIVE_PLAY | 30s | +1.79pp | [+1.58, +2.01]pp | 0.0011 | 151 |
| NFL.EVENT.FOURTH_DOWN_CONVERSION | 30s | +1.59pp | [+1.04, +2.21]pp | 0.0011 | 78 |
| NFL.EVENT.EXPLOSIVE_PLAY | 20s | +1.40pp | [+1.20, +1.62]pp | 0.0011 | 152 |
| NFL.EVENT.FIRST_DOWN | 30s | +1.32pp | [+1.17, +1.48]pp | 0.0011 | 153 |
| NFL.SCORE.PAT_MADE | 60s | +1.13pp | [+0.76, +1.55]pp | 0.0011 | 133 |
| NFL.EVENT.FOURTH_DOWN_CONVERSION | 20s | +1.06pp | [+0.58, +1.56]pp | 0.0011 | 72 |
| NFL.EVENT.FIRST_DOWN | 20s | +1.00pp | [+0.87, +1.14]pp | 0.0011 | 153 |
| NFL.SCORE.PAT_MADE | 30s | +0.96pp | [+0.59, +1.35]pp | 0.0011 | 132 |
| NFL.EVENT.SUCCESS | 60s | +0.90pp | [+0.70, +1.13]pp | 0.0011 | 153 |
| NFL.SCORE.PAT_MADE | 20s | +0.85pp | [+0.50, +1.23]pp | 0.0011 | 128 |
| NFL.EVENT.SUCCESS | 30s | +0.80pp | [+0.71, +0.90]pp | 0.0011 | 153 |

## 主要下降方向（未匹配）

| Factor | Horizon | Change | 95% CI | q | Games |
|---|---:|---:|---:|---:|---:|
| NFL.SEQUENCE.RED_ZONE_EMPTY_DRIVE | 60s | -7.42pp | [-9.13, -5.76]pp | 0.0011 | 51 |
| NFL.TURNOVER.ON_DOWNS | 60s | -6.54pp | [-8.60, -4.79]pp | 0.0011 | 70 |
| NFL.SCORE.FG_MISSED | 30s | -6.47pp | [-8.53, -4.69]pp | 0.0011 | 33 |
| NFL.COMBO.RED_ZONE_TURNOVER | 60s | -5.38pp | [-7.56, -3.45]pp | 0.0011 | 50 |
| NFL.SEQUENCE.BACK_TO_BACK_TURNOVERS | 60s | -5.23pp | [-7.47, -3.31]pp | 0.0011 | 36 |
| NFL.TURNOVER.FUMBLE_LOST | 60s | -4.89pp | [-7.27, -2.86]pp | 0.0011 | 43 |
| NFL.SCORE.FG_MISSED | 20s | -4.58pp | [-7.05, -2.77]pp | 0.0011 | 32 |
| NFL.SEQUENCE.RED_ZONE_EMPTY_DRIVE | 30s | -4.45pp | [-5.80, -3.08]pp | 0.0011 | 48 |
| NFL.SEQUENCE.BACK_TO_BACK_TURNOVERS | 30s | -3.90pp | [-5.47, -2.57]pp | 0.0011 | 34 |
| NFL.EVENT.FUMBLE_RECOVERY | 60s | -3.90pp | [-5.56, -2.43]pp | 0.0011 | 65 |
| NFL.TURNOVER.INTERCEPTION | 30s | -3.67pp | [-4.86, -2.60]pp | 0.0011 | 70 |
| NFL.SEQUENCE.RED_ZONE_EMPTY_DRIVE | 20s | -3.08pp | [-4.92, -1.46]pp | 0.0011 | 41 |

## 状态名词

- PASS：只表示样本支持门通过，不代表规律成立。
- INSUFFICIENT_SUPPORT：有效 episode/game 不足。
- DATA_GAP：正式所需字段或数据层缺失。
- N/R：未报告或不适用，不等于 0。

## 当前执行状态

- Exact-153：已发布并核验 hash；本报告与热力图直接读取该 immutable object。
- Matched controls：代码与目标测试已完成 reviewer 修复；完整 153 场 publication 仍须独立生成，不能用当前未匹配均值替代。
- Reference value：可以生成 diagnostic `reference_gap`；当前 `vegas_wp` 对应的 with-spread 模型 bytes 尚未独立验证，正式 gate 关闭。
- X-14：用于未来比赛的真实 event-receive、book-ready、bid/ask、t50/t90；接口与目标测试已完成，但尚无未来比赛真实 session。历史 X-13 不能回答这些真实延迟问题。

## Holdout Gate

只有体育专家审核并冻结 shortlist hash 后，才允许一次性读取 81 场 Holdout。
