# NFL X-13：153 场 Factor × 胜率 × 市场概率研究报告

状态：`PRELIMINARY_SOURCE_TIME_ONLY`  
Development：153 场  
Final holdout：81 场，`SEALED_UNREAD`  
结论日期：2026-07-27

## 结论

153 场历史数据支持一个清楚的描述性结论：

> Polymarket 的实际成交价格在多个高信息量 NFL 事件后呈现 10–60 秒逐步靠近 retrospective nflfastR 胜率变化的路径；最清楚的是 explosive play、first down、sack、negative play 和 turnover-on-downs。

这不是实时 latency 或可交易 alpha 证据。历史 source time 精度、不同 horizon 的实际成交样本变化、Kalshi 稀疏成交和 matched-control balance 失败，使本轮不能冻结正式 shortlist，也不能读取 81 场 holdout。

## 已完成的数据与计算

- 153/153 场 development bundle 已验证。
- 4,773,299 条实际市场观察。
- 610,932 条 reaction paths。
- 496,996 条 factor observations。
- 351,855 条严格有效 empirical distribution observations。
- 2,223 条 factor × venue × horizon 分布摘要。
- 21,819 条 episode-level nflfastR reference observations。
- 20,007 条非 OT、可支持的 retrospective reference observations。
- 351,855 条 reference-path joins；209,992 条满足 `|reference_delta| ≥ 1pp`。
- 8,083,596 对 exact matched controls。
- 23,211 个 covariate-balance groups；81 个通过 `SMD ≤ 0.10`，全部来自 Polymarket。
- 49 条通过 balance 后的 cross-game matched results。
- 10,000 次 game-cluster bootstrap、LOO、最大单场贡献与 factor-family BH correction。

所有历史研究对象来自已冻结 development artifacts；没有读取 81 场 holdout reaction，没有网络抓取，也不需要 API key。

## 主要经验结果

下表为 Polymarket moneyline 的实际成交变化，方向已统一到事件受益球队。数值是不同 horizon 的实际可观察样本，不做 forward-fill。

| Factor | 20s mean / 方向概率 | 30s mean / 方向概率 | 60s mean / 方向概率 | nflfastR completion median 20/30/60s |
|---|---:|---:|---:|---:|
| Explosive play | +1.95pp / P(up)=76.7% | +2.52pp / 74.9% | +2.85pp / 82.1% | 0.31 / 0.50 / 0.61 |
| First down | +1.51pp / P(up)=70.6% | +1.76pp / 73.4% | +2.20pp / 77.8% | 0.31 / 0.45 / 0.55 |
| Sack | −2.18pp / P(down)=81.0% | −3.04pp / 86.8% | −3.73pp / 73.6% | 0.44 / 0.55 / 0.67 |
| Negative play | −1.33pp / P(down)=62.8% | −1.83pp / 68.2% | −2.71pp / 65.3% | 0.31 / 0.45 / 0.56 |
| 3rd/4th-down failure | −2.29pp / P(down)=68.8% | −3.12pp / 77.5% | −4.00pp / 76.3% | 0.36 / 0.60 / 0.78 |
| Interception | −2.24pp / P(down)=62.1% | −6.64pp / 91.3% | −7.00pp / 88.2% | 0.18 / 0.62 / 0.92 |
| Lost fumble | −3.30pp / P(down)=69.2% | −7.05pp / 90.9% | −5.84pp / 77.8% | 0.14 / 0.49 / 0.65 |
| Turnover on downs | −6.64pp / P(down)=83.3% | −7.74pp / 88.2% | −6.29pp / 90.9% | 0.39 / 0.93 / 1.22 |

解释：

- Explosive play、first down、sack、negative play 的样本覆盖相对广，方向和 game-cluster CI 较稳定。
- Interception、lost fumble、turnover on downs 的效应更大，但样本数低，属于高价值扩样候选，不能据此估计稳定联盟效应。
- `3rd/4th-down failure` 必须拆开重新定义；当前数值只说明混合定义具有明显方向，不能进入 shortlist。
- Passing TD 的不同 horizon 样本组成不一致，20–60 秒没有稳定单调路径；不能把早期 +0.82pp 当作完整 TD repricing。

## 时间路径

`completion_fraction = market_delta / reference_delta` 只在 `|reference_delta| ≥ 1pp` 时计算。

- Explosive play：10s 约 0.05，20s 0.31，30s 0.50，60s 0.61。
- First down：10s 约 0，20s 0.31，30s 0.45，60s 0.55。
- Sack：10s 0，20s 0.44，30s 0.55，60s 0.67。
- Negative play：10s 0，20s 0.31，30s 0.45，60s 0.56。
- 3rd/4th-down failure：10s 0，20s 0.36，30s 0.60，60s 0.78。
- Interception：10s 约 0，20s 0.18，30s 0.62，60s 0.92，但样本较少。

这是 `progressive source-time adjustment candidate`，不是同一笔盘口的精确连续反应曲线。原因是每个 horizon 必须存在实际成交，因此各 horizon cohort 不完全相同。

10→60 秒的完整 paired paths 很少。例如 explosive play 的 Polymarket paired path 只有 22 条，first down 92 条，sack 9 条。因此 continuation/reversal/overshoot 只作 case-level 描述，不作为 shortlist 门。

## 比赛状态分层

状态分层显示反应幅度与 event 本身同样重要：

- Explosive play 在 30 秒时，一比分比赛平均 +3.09pp，多比分比赛 +1.35pp。
- Explosive play 在事前概率 20–80% 区间反应更大；极端 `<20%` 或 `≥80%` 更小。
- Sack 在 60 秒时，一比分比赛平均 −5.93pp，多比分比赛仅 −0.38pp。
- Negative play 在 60 秒时，一比分比赛平均 −4.00pp，多比分比赛 −0.59pp。

这些是未完全控制的 descriptive breakdown。OT 样本很少，且 nflfastR OT 支持未证明，正式 reference comparison 已排除 OT。

## Polymarket 与 Kalshi

历史数据不能证明 venue-specific alpha：

- Polymarket 对主要 factor 的有效成交观察通常是 Kalshi 的 5–10 倍。
- Kalshi 价格变化大量为 0，且每个 event/horizon 通常只有约一笔实际成交。
- 两边 staleness 均约 2 秒左右，单看 staleness 不能解释全部差异。
- Same-episode pair 极少：主要 factor × horizon 通常只有 1–3 对。
- Exact matcher 产生 8,083,596 对 controls，但 Kalshi 的 balance group 没有一个达到 `SMD ≤ 0.10`。

因此当前结论是：

> activity/coverage 和样本组成是首要解释候选；无法排除 venue-specific reaction，但现有历史成交不足以识别它。

不能把 Polymarket 与 Kalshi 的 raw 均值差解释为 hedge、套利、速度或执行机会。

## Matched-control 结果

匹配条件包括 venue、market family、outcome orientation、horizon，并控制：

- pre-event probability；
- game seconds remaining；
- score margin；
- staleness；
- liquidity。

平衡结果非常严格：

- 23,211 个 balance groups 中只有 81 个通过。
- 49 条 cross-game results 全部是 Polymarket。
- `NFL.EVENT.SUCCESS`、`NFL.STATE.DISTANCE_MEDIUM`、`NFL.VALUE.YAC_SURPRISE` 的少数 horizon 达到统计 support，但前两者分别是过宽事件定义和状态控制变量，不是已接受的体育 factor。
- 体育审核接受的 8 个原子 factor 中，没有一个同时通过双 venue、support、balance、q、LOO 和单场贡献门。

所以不冻结 shortlist，81 场 holdout 继续封存。

## 体育语义审核

已接受、值得未来扩样的原子 factor：

- Sack。
- Passing TD。
- Rushing TD。
- Return TD。
- Safety。
- Interception。
- Lost fumble。
- Turnover on downs。

必须重定义后才能研究：

- 3rd-down failure 与 4th-down failure 分开。
- Fumble recovery 区分 lost/not-lost 与 recovery team。
- FG made/missed/blocked 分开并带距离/state。
- Punt return 使用 field-position value，不用“positive return”二元标签。
- Red-zone empty drive 明确 drive terminal state。

拒绝作为独立 alpha factor：

- Generic pass/rush/success/punt。
- Distance/quarter/red-zone 等单独状态；它们只能作为 moderator/control。

## Shortlist 与 holdout 决策

当前 shortlist：`NOT_FROZEN_NO_FACTOR_PASSES_ALL_GATES`。

81 场 final holdout 不读取。下一次可以冻结 shortlist 的前提是：

1. 扩大 Kalshi 同 episode 的真实成交覆盖，或明确将研究问题改为 Polymarket-only。
2. 对 accepted factor 获得足够 balance-pass games。
3. 固定 factor、state bucket、horizon、reference treatment 和 exclusion。
4. 只在锁定 hash 之后读取一次 holdout。

## Claim boundary

本报告允许：

- 历史实际成交的经验分布。
- Source-time interval correlation。
- Retrospective nflfastR reference gap。
- Matched-control 描述与失败原因。

本报告不允许：

- 真实 event-feed latency。
- Bid/ask、L2、OFI、depth 或可执行价。
- 因果结论。
- Hedge、套利或可交易 alpha。
- 使用 holdout 调参。

