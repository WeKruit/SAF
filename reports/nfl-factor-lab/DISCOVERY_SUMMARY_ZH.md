# X-13 NFL Sports Factor Lab：20 场 Discovery 中文摘要

状态：`PRELIMINARY_SOURCE_TIME_ONLY`

本报告只描述历史 source-time 关联，不声称因果、真实 feed latency、可执行价格、交易性或 alpha。历史 holdout reaction 仍未读取。

## 已完成的数据与计算

- 冻结 discovery：20 场。
- 原始目标 PBP：3,699 行。
- 正式 `SportsFeatureViewV1`：3,655 行。
- 两次独立 feature build 的 canonical hash 均为
  `sha256:38d431567e9206662e4f92f2931c3c1df141c2d06972e6c420b2b5e86770a8f3`。
- 显式 factor definitions：104。
- 注册模板展开后的运行时 factor universe：1,472。
- 正式 factor × market family × horizon 结果格：1,747。
- 统计口径：10,000 次 game-cluster bootstrap、leave-one-game-out、BH correction、最大单场贡献和双 venue 方向检查。
- 历史正式推断当前只使用 moneyline actual trades；没有用成交价伪造 BBO/L2。

## 等待体育专家审核的三个结果

| Factor | Horizon | Episodes / Games | Kalshi / Poly episodes | 平均价格变化 | 95% CI | Diagnostic residual | 说明 |
|---|---:|---:|---:|---:|---:|---:|---|
| `NFL.EVENT.EXPLOSIVE_PLAY` | 30s | 58 / 18 | 9 / 51 | +2.706pp | [+1.759, +3.906]pp | +1.341pp | 两 venue 同方向；LOO 同号率 100%；最大单场贡献 18.1%。这是最值得人工检查的候选，但它仍是爆发性 play 后的条件均值，不是可交易超额收益。 |
| `NFL.PERSONNEL.BETWEEN_PLAY_ROSTER_CHANGE` | 1s | 131 / 20 | 58 / 74 | +0.178pp | [+0.067, +0.295]pp | -0.064pp，CI 跨 0 | 原始变化很小，且扣除 football-value baseline 后消失。人员数据仍是 research-only，相邻 snap 差分也不是官方 substitution timestamp；优先建议 `DATA_GAP` 或 `REJECT`。 |
| `NFL.SEQUENCE.ANSWER_SCORE` | 5s | 18 / 13 | 4 / 14 | +0.799pp | [+0.276, +1.375]pp | +0.741pp | 两 venue 同方向；LOO 同号率 100%；最大单场贡献 21.8%。体育机制合理，但样本很小、Kalshi 仅 4 个 episode，适合人工逐 case 审核。 |

这里的 `pp` 是合约概率价格的百分点。三个结果都只能进入 `AWAITING_EXPERT_REVIEW`，不能自动进入 shortlist。

## 没有升级的典型结果

- 原先 discovery-only 的 `distance > 10`：10 秒平均约 -0.267pp，CI 约
  [-0.492, -0.021]pp；但只有 65 episodes，未达到 primary gate
  `≥300 episodes / ≥16 games / 每 venue ≥100`，BH q≈0.305，因此为
  `INSUFFICIENT_SUPPORT`。
- `TD → PAT/2PT finalized`、PAT、pass TD 等在 60 秒上有较大正向条件均值，
  但存在 venue 不一致、单场贡献超限或支持不足，不能 shortlist。
- 所有 binary moderators 目前只报告
  `DESCRIPTIVE_CONDITIONAL_MEAN`。在没有正确 target-vs-complement base cohort
  之前，它们被代码硬性禁止进入 `AWAITING_EXPERT_REVIEW`。

## 显式数据缺口

- 961 个结果格为 `DATA_GAP`，其中大部分是已登记二维组合在当前实际成交覆盖下没有合格 observation。
- nfl4th 的 go / FG / punt action-conditional WP 尚未构建：本机没有固定 R/nfl4th runtime，因此 `fourth_down_decision_regret` 与 `punt_execution_surprise` 不生成伪值。
- 无 tracking、route、coverage、pressure、球员路径、官方 substitution timestamp 和完整 PIT injury/news 层；相关 Big Data Bowl 方法仅登记 method/data-gap card。
- 历史 Polymarket/Kalshi 没有完整 L2；不能研究 OFI、depth、bid/ask 谁先动或执行性 spread。
- 当前历史结果缺 `local_received_time`，不能把 source-time horizon 解释为真实反应 latency。

## 人工 Gate

专家需要对三个待审核结果分别选择：

- `ACCEPT`
- `REDEFINE`
- `DATA_GAP`
- `REJECT`

只有完成审核的 `ACCEPT` 定义才会产生 immutable `ShortlistLockV1`。在此之前，系统拒绝读取冻结的另外 20 场 holdout reaction。

