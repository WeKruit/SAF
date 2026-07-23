结论：目前没有已验证的 prediction-market alpha，也没有已验证的 game-state 与 prediction market 同时点“对称性”；可用于 matched-as-of 比较的行数是 0。

# 当前验证报告（v1）

- 日期：2026-07-23
- 结论标签：`CURRENT_VALIDATION_SUMMARY_NON_PROMOTIONAL`
- 范围：NBA、NFL、足球、MLB、F1 的 game-state、概率模型、运行速度，以及 prediction-market 同时点对齐
- 边界：本报告不把 reducer 正确性当作模型准确性，不把 reducer 延迟当作模型推理延迟，也不把 sportsbook spread prior 当作 prediction-market 概率

## 2026-07-23 本轮增量

- NFL：已按官方 `fastrmodels`/`nflfastR` 源码冻结 no-spread 11 维和
  spread 12 维的特征顺序与变换。预测 cutoff 只能从与 reducer state
  同一份、通过 normalized envelope + 两个 raw parents 复验的 nflverse
  `EventEnvelope.source_at` 派生；canonical game、event ID、sequence、
  source play、season、raw object hash 和 parent time 必须全链一致。
  receiver/spread 观测时间、声明的 manifest digest、PIT 状态及对应
  UBJ SHA-256 也进入 feature hash；manifest digest 在该 seam 中只记录
  声明，不冒充已复验。当前仍未执行官方 booster，因此没有新的官方
  模型 accuracy 或 latency 数字。
- 足球：已实现使用当前 half、score difference、red-card difference 的
  state-conditioned dynamic-intensity head。新的 transition v1 protocol
  使用完整比赛日期组的 50% base fit / 25% temperature calibration /
  25% final test，最终概率经过单一 multiclass temperature，再报告 OVR
  calibration、game-cluster bootstrap 及相对静态 pregame head 的 paired
  CI。历史 Dixon–Coles 1X2 结果继续只读保留在 v0，不在 v1 重算或改名。
  目前只通过 synthetic/fixture 工程验证；Team H 尚未登记新的 model IDs
  和真实运行 hashes，所以没有新的 empirical X-12 transition 结果。
- Event seam：新 latency v1 路径使用真实 StatsBomb EventEnvelope ID，
  依次执行 state/event、reducer、feature、dynamic model、temperature
  calibration 和 ModelOutputV1 validation。
  旧 `model_latency_v0.json` 的足球数字属于已退役静态 head，现行代码
  会同时按 report version 与 model identity 拒绝它；在
  `MODEL-SOCCER-DYNAMIC-INTENSITY` registry row、动态 evidence 和原始
  实验锁全部完成前，不发布替代速度数字。
- 治理：新 reproduction gate 会复算真实 code/data/model bindings，
  拒绝 future-dated amendment/evaluation/result，继承原实验的全部
  PIT/split/calibration/bootstrap/config/H-approval locks，并要求
  X-11/X-12 的精确新模型集合；X-12 code hash 同时覆盖模型实现和完整
  runner。离线 X-12 snapshot 没有真实 reducer event ID 时，不再合成
  `state_event_id`，而是明确 `contract_output=null`。

## 先区分四件事

1. **State replay correctness**：同一冻结事件流重复回放且 canonical hash 一致只能证明确定性；要声称字段正确，还必须直接对照独立 native 字段或外部 oracle，不能让同一个 adapter 同时生成结果和“答案”。
2. **概率模型质量**：Brier 与 log loss 都是越低越好；calibration slope 越接近 1、intercept 越接近 0，概率校准越接近理想状态。它们不是“命中率”，不同目标、类别数和 horizon 的 Brier 不能直接横向比较。
3. **速度**：reducer latency 只测 `state + event -> next state`。本报告只把 `state + event -> reducer -> feature -> fitted registered transition model -> ModelOutputV1 validation` 记作 full path；它仍不含网络、源文件读取、registry 加载、market join 或训练。Latency 使用 evidence 对应的冻结 fitted instance：有 walk-forward 的 NFL 取最后一个 fold，足球动态 v1 取冻结的 base-fit、dynamic-fit 与 calibration parameters；准确性仍按各自冻结 protocol 的完整 final-test 聚合，两者共享 model family、数据和配置绑定，但不冒充同一组 fitted parameters。
4. **Prediction-market 对齐/alpha**：至少需要同一个 canonical game/condition/outcome、PIT model cutoff、同时间可执行 bid/ask、双边正 depth、pause 状态和 venue-rule snapshot。当前这些条件没有在同一行同时成立。[对齐审计](./prediction_market_alignment_audit_v0.json)记录 `matched_as_of_rows=0`。

## 逐赛事结果

| 赛事 | 数据覆盖与权利状态 | State 验证 | 概率模型质量 | 本机速度 | Prediction-market join |
|---|---|---|---|---|---|
| NBA | 当前只有 X-06 synthetic contract fixture；真实 NBA game 数为 0。O-005 仍为 `BLOCKED`，所以没有合法资格产出正式真实 NBA 结果。[NBA reducer validation](./nba_state_engine_validation_v0.json) / [NBA baseline](./nba/baseline_v0.md) | 12 个 synthetic event 回放两次，hash 一致；覆盖 score、foul、timeout、possession、period、terminal。标签为 `PRELIMINARY_ENGINEERING_VALIDATION`，不可当作真实比赛验证。[证据](./nba_state_engine_validation_v0.json) | 未测。prior/logistic/GBDT/possession-transition 只有受治理的代码与 contract，没有 empirical Brier、log loss 或 calibration。[证据](./nba_state_engine_validation_v0.json) | Reducer-only：p50 `3,416 ns`、p95 `3,500 ns`、p99 `3,542 ns`；模型推理未测。[证据](./nba_state_engine_validation_v0.json) | `not_aligned`；canonical game-condition-outcome binding 为 0，matched-as-of 为 0。[证据](./prediction_market_alignment_audit_v0.json) |
| NFL | nflverse 2015–2025 REG/POST，共 3,028 games、532,376 rows；数据集许可为 `approved`，但 spread 的精确赛前观察时间不可证明，因此 X-11 为 `PRELIMINARY`、`PIT_UNPROVEN`，不是正式结果。[X-11 evidence](./nfl/x11_real_data_pipeline_evidence_v0.json) | Reducer v2 在一个完整真实比赛的 182 rows / 181 transitions 上回放两次，直接对照 native pre/post score、timeout、clock 字段与显式 context-carry 规则，hash 一致；修复了 scoring-row 的错位语义，并以单场 same-row TD/XP assertions 验证。旧的 281/285 season scan 已作废，v2 尚未完成 season census；suspension、clock correction、`order_sequence` 和 postseason OT rules 仍是 P1。[NFL replay](./nfl_real_replay_validation_v1.json) | 2020–2025 walk-forward 评估 1,693 games；二分类 1,688 games / 36,734 observations。Logistic：Brier `0.15348`、log loss `0.46527`、slope `0.94255`、intercept `0.04234`；GBDT：`0.16014`、`0.49479`、`1.64871`、`-0.08031`；spread prior：`0.21084`、`0.60882`、`1.03777`、`-0.01190`。下一 drive 五分类为 Brier `0.73096`、log loss `1.44065`，36,867 observations。这些模型走独立 X-11 pipeline，未因 reducer 修复而改写。[X-11 evidence](./nfl/x11_real_data_pipeline_evidence_v0.json) | Reducer-v2 单场 p50/p95/p99：`3,709/3,875/3,917 ns`。[NFL replay](./nfl_real_replay_validation_v1.json) 历史 v0 工程 benchmark 的下一 drive full path 为 `0.178333/0.181542/0.186917 ms`（约 `5,648 events/s`），model-only 为 `0.107959/0.110875/0.112541 ms`；整个 v0 report 已被当前 v1 validator 退役，因此这些数值不是现行 governed latency，v1 尚未发布。[历史 latency](./model_latency_v0.json) | `not_aligned`；4 个历史 NFL condition / 17 fills 只有成交、没有 L2 与 local receive time，canonical binding 和 matched-as-of 都为 0。[对齐审计](./prediction_market_alignment_audit_v0.json) |
| 足球 | StatsBomb Open Data，Premier League 2015/16：380 matches、1,313,773 events；许可为 `research_only`，离线事件可用性不等于 live PIT。[X-12 evidence](./soccer/x12_real_data_poc_v0.json) | Reducer v2 对冻结全季运行两轮：两轮都 380/380 完成、每轮 1,313,773 events、最终比分 mismatch 0、fail-closed 0、聚合 hash 相同；显式保留 139 个同 period clock regression 和 3 个 source coordinate anomaly。完整 registry-backed envelope integration 仍只在一场 3,175-event 比赛执行；全季 census 不是逐事件独立 oracle。[Soccer replay](./soccer_real_replay_validation_v1.json) | Checked-in Dixon–Coles 1X2：280 observations，Brier `0.63331`、log loss `1.05664`；相对简单 expanding baseline 的 Brier / log-loss CI 都跨 0。旧五分钟静态 head：5,040 observations，Brier `0.24553`、log loss `0.49220`；它已被动态 head 取代，不能作为新模型准确率。新动态 head 尚无已登记真实结果。[X-12 evidence](./soccer/x12_real_data_poc_v0.json) | Reducer-v2 单场全事件 p50/p95/p99：`0.036292/0.047250/0.049500 ms`。[Soccer replay](./soccer_real_replay_validation_v1.json) 旧静态五分钟 full path 的 `0.094667/0.097375/0.099792 ms` 已退役，不可作为动态模型速度；新 latency 数字待动态 evidence 注册后重测。[历史 latency](./model_latency_v0.json) | `not_aligned`；没有 PIT market prior、没有 joinable empirical ModelOutputV1，canonical binding 和 matched-as-of 都为 0。[对齐审计](./prediction_market_alignment_audit_v0.json) |
| MLB | Retrosheet 2025 inventory：2,430 games、216,845 play records；`research_only`，源数据可能后续修订，因此只认冻结字节与 hash。[MLB inventory](./mlb/source_inventory_v0.json) | 一个完整真实比赛 88 events / 87 next-observation comparisons，mismatch 为 0，两次 hash 一致、terminal=true。标签为 `ENGINEERING_VALIDATION`；使用 program-root 验证的 EventEnvelope，并绑定原始行、后继行、cwevent binary/command/output 与 manifest，仍不是已登记的概率实验。[MLB replay](./mlb_real_replay_validation_v0.json) | 未测：没有已登记 MLB 概率模型、Brier、log loss 或 calibration。[MLB replay](./mlb_real_replay_validation_v0.json) | Reducer-only（含完整 offline provenance continuity 校验）p50/p95/p99：`17.042/17.709/18.000 µs`；模型 inference 未测。[MLB replay](./mlb_real_replay_validation_v0.json) | `not_aligned`；没有概率输出、canonical game-condition mapping 或同时间 L2，matched-as-of 为 0。[对齐审计](./prediction_market_alignment_audit_v0.json) |
| F1 | Jolpica 仅完成 bounded inventory：2025 年 24 个已完成分站 winner result；另有 round 1 的 1 条 lap timing 样本和 82 条 pit-stop records。Jolpica 为 `research_only` 且 release 未锁；FastF1 因 upstream rights unresolved 为 `blocked`，未下载 timing/telemetry。[F1 inventory](./f1/source_inventory_v0.json) | 未测：没有 F1 state reducer 或 replay artifact。[F1 inventory](./f1/source_inventory_v0.json) | 未测：没有模型、Brier、log loss 或 calibration。[F1 inventory](./f1/source_inventory_v0.json) | Reducer 与模型 inference 都未测。[Latency](./model_latency_v0.json) | `not_aligned`；没有 race-condition-outcome mapping 或同时间 L2，matched-as-of 为 0。[对齐审计](./prediction_market_alignment_audit_v0.json) |

## 现在能说的“准确性”

NFL 是当前唯一同时有多年真实数据、game-grouped chronological
walk-forward 和真实概率指标的赛事；它另有一个已退役 v0 artifact
中的历史模型推理 benchmark，但还没有 current v1 governed latency。
它的 logistic 在这份 POC 样本上优于 spread-derived prior：
model-minus-prior 的 Brier 95% CI 为 `[-0.06483, -0.05133]`，log-loss
CI 为 `[-0.16048, -0.12926]`。[X-11 evidence](./nfl/x11_real_data_pipeline_evidence_v0.json)

但这里的 **spread prior 是 sportsbook spread 派生 prior，不是 Polymarket/Kalshi prediction-market probability**；而且 nflverse 对该 spread 没有精确的赛前 observation timestamp，所以这只能描述当前 X-11 POC 内部比较，不能转述为 prediction-market alpha。

足球的 Dixon–Coles 点估计优于简单 empirical baseline，但 paired bootstrap CI 跨 0，因此当前证据不支持“稳定优于简单 baseline”。此外 X-12 没有 PIT market prior，StatsBomb 许可为 `research_only`，结果标签是 `POC_NO_PIT_MARKET_PRIOR`。[X-12 evidence](./soccer/x12_real_data_poc_v0.json)

NBA、MLB 和 F1 没有可报告的真实概率模型准确性；NBA 的 12-event synthetic replay、MLB 的单场真实 replay、F1 的数据 inventory 都不能替代预测评估。

足球未来 5 分钟的现有 Brier / log loss 只评价“pregame intensity 重复到各 cutoff”的 POC，不能证明 `current state + event -> next transition distribution` 已经正确。NFL 下一 drive head 有真实 walk-forward 指标，但 reducer v2 还没有成为它的受登记特征源；两条证据必须继续分开。

## 对公开成熟研究的复现决定

- NFL 第一优先是运行已冻结的官方 nflverse `fastrmodels` no-spread/spread booster，而不是使用发布的 `home_wp` 或 README 数字；之后再按同一特征定义逐 fold 重训，比较现有 SAF logistic/GBDT 和冻结 spread prior。
- 足球已完成 Maia-family state-conditioned Poisson intensity 的核心实现，只使用 cutoff 时已知的 team base rate、half、score difference 和 red-card difference；下一步是先注册新模型与 hashes，再运行真实 expanding walk-forward。
- 新拟合前必须由 Team H amendment 冻结 model ID/version、特征、manifest、fold、seed 和 calibration interval。当前完成了 primary-source review、核心接口和 fail-closed 工程测试；尚未运行官方 NFL booster 或新的真实足球动态模型实验。
- NBA 真实模型继续受数据权利阻塞；MLB/F1 本阶段仍是 inventory/evidence，不把论文或 README 指标当 SAF 结果。

完整来源、许可、PIT 风险与 frozen calibration protocol 见 [公开 baseline 与校准审查](./research_calibration_review_v1.md)。

## Prediction-market 数据验证到了哪里

- PMXT Phase 0 已盘点 2,401 / 2,404 小时、945,792,482,370 bytes，缺 3 个小时；这证明 archive inventory 可枚举，不证明 L2 已全量重建。[Phase 0 inventory](../data-audit/phase0_inventory.json)
- X-01 的 2026-05-28 full-day preflight 已冻结 24 个对象、11,479,773,675 compressed bytes、1,971,336,963 rows、51,115 markets 和 102,219 assets；但 `reconstruction_executed=false`、`x01_formal_gate_passed=false`。[X-01 preflight](../data-audit/x01_full_day_preflight_v1.json)
- Polymarket forward capture 截止审计包含 17 个 sealed manifests、39,624 capture records 和 3,212 个 native sport condition IDs；只有 10 个 quote markets，432 个 book event 全部缺至少一侧，canonical sport-game mapping 为 0。[对齐审计](./prediction_market_alignment_audit_v0.json)
- 审计 cutoff 之后又封存 UTC
  [hour 16](../../var/raw/manifests/source=polymarket/stream=market/date=2026-07-23/hour=16/20ddf6118cfa425f4b0eeb5cd9ebae5063336e985e1f8f356212e120afbf3d53.manifest.json)、
  [hour 17](../../var/raw/manifests/source=polymarket/stream=market/date=2026-07-23/hour=17/00291a7cb409536d1cd88ee0e46023ea8c3e91b622845008d5bef67f4bd9c7a0.manifest.json)、
  [hour 18](../../var/raw/manifests/source=polymarket/stream=market/date=2026-07-23/hour=18/f698048e5cf7529d5a5b47e316cba8d7f69f25a4cdca30e2d02e07c6694cabf7.manifest.json)
  和
  [hour 19](../../var/raw/manifests/source=polymarket/stream=market/date=2026-07-23/hour=19/40f11b9144be4e9b0225fe908114e3d5adba34c0e5c4862dcc8dae7036040088.manifest.json)
  四个 manifest，共 28,654 records；它们证明 recorder 仍在产出
  immutable segment，但尚未纳入对齐审计，也没有改变 canonical
  binding / matched-as-of 为 0 的结论。
- Polymarket-v1 bounded NFL extract 只有 4 个 condition 和 17 条 fill；它没有历史 L2、没有 local receive time，也没有 canonical game binding。[bounded extract](../data-audit/polymarket_v1_bounded_sports_extract_v0.json)

所以当前“对称性”没有可计算样本：没有同一比赛、同一 outcome、同一 as-of cutoff 下的模型概率与可执行双边盘口。相关性、lead-lag、mispricing、predictive disagreement 和 alpha 均未被验证。

## SAF public repository 边界

[WeKruit/SAF](https://github.com/WeKruit/SAF) 是 public repository。仓库提交代码、contracts、tests、evidence、registries，以及不含凭证/敏感信息的 manifests；本机 `var/raw` 约 37GB 的 raw bytes 不提交。该边界由 [`.gitignore`](../../.gitignore) 中只放行 `var/raw/manifests/**`、继续排除 `var/raw` 数据对象的规则执行。

所有上述结论持续受 NO-GO 约束：没有真实资金动作、没有 maker、没有 queue-fill 精确声称、没有 multi-venue live arbitrage、没有 live copy trading、没有 LLM hot path、没有 RL，也没有未登记回测。
