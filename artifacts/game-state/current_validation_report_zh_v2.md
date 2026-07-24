结论：**已验证的 prediction-market alpha：无；可用于 same-as-of 对称性检验的行数：0。**

# 当前验证报告（v2）

- 报告日期：2026-07-23（America/Chicago）
- 范围：NBA、NFL、足球、MLB、F1 的 game state、reducer、概率模型、当前受治理 full-path latency，以及 prediction-market 对齐
- 结论标签：`CURRENT_VALIDATION_SUMMARY_NON_PROMOTIONAL`

## 口径

1. **State 正确性**：确定性只说明同一输入可重复。同源字段一致性不是独立 oracle；“实现独立”也不等于“数据源独立”。
2. **模型质量**：Brier、log loss 越低越好；二分类 calibration slope/intercept 的理想值为 `1/0`。它们不是命中率，不同目标或类别数不能直接横比。
3. **速度**：reducer latency 只测 `state + event -> next state`。只有包含事件构造、reducer、特征、已加载模型和概率输出校验的证据，才在本报告中称为当前受治理 full path。
4. **市场对齐**：回溯性的 game/outcome 身份绑定不等于 matched-as-of。后者还要求 PIT 模型输出、同时间可执行双边深度、本地接收时间、暂停状态和 PIT venue rules。

## 一览

| 赛事 | State 证据等级 | 当前模型证据 | 当前受治理 full path | Canonical market binding | Matched-as-of | Alpha / 对称性 |
|---|---|---|---|---|---:|---|
| NBA | 仅 12 条 synthetic fixture | 无真实经验结果 | 无 | 0 | 0 | 未验证 |
| NFL | 2025 全季 285/285；实现独立但同源的 native-field oracle | 官方 no-spread V2，`PRELIMINARY`、`PIT_UNPROVEN` | 有 | 仅 3 场回溯身份绑定；无 as-of 全链绑定 | 0 | 未验证 |
| 足球 | 380 场 strict partial replay；同源比分一致性，无独立 oracle | 仅旧 Dixon–Coles POC；当前动态模型无经验结果 | 无 | 0 | 0 | 未验证 |
| MLB | 2025 regular-season 全量 census；仅同源 next-row 比较 | 无 | 无 | 0 | 0 | 未验证 |
| F1 | 仅 source inventory | 无 | 无 | 0 | 0 | 未验证 |

## NBA

- **State 覆盖与正确性**：当前只有 12 条 synthetic event，双次 replay hash 一致，覆盖 score、foul、timeout、possession、period 和 terminal。真实 NBA 比赛数为 0；这不是独立真实数据 oracle，也不能证明真实比赛字段正确。O-005 仍为 `BLOCKED`。[证据](./nba_state_engine_validation_v0.json)
- **Reducer latency**：synthetic reducer-only p50/p95/p99 为 `3,416 / 3,500 / 3,542 ns`，不含模型。
- **模型准确性与校准**：无可报告的真实 Brier、log loss 或 calibration；没有合格的 fitted NBA model。
- **当前 full-path model latency**：无。
- **市场绑定、matched-as-of、alpha**：canonical game-condition-outcome binding 为 0，matched-as-of 为 0；alpha 与对称性均未验证。[对齐审计](./prediction_market_alignment_audit_v0.json)

## NFL

- **State 覆盖与正确性**：Reducer v3 对 2025 REG/POST 的 `285/285` 场、`48,771` 行、`48,486` 次 transition 运行两轮，0 fail-closed game、0 failure，canonical hash 一致。Oracle 不调用 reducer 或 adapter helper，逐字段读取 native row，因此相对实现是独立的；但它仍使用同一冻结 nflverse 数据，并非外部第二数据源。全部已登记 field audit mismatch 为 0。全季 registry-backed EventEnvelope integration 仍为 `false`，观察模式是 offline reconstruction，不是 live PIT。[全季 state 证据](./nfl_season_state_validation_v2.json)
- **Reducer latency**：当前全季 reducer-only p50/p95/p99 为 `8,000 / 8,667 / 17,500 ns`；不含解析、I/O、envelope、模型、market join 或网络。
- **当前模型准确性与校准**：官方 `fastrmodels` no-spread V2 使用 2021–2025 的 `205,607` 个 eligible rows、`1,420` 场非平局比赛；官方模型未重训，也未拟合额外 calibrator。Row-micro Brier `0.166284941293`、log loss `0.491426483873`、calibration slope `0.987612141958`、intercept `-0.047990070225`。结果为 `PRELIMINARY`，观察模式是 offline reconstruction，PIT 状态为 `PIT_UNPROVEN`。[JSON 证据](./nfl/fastrmodels_no_spread_reproduction_v2.json) / [简表](./nfl/fastrmodels_no_spread_reproduction_v2.md)
- **当前受治理 full-path model latency**：p50/p95/p99 为 `0.468896 / 1.611483 / 4.412516 ms`（1,000 samples）。路径包含 normalized event construction、state transition、官方 feature projection、预加载官方 booster 和 probability validation；不含 I/O、registry loading、network 或 market join，因此不是端到端 live SLA。
- **旧 X-11 POC，仅作历史上下文**：logistic 的 Brier/log loss/slope/intercept 为 `0.15348 / 0.46527 / 0.94255 / 0.04234`；GBDT 为 `0.16014 / 0.49479 / 1.64871 / -0.08031`；spread prior 为 `0.21084 / 0.60882 / 1.03777 / -0.01190`。[POC 证据](./nfl/x11_real_data_pipeline_evidence_v0.json) 这些数值不是官方 no-spread V2，也不是 prediction-market 比较；spread prior 来自 sportsbook spread，且其精确观察时间不可证明。
- **市场绑定**：专项回溯审计把 4 个历史 condition、17 条 fill 中的 3 场比赛绑定为 6 份 canonical outcome documents；另 1 个取消并 50/50 结算的 condition 未绑定。该 metadata 是赛后于 2026 年抓取，只能证明 retrospective identity，且没有 L2、本地接收时间、joinable model output 或 PIT venue-rule snapshot。[回溯绑定审计](./nfl/polymarket_v1_game_binding_audit_v0.json)
- **Matched-as-of、alpha、对称性**：`matched_as_of_rows=0`；无 profit/return 计算，无可执行双边对称性样本，prediction-market alpha 未验证。

## 足球

- **State 覆盖与正确性**：StatsBomb Premier League 2015/16 strict replay 扫描 380 场、`1,313,773` 条 source events；两次运行一致。Reducer 只推进每场首个 strict violation 之前的有效前缀，共 `1,090,826` 次成功 reduction；137 场在首个异常处 fail closed，243 场推进到源文件末尾但 finalization 仍 unproven，`finished/terminal` 被证明的场数为 0。不得把这 380 场称为 reducer 已完成的比赛。[strict replay](./soccer_real_replay_validation_v2.json)
- **正确性边界**：冻结 match metadata 与 event rows 的最终比分是 `380/380` 同源一致；两者都来自 StatsBomb，因此不是独立 oracle，也不证明 partial reducer state 是最终比分。
- **Reducer latency**：成功前缀调用的 reducer-only p50/p95/p99 为 `0.037917 / 0.052750 / 0.094292 ms`；不含失败调用、event adaptation、envelope、特征、模型或 market join。
- **模型准确性与校准**：当前 checked-in 经验结果只有旧的 pregame Dixon–Coles POC，共 280 个 held-out observations：三分类 Brier `0.633308625042`、log loss `1.056635838335`。OVR slope/intercept 分别为 home `0.744264 / 0.025523`、draw `-0.132576 / -1.080458`、away `0.560946 / -0.531134`；相对简单 expanding baseline 的 Brier 与 log-loss paired CI 均跨 0。[X-12 POC](./soccer/x12_real_data_poc_v0.json)
- **当前动态模型与 full-path latency**：state-conditioned dynamic-intensity model 只有工程实现，没有已登记 empirical result，也没有当前 full-path latency。旧 [`model_latency_v0`](./model_latency_v0.json) 对应已退役静态 head，本报告不把其中任何数字当作当前结果。[复现边界](./mature_model_reproduction_audit_v2.md)
- **市场绑定、matched-as-of、alpha**：canonical binding 为 0，matched-as-of 为 0；没有 PIT market prior 或 joinable model output，alpha 与对称性未验证。[对齐审计](./prediction_market_alignment_audit_v0.json)

## MLB

- **State 覆盖与正确性**：Retrosheet 2025 regular-season 全量 census 覆盖 `2,430` 场、`216,845` 条 native play records；cwevent 产出 189,311 个 state-transition events。`2,213` 场完全支持，`217` 场含 unsupported gap；共有 `932` 次 reducer call fail closed。两次运行 canonical hash 一致，但比较答案来自同一冻结 Retrosheet/cwevent 流的 immediate next row，不是独立 oracle。600 个 bases mismatch 对应当前不支持的 automatic-runner context；full native lifecycle 与 runner-identity lifecycle 都未证明。[regular-season census](./mlb/season_state_census_v0.json)
- **Reducer latency**：reducer-only p50/p95/p99 为 `6,958 / 8,084 / 10,292 ns`；不含 cwevent、CSV parsing、event construction、比较、hash、模型、网络或 market join。
- **模型准确性与校准**：无已登记 MLB 概率模型，无 Brier、log loss 或 calibration。
- **当前 full-path model latency**：无。
- **市场绑定、matched-as-of、alpha**：canonical binding 为 0，matched-as-of 为 0；alpha 与对称性未验证。[对齐审计](./prediction_market_alignment_audit_v0.json)

## F1

- **覆盖与正确性**：当前只是 inventory：Jolpica 有 2025 年 24 个分站的 winner result、round 1 的 1 条 lap timing 样本和 82 条 pit-stop records。没有 F1 reducer 或 replay；Jolpica 为 `research_only` 且 release unresolved，FastF1 因 timing rights 未解决而 blocked，未下载 timing/telemetry。[inventory](./f1/source_inventory_v0.json)
- **Reducer latency**：无。
- **模型准确性、校准与当前 full-path latency**：均无。
- **市场绑定、matched-as-of、alpha**：canonical race-condition-outcome binding 为 0，matched-as-of 为 0；alpha 与对称性未验证。[对齐审计](./prediction_market_alignment_audit_v0.json)

## Prediction-market 数据为什么仍不能给出 alpha

- PMXT Phase 0 只是 archive inventory：覆盖 `2,401 / 2,404` 小时，报告总量 `945,792,482,370 bytes`（约 `945.8 GB`），缺 3 小时。它不证明 L2 已重建。[Phase 0 inventory](../data-audit/phase0_inventory.json)
- X-01 的 2026-05-28 full-day preflight 有 24 个对象、`11,479,773,675` compressed bytes（约 `11.48 GB`）、`1,971,336,963` rows；但 `reconstruction_executed=false`、`x01_formal_gate_passed=false`。[X-01 preflight](../data-audit/x01_full_day_preflight_v1.json)
- NFL 的 3 场回溯身份绑定没有改变 as-of 结论：metadata 是赛后抓取，17 条记录是 fills 而不是 L2，也没有本地接收时间或模型输出。
- 冻结的总对齐审计仍记录 full-path canonical assertion count 为 0；NFL 专项审计新增的 3 场、6 份 outcome metadata 只属于赛后身份绑定，不回写该 v0 审计，也不形成 as-of join。
- Sportsbook spread prior 不是 Polymarket/Kalshi prediction-market probability，不能据此声称市场 alpha。

因此，没有任何赛事同时具备同一 canonical game/outcome、PIT model cutoff、同时间可执行 bid/ask 与正 depth、pause state、quote-age policy 和 PIT venue rules。可计算的 same-as-of 对称性行数仍为 0；lead-lag、mispricing、predictive disagreement、profit 和 alpha 均未验证。[总对齐审计](./prediction_market_alignment_audit_v0.json)

## Public repository 与 NO-GO

[WeKruit/SAF](https://github.com/WeKruit/SAF) 是 public repository。代码、contracts、tests、evidence、registries 和不含敏感信息的 manifest 可以提交；标准 `data/raw`、`var/raw` 根下的 raw objects 继续由 [`.gitignore`](../../.gitignore) 排除。

当前 production decision 是 `NOT_READY`。NO-GO 继续禁止：真实资金交易、live maker、精确 queue-fill 声称、multi-venue live arbitrage、live copy trading、LLM hot path、RL、大规模微服务化，以及未登记 quick backtest；README 或论文中的收益也不能当作本仓库证据。[Production Readiness](../compliance/production_readiness_v0.md)
