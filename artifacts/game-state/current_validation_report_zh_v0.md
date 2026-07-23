结论：目前没有已验证的 prediction-market alpha，也没有已验证的 game-state 与 prediction market 同时点“对称性”；可用于 matched-as-of 比较的行数是 0。

# 当前验证报告（v0）

- 日期：2026-07-23
- 结论标签：`CURRENT_VALIDATION_SUMMARY_NON_PROMOTIONAL`
- 范围：NBA、NFL、足球、MLB、F1 的 game-state、概率模型、运行速度，以及 prediction-market 同时点对齐
- 边界：本报告不把 reducer 正确性当作模型准确性，不把 reducer 延迟当作模型推理延迟，也不把 sportsbook spread prior 当作 prediction-market 概率

## 先区分四件事

1. **State replay correctness**：同一冻结事件流重复回放，逐字段状态一致且 canonical hash 一致。这只能证明 adapter/reducer 的确定性和 fail-closed 行为。
2. **概率模型质量**：Brier 与 log loss 都是越低越好；calibration slope 越接近 1、intercept 越接近 0，概率校准越接近理想状态。它们不是“命中率”，不同目标、类别数和 horizon 的 Brier 不能直接横向比较。
3. **速度**：reducer latency 只测 `state + event -> next state`。本报告只把 `state + event -> reducer -> feature -> fitted registered transition model -> ModelOutputV1 validation` 记作 full path；它仍不含网络、源文件读取、registry 加载、market join 或训练。Latency 使用最后一个训练 fold 的冻结模型实例；准确性是全部 walk-forward folds 的聚合结果，两者共享 model family、数据和配置绑定，但不冒充同一组 fitted parameters。
4. **Prediction-market 对齐/alpha**：至少需要同一个 canonical game/condition/outcome、PIT model cutoff、同时间可执行 bid/ask、双边正 depth、pause 状态和 venue-rule snapshot。当前这些条件没有在同一行同时成立。[对齐审计](./prediction_market_alignment_audit_v0.json)记录 `matched_as_of_rows=0`。

## 逐赛事结果

| 赛事 | 数据覆盖与权利状态 | State 验证 | 概率模型质量 | 本机速度 | Prediction-market join |
|---|---|---|---|---|---|
| NBA | 当前只有 X-06 synthetic contract fixture；真实 NBA game 数为 0。O-005 仍为 `BLOCKED`，所以没有合法资格产出正式真实 NBA 结果。[NBA reducer validation](./nba_state_engine_validation_v0.json) / [NBA baseline](./nba/baseline_v0.md) | 12 个 synthetic event 回放两次，hash 一致；覆盖 score、foul、timeout、possession、period、terminal。标签为 `PRELIMINARY_ENGINEERING_VALIDATION`，不可当作真实比赛验证。[证据](./nba_state_engine_validation_v0.json) | 未测。prior/logistic/GBDT/possession-transition 只有受治理的代码与 contract，没有 empirical Brier、log loss 或 calibration。[证据](./nba_state_engine_validation_v0.json) | Reducer-only：p50 `3,416 ns`、p95 `3,500 ns`、p99 `3,542 ns`；模型推理未测。[证据](./nba_state_engine_validation_v0.json) | `not_aligned`；canonical game-condition-outcome binding 为 0，matched-as-of 为 0。[证据](./prediction_market_alignment_audit_v0.json) |
| NFL | nflverse 2015–2025 REG/POST，共 3,028 games、532,376 rows；数据集许可为 `approved`，但 spread 的精确赛前观察时间不可证明，因此 X-11 为 `PRELIMINARY`、`PIT_UNPROVEN`，不是正式结果。[X-11 evidence](./nfl/x11_real_data_pipeline_evidence_v0.json) | 一个完整真实比赛 182 rows / 181 transitions，逐字段 mismatch 为 0，两次 final hash 一致。全 2025 partition 工程扫描 285 games：281 完整通过，4 场在缺 clock、clock 回退或负 timeout 处 fail closed；每轮在边界前完成 48,298 次逐状态比较。[NFL replay](./nfl_real_replay_validation_v0.json) | 2020–2025 walk-forward 评估 1,693 games；二分类 1,688 games / 36,734 observations。Logistic：Brier `0.15348`、log loss `0.46527`、slope `0.94255`、intercept `0.04234`；GBDT：`0.16014`、`0.49479`、`1.64871`、`-0.08031`；spread prior：`0.21084`、`0.60882`、`1.03777`、`-0.01190`。下一 drive 五分类为 Brier `0.73096`、log loss `1.44065`，36,867 observations。[X-11 evidence](./nfl/x11_real_data_pipeline_evidence_v0.json) | Reducer-only p50/p95/p99：`3,792/3,958/4,042 ns`。[NFL replay](./nfl_real_replay_validation_v0.json) 下一 drive transition full path：`0.173500/0.178250/0.180625 ms`，约 `5,808 events/s`。最终胜负 logistic/GBDT 尚未接入同一 full-path contract，因此不报告旧的 inference-only 数字。[Latency](./model_latency_v0.json) | `not_aligned`；4 个历史 NFL condition / 17 fills 只有成交、没有 L2 与 local receive time，canonical binding 和 matched-as-of 都为 0。[对齐审计](./prediction_market_alignment_audit_v0.json) |
| 足球 | StatsBomb Open Data，Premier League 2015/16：380 matches、1,313,773 events；许可为 `research_only`，离线事件可用性不等于 live PIT。[X-12 evidence](./soccer/x12_real_data_poc_v0.json) | 一个真实比赛 3,175 events 完整回放两次，hash 一致，比分/红牌/换人等 invariant 通过；源文件没有显式 `Match End`，因此 reducer 正确地保留 `terminal=false`。1 个 source-clock anomaly 默认拒绝，只有 normalized envelope 同时携带 `clock_jump` 与 `out_of_order` 才被接受并保留，没有静默改写。[Soccer replay](./soccer_real_replay_validation_v0.json) | Dixon–Coles 1X2：280 observations，Brier `0.63331`、log loss `1.05664`；OVR slope/intercept：home `0.744/0.026`、draw `-0.133/-1.080`、away `0.561/-0.531`。相对简单 expanding baseline 的 Brier delta `-0.02444`、95% CI `[-0.06038, 0.01399]`；log-loss delta `-0.02917`、CI `[-0.08354, 0.02888]`，两个 CI 都跨 0，不能声称稳定改善。未来 5 分钟 goal transition：5,040 observations，Brier `0.24553`、log loss `0.49220`。[X-12 evidence](./soccer/x12_real_data_poc_v0.json) | Reducer-only p50/p95/p99：`0.036416/0.047500/0.052208 ms`。[Soccer replay](./soccer_real_replay_validation_v0.json) 未来 5 分钟 transition full path：`0.092125/0.095125/0.098292 ms`，约 `10,835 events/s`。Dixon–Coles 尚未接入同一 full-path contract。[Latency](./model_latency_v0.json) | `not_aligned`；没有 PIT market prior、没有 joinable ModelOutputV1，canonical binding 和 matched-as-of 都为 0。[对齐审计](./prediction_market_alignment_audit_v0.json) |
| MLB | Retrosheet 2025 inventory：2,430 games、216,845 play records；`research_only`，源数据可能后续修订，因此只认冻结字节与 hash。[MLB inventory](./mlb/source_inventory_v0.json) | 一个完整真实比赛 88 events / 87 next-observation comparisons，mismatch 为 0，两次 hash 一致、terminal=true。标签为 `ENGINEERING_VALIDATION`；使用 program-root 验证的 EventEnvelope，并绑定原始行、后继行、cwevent binary/command/output 与 manifest，仍不是已登记的概率实验。[MLB replay](./mlb_real_replay_validation_v0.json) | 未测：没有已登记 MLB 概率模型、Brier、log loss 或 calibration。[MLB replay](./mlb_real_replay_validation_v0.json) | Reducer-only（含完整 offline provenance continuity 校验）p50/p95/p99：`17.042/17.709/18.000 µs`；模型 inference 未测。[MLB replay](./mlb_real_replay_validation_v0.json) | `not_aligned`；没有概率输出、canonical game-condition mapping 或同时间 L2，matched-as-of 为 0。[对齐审计](./prediction_market_alignment_audit_v0.json) |
| F1 | Jolpica 仅完成 bounded inventory：2025 年 24 个已完成分站 winner result；另有 round 1 的 1 条 lap timing 样本和 82 条 pit-stop records。Jolpica 为 `research_only` 且 release 未锁；FastF1 因 upstream rights unresolved 为 `blocked`，未下载 timing/telemetry。[F1 inventory](./f1/source_inventory_v0.json) | 未测：没有 F1 state reducer 或 replay artifact。[F1 inventory](./f1/source_inventory_v0.json) | 未测：没有模型、Brier、log loss 或 calibration。[F1 inventory](./f1/source_inventory_v0.json) | Reducer 与模型 inference 都未测。[Latency](./model_latency_v0.json) | `not_aligned`；没有 race-condition-outcome mapping 或同时间 L2，matched-as-of 为 0。[对齐审计](./prediction_market_alignment_audit_v0.json) |

## 现在能说的“准确性”

NFL 是当前唯一同时有多年真实数据、game-grouped chronological walk-forward、概率指标和模型推理 latency 的赛事。它的 logistic 在这份 POC 样本上优于 spread-derived prior：model-minus-prior 的 Brier 95% CI 为 `[-0.06483, -0.05133]`，log-loss CI 为 `[-0.16048, -0.12926]`。[X-11 evidence](./nfl/x11_real_data_pipeline_evidence_v0.json)

但这里的 **spread prior 是 sportsbook spread 派生 prior，不是 Polymarket/Kalshi prediction-market probability**；而且 nflverse 对该 spread 没有精确的赛前 observation timestamp，所以这只能描述当前 X-11 POC 内部比较，不能转述为 prediction-market alpha。

足球的 Dixon–Coles 点估计优于简单 empirical baseline，但 paired bootstrap CI 跨 0，因此当前证据不支持“稳定优于简单 baseline”。此外 X-12 没有 PIT market prior，StatsBomb 许可为 `research_only`，结果标签是 `POC_NO_PIT_MARKET_PRIOR`。[X-12 evidence](./soccer/x12_real_data_poc_v0.json)

NBA、MLB 和 F1 没有可报告的真实概率模型准确性；NBA 的 12-event synthetic replay、MLB 的单场真实 replay、F1 的数据 inventory 都不能替代预测评估。

## Prediction-market 数据验证到了哪里

- PMXT Phase 0 已盘点 2,401 / 2,404 小时、945,792,482,370 bytes，缺 3 个小时；这证明 archive inventory 可枚举，不证明 L2 已全量重建。[Phase 0 inventory](../data-audit/phase0_inventory.json)
- X-01 的 2026-05-28 full-day preflight 已冻结 24 个对象、11,479,773,675 compressed bytes、1,971,336,963 rows、51,115 markets 和 102,219 assets；但 `reconstruction_executed=false`、`x01_formal_gate_passed=false`。[X-01 preflight](../data-audit/x01_full_day_preflight_v1.json)
- Polymarket forward capture 截止审计包含 17 个 sealed manifests、39,624 capture records 和 3,212 个 native sport condition IDs；只有 10 个 quote markets，432 个 book event 全部缺至少一侧，canonical sport-game mapping 为 0。[对齐审计](./prediction_market_alignment_audit_v0.json)
- Polymarket-v1 bounded NFL extract 只有 4 个 condition 和 17 条 fill；它没有历史 L2、没有 local receive time，也没有 canonical game binding。[bounded extract](../data-audit/polymarket_v1_bounded_sports_extract_v0.json)

所以当前“对称性”没有可计算样本：没有同一比赛、同一 outcome、同一 as-of cutoff 下的模型概率与可执行双边盘口。相关性、lead-lag、mispricing、predictive disagreement 和 alpha 均未被验证。

## SAF public repository 边界

[WeKruit/SAF](https://github.com/WeKruit/SAF) 是 public repository。仓库提交代码、contracts、tests、evidence、registries，以及不含凭证/敏感信息的 manifests；本机 `var/raw` 约 37GB 的 raw bytes 不提交。该边界由 [`.gitignore`](../../.gitignore) 中只放行 `var/raw/manifests/**`、继续排除 `var/raw` 数据对象的规则执行。

所有上述结论持续受 NO-GO 约束：没有真实资金动作、没有 maker、没有 queue-fill 精确声称、没有 multi-venue live arbitrage、没有 live copy trading、没有 LLM hot path、没有 RL，也没有未登记回测。
