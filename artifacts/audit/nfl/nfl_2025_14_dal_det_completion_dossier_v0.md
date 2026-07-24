# NFL 单场完成档案：2025_14_DAL_DET

- Owner：D2+C+H
- Version：v0
- Due gate：2026-08-05_W2_review
- Status：`PRELIMINARY_RESEARCH_ONLY`；不是交易、alpha 或执行结论。

## 本次完成的单场范围

本档案只覆盖 Dallas Cowboys at Detroit Lions（2025-12-04），不扩展到第二场比赛或其他赛事。

| 层 | 已验证的事实 | 结论 |
|---|---|---|
| NFL game state | 受治理的 nflverse 2025 原始 Parquet 通过 immutable manifest 校验；192 行、191 transitions；两次 replay 的 trace hash 一致。 | `PASS`：可复现的离线 state replay。 |
| Polymarket | 已绑定唯一 event、condition 与两个 outcome；3,922 笔历史成交原件逐对象校验。 | `PASS`：身份和历史成交库存；不是 L1/L2。 |
| Kalshi | 已绑定 Detroit 与 Dallas 两个 winner ticker；73,844 笔历史成交、两个连续 1 分钟 candle 序列逐对象校验。 | `PASS`：身份、成交及 interval-end candle 库存；不是历史 L2。 |
| 原件完整性 | nflverse 1 + Polymarket 2 + Kalshi 78，共 81 个 manifest/object 对已验证。 | `PASS`：篡改、缺对象、重复 Kalshi trade ID、非连续 candle 都会 fail closed。 |
| Game Book | 官方 PDF 明示仅供媒体报道，其他用途需 NFL 书面许可；项目没有该许可，也未注册为可用数据集。 | `BLOCK`：不存入 raw、不喂给 LLM、不作为正式验证来源。 |
| 时间合并 | NFL source event time、Polymarket trade time、Kalshi trade/candle time 已合并为 24,355 行的确定性 source-time timeline；可供人工审阅和可视化。 | `PASS`：纯时间轴合并。`BLOCK`：不能把相邻时间解释为 latency、因果反应、overreaction、可执行 bid/ask 或 alpha。 |

机器可读的逐对象审计见 [evidence audit v0](nfl_2025_14_dal_det_evidence_audit_v0.json)，合并时间轴见 [source timeline v0](nfl_2025_14_dal_det_source_timeline_v0.json)。输入 state trace 为 [NFL replay](../../game-state/nfl/nfl_2025_14_dal_det_state_replay_v0.json)，venue mapping 为 [market mapping](../../market-observation/nfl/nfl_2025_14_dal_det_market_mapping_v0.json)。

## 明确禁止的推论

- 不把交易成交价称为当时可成交的 bid 或 ask。
- 不把 1 分钟 Kalshi candle 的 end timestamp 当成事件发生时的 quote。
- 不从 source timestamps 推断本地接收 latency 或“谁先反应”；时间轴内的 0/1/2/5/10/30/60 秒仅是可调的假设情景，不是 p50/p95/p99 测量值。
- 不输出模型准确率、市场对称性、概率路径、barrier-hit、执行收益或 alpha。

## 这场切片的停止条件

单场的可取得数据、原件校验、state replay、venue identity、时间语义审计和不可得字段均已闭环。后续工作必须有新的、独立授权，而不是在本切片上继续推断：

1. 可研究/存储 Game Book 的书面许可，或可替代的已注册独立官方数据源；
2. 对同一比赛同时覆盖 game event 与市场的 PIT source-time semantics 与 error bounds；
3. 若要研究可执行性，还需要连续 L2、双边 quote/depth、pause state、venue-rule snapshot 与本地 receive time。

在这些条件满足前，本场的正式结果资格保持 `false`。
