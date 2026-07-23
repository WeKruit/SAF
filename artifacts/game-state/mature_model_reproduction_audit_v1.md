# 成熟公开模型复现审计 v1

- Owner: Team D2 + D3 + H
- Version: v1
- Due gate: W2 review
- Status: ENGINEERING_VALIDATION_ONLY
- Updated: 2026-07-23

本文只冻结公开模型定义、实现边界和当前 SAF 差距，不把 README、
上游预计算预测或未登记运行写成经验结果。

## NFL：fastrmodels regulation win probability

冻结来源：

- fastrmodels source commit:
  `75c7b68bc49535370236c38c9826265da075bd71`
- nflfastR feature helper commit:
  `ead5e2f9641490f692d923c04835bd3b90275b4e`
- official `wp_model.ubj`:
  `sha256:ff58c98d22e3e36f50f6e82f8e68d3ed0920101132a50d30bb70291df33c0a4c`,
  106,951 bytes
- official `wp_model_spread.ubj`:
  `sha256:a8efe70cf64f459187ef06ebaae08b7a1012661b3446e1336a0a5ece2ba86322`,
  1,219,887 bytes
- License: MIT; redistribution must retain the upstream copyright and
  license notice.

SAF 已经实现并测试从 `NFLGameState` 到官方有序 11/12 维向量的纯
feature seam，包括：

- `receive_2h_ko`
- possession-relative score、timeouts 和 spread
- `half_seconds_remaining`
- `spread_time`
- `Diff_Time_Ratio`
- down、distance、field position

该 seam 现在不再接受调用方自报 cutoff：它复验同一份 normalized
`EventEnvelope`、两个 raw parents、canonical game/event/sequence、
source play、season 与 raw object hash，并只使用该 envelope 的
`source_at`。second-half receiver 与 spread 的观测时间不得晚于这个
cutoff；所选 no-spread/spread UBJ 的精确 SHA-256 也进入 feature hash。
dataset/source manifest digest 作为声明进入 hash，但该 seam 不冒充已
复验 manifest 内容。任何 post-cutoff 外部输入、lineage 脱节或 game
mismatch 都会失败。

尚未形成官方模型经验结果。原因：

1. 当前 X-11 的 `home_wp` 是上游发布列 comparator，不是本地官方
   booster 推理。
2. 官方冻结权重的训练语料含 2001–2020；它不能作为 2020
   out-of-sample comparator。
3. spread 没有 point-in-time observation manifest，spread variant
   只能保持 PIT_UNPROVEN。
4. 官方 UBJ、代码、数据和评价窗尚未在 Team H reproduction
   amendment 中原子锁定。

Team H 的新 reproduction contract 已经预留两个独立的新模型身份：
`MODEL-NFL-FASTRMODELS-NO-SPREAD` 与
`MODEL-NFL-FASTRMODELS-SPREAD`。它不会把旧的
`MODEL-NFL-NFLFASTR-COMPARATOR` 重新命名成官方本地 inference；
在新 model registry rows、真实 UBJ objects 和 hashes 就绪前，登记会
fail closed。

准确命名：

- 本地 booster 运行应称
  `fastrmodels regulation wp_model inference reproduction`。
- 除非实现 nflfastR 的 OT、PAT、kickoff、admin-row 和终场覆盖，
  不得称为完整 `nflfastR home_wp` 复现。

Primary sources:

- https://github.com/nflverse/fastrmodels/blob/75c7b68bc49535370236c38c9826265da075bd71/data-raw/MODELS.R
- https://github.com/nflverse/nflfastR/blob/ead5e2f9641490f692d923c04835bd3b90275b4e/R/helper_add_ep_wp.R
- https://github.com/nflverse/fastrmodels/releases/tag/model_archive

## 足球：动态事件强度

冻结来源：

- Maia et al., arXiv `2312.04338v1`
- StatsBomb/Hudl open-data commit:
  `b0bc9f22dd77c206ddedc1d742893b3bbe64baec`
- StatsBomb data use remains `research_only`; it is not an OSI/SPDX
  open-source license.

公开论文定义的是带动态回归量的 Cox process / doubly stochastic
Poisson event intensity，不是 Cox proportional-hazards survival
model。论文没有可冻结的官方代码仓库，因此 SAF 不能声称复现了官方
实现。

SAF 当前实现的准确名称是：

`Maia-family dynamic-covariate adaptation with frozen Dixon-Coles
base-rate offset`

它只使用当前已知的：

- 上/下半场；
- 主客比分差；
- 主客罚下人数差（Red Card / Second Yellow）；
- 在最早 50% 完整比赛日期组中冻结的赛前 Dixon-Coles 主客进球基率。

它输出 frozen-state 300-second competing-Poisson distribution：
`home_goal / away_goal / no_goal`。它不是完整 Maia G4S5R 复现，
不包含球员价值、未来红牌联合过程、补时模型或因果解释。

核心接口已经通过 synthetic TDD；真实 X-12 运行仍必须先满足：

1. dismissal timeline、score timeline 和 state feature hash 均可复算；
2. 同一比赛不能同时进入 fit 与 held-out fold；
3. period 3–5、未来 card sequence 和不足完整 300 秒的状态 fail
   closed；
4. Team H 先锁定 exact model row、code/data hash 和 reproduction spec；
5. transition 使用完整日期组的 50% base fit / 25% temperature
   calibration / 25% final test，最终概率经过单一 multiclass
   temperature；
6. calibrated transition 的 OVR calibration、game-cluster CI 与静态
   comparator paired delta 完整报告。

代码路径现在已经实现第 1、2、3、5、6 项的工程门禁：动态结果与零动态
系数的赛前 Dixon-Coles comparator 使用同一 game-cluster draw 计算
paired CI，并报告 calibrated OVR slope/intercept；base、calibration 与
final test 的比赛和日期组互斥。第 4 项仍未完成，因此没有运行或发布
新的 empirical X-12 evidence。历史 1X2 expanding Dixon-Coles 指标仍
只属于 `x12_real_data_poc_v0.json`；transition v1 只按 path+hash 引用
该 artifact，不重算也不把它迁移成 v1 结果。

旧 `model_latency_v0.json` 测的是被替换的静态五分钟 head；当前 v1
validator 会按 report version 与 model inventory 拒绝整个旧 artifact。
新 latency seam 已经使用真实
StatsBomb EventEnvelope ID，执行
`reducer → feature → dynamic intensity → temperature calibration →
ModelOutputV1`；但在新的
动态系数证据和 registry binding 产生前，不生成新的速度数字，也不把
旧准确率重新贴到新模型上。

离线 X-12 evaluation snapshot 目前只保存 source object/timeline
lineage，没有逐 cutoff 的真实 reducer EventEnvelope ID。证据构造器
因此不再合成 `state_event_id`，而是明确输出
`contract_output=null` 和 open gate；这条 empirical 路径在真实事件
绑定完成前不能用于 prediction-market as-of join。

Primary sources:

- https://arxiv.org/abs/2312.04338v1
- https://doi.org/10.1016/j.ijforecast.2025.10.006
- https://github.com/hudl/open-data/commit/b0bc9f22dd77c206ddedc1d742893b3bbe64baec

## 治理结论

X-11/X-12 原有 scope 只绑定旧 model IDs；直接覆盖 registry 行会重写
历史身份，不能证明模型在 evaluation 前已冻结。正确最短路径是 Team H
追加一个 append-only `register_reproduction` amendment：

- 保留 base registration hash 和旧 scope；
- 绑定新 scope、dataset IDs、model ID/version/record hash；
- 绑定 code hash、data hash 和 reproduction spec hash；
- evaluation 时间必须严格晚于 amendment；
- 任何 hash、model version、license 或 scope 不一致都 fail closed。

此外，reproduction amendment、evaluation 和 result 都不能填写未来
时间；代码 hash 会从固定源码对象重新计算，data hash 会与 dataset
registry manifest 复算。X-11 必须同时绑定两个新 fastrmodels 变体；
X-12 必须绑定 Dixon-Coles base-rate 与新的
`MODEL-SOCCER-DYNAMIC-INTENSITY`，不能用任意 legacy model 集合替代。

在上述 gate 完成前，新模块的输出只能称 engineering validation，
不能写成 X-11/X-12 empirical result。
