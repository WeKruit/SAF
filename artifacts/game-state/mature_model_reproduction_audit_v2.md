# 成熟公开模型复现审计 v2

- Owner: Team D2 + D3 + H
- Version: v2
- Due gate: W2 review
- Audit status: NFL no-spread V2 已完成治理复现；足球动态强度仍为开放项
- Updated: 2026-07-23 (America/Chicago)

本文按当前代码、协议、注册表和已登记 artifact 冻结事实边界。NFL
结论已经从“仅工程实现”更新为一项真实、受治理的官方 no-spread
booster 复现；该结果仍是 `PRELIMINARY`、`PIT_UNPROVEN`。足球则必须把
严格状态重放证据、历史 Dixon–Coles POC 和尚未登记的动态强度模型分开，
不能在三者之间迁移准确率或延迟。

本次治理没有兼容层：旧身份不会被改名、映射或别名化为新模型，旧结果
也不会被重新贴到新模型上。

## NFL：官方 fastrmodels no-spread V2

### 审计结论

SAF 已完成准确名称为
`fastrmodels regulation wp_model no-spread inference reproduction` 的
本地复现，并在冻结的 2021–2025 nflverse REG/POST regulation rows 上
完成经验评价。它加载官方 UBJSON booster，不重训、不拟合 calibrator、
不读取 spread，也不使用 market data。

这项结论只覆盖官方 no-spread booster 的 regulation inference。它不等于
完整 `nflfastR home_wp` 管线复现；后者还包含 OT、PAT、kickoff、
administrative row 和终场覆盖等 helper 行为。

### 冻结来源、资产与运行时

- fastrmodels feature/training specification commit:
  `75c7b68bc49535370236c38c9826265da075bd71`
- nflfastR feature helper commit:
  `ead5e2f9641490f692d923c04835bd3b90275b4e`
- official model archive tag commit:
  `9f2495fdb4943087ca663d96706eb5df7973aff4`
- official GitHub release asset ID: `253928623`
- asset URL:
  `https://github.com/nflverse/fastrmodels/releases/download/model_archive/wp_model.ubj`
- asset length: `106951` bytes
- asset SHA-256:
  `sha256:ff58c98d22e3e36f50f6e82f8e68d3ed0920101132a50d30bb70291df33c0a4c`
- static manifest SHA-256:
  `sha256:080d98f34495fe59a532b7c24e17536f471700e92ac8415b682234d7241fe3cb`
- schema fingerprint:
  `sha256:0032e7efd41481e00519a018d5c572de559b191f790eebafcdaf1307e0942987`
- dataset ID: `DS-NFL-FASTRMODELS`
- license: MIT；再分发必须保留上游版权与许可证声明
- runtime: `xgboost==3.3.0`
- booster shape: 11 个有序特征，65 个 boosted rounds

官方冻结 booster 的上游训练边界是 2001–2020。SAF 没有重新取得、
再分发或重拟合该训练集；本次 2021–2025 评价严格位于该训练边界之后。

### V1 无效、append-only V2 修订与永久退役

X-11 的 base registration hash 是
`sha256:1706e20201346560f38b4bf1ab3f040c8318f871d809eeff03666827b1b5ec4e`。
Team H 于 `2026-07-23T23:49:01Z` 追加 sequence 1，登记
`REPRO-X11-NFL-FASTRMODELS-V1`；该 amendment hash 是
`sha256:ad594d2aa06ff7ecc99ba4389d53c2973f1aba8bc922bba45a7c0cedc3ed6177`。

V1 随后在任何评价发生前被判定无效：它只看
`game_seconds_remaining=1800`，没有用 `qtr` 区分 Q2 结束与 Q3 开始，
因而错误拒绝了两个有效的 2021 Q2 halftime rows。正确语义是：

- `qtr <= 2`:
  `half_seconds_remaining = game_seconds_remaining - 1800`
- `qtr >= 3`:
  `half_seconds_remaining = game_seconds_remaining`

V1 没有结果。Team H 于 `2026-07-24T00:41:31Z` 追加 sequence 2，
以 `supersede_reproduction` 原子取代 V1；prior hash 保持为 V1
amendment hash，新 amendment hash 为
`sha256:0dcd4a1a62c7790967023b2383a2cb93eaf35b25e3e4d64baabe8decb8f45960`。
V2 使用全新的 reproduction、scope、model 和 protocol 身份：

- reproduction: `REPRO-X11-NFL-FASTRMODELS-V2`
- scope: `team_h_nfl_fastrmodels_reproduction_v2`
- model: `MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1`
- protocol: `registries/protocols/x11_fastrmodels_no_spread_v1.json`

effective registry 将 V1 scope 的 `authorized` 永久置为 `false`。治理代码
拒绝用通用 `authorize_scopes` 重新授权 reproduction scope，也拒绝 V2
复用 V1 的 reproduction ID、scope 或 model identity。因此 V1 只保留在
append-only 历史链中，不能再次产出结果；不存在 V1→V2 兼容别名。

V2 评价于 `2026-07-24T01:06:45Z` 开始，严格晚于 sequence 2。
Team H 于 `2026-07-24T01:25:04Z` 才追加 sequence 3 的结果引用；
该 amendment hash 是
`sha256:ee5a0ae12f8de8e9a3fe093193a4d730d8281c4a82ff631ae245cf71680879cc`。
顺序因此是“冻结 V2 → 评价 → 追加结果”，没有评价后补登记。

### 原子绑定

| 绑定 | 冻结值 |
|---|---|
| model | `MODEL-NFL-FASTRMODELS-NO-SPREAD-CLOCK-V1` / `v1` |
| model record SHA-256 | `sha256:df098fb7c050669f445bf7115c4575d3409fa591e57ea7eddff511d84f2cf3d3` |
| datasets | `DS-NFL-FASTRMODELS`, `DS-NFLVERSE` |
| code paths | `src/prediction_market/models/nfl.py`, `src/prediction_market/models/nfl_fastrmodels.py` |
| code SHA-256 | `sha256:3c1b92679df15dd6a8bce1d4c4bddbcf5c81944f59b3fe85ded7e0e315161e75` |
| data SHA-256 | `sha256:6342fceb33fba8b7f2f3b601f85e11a8e201898419013d0e1a46f9f66623fbc4` |
| protocol SHA-256 | `sha256:3d75a366ed6e9627d50b4831ace348c63890160e2ead8241fddb0ce0fb917bdd` |
| reproduction spec SHA-256 | `sha256:82d81413d3fe851003b64dd014d45a83fcb881ce04487134596fe342dfa4e1d1` |
| pre-evaluation registration head | `sha256:0dcd4a1a62c7790967023b2383a2cb93eaf35b25e3e4d64baabe8decb8f45960` |
| evaluation runner SHA-256 | `sha256:ddd107a17eb0cc8dc15238f399a6e32a223851eb9fcfd1478adf012eecd00898` |
| registered result/metrics SHA-256 | `sha256:a5d0c7e984229909eda06f602d83ac86eca202c1b1cdd88ae159d522bab1bfe0` |

model registry 同时把 `pit_feature_contract` 与
`parameter_config_sha256` 绑定到上述 protocol SHA-256。结果引用中的
code、data、datasets、model 和 registration head 与 sequence 2 完全
一致；任何替换都会破坏 registration/result validation。

### 模型与评价协议

官方有序特征为：

1. `receive_2h_ko`
2. `home`
3. `half_seconds_remaining`
4. `game_seconds_remaining`
5. `Diff_Time_Ratio`
6. `score_differential`
7. `down`
8. `ydstogo`
9. `yardline_100`
10. `posteam_timeouts_remaining`
11. `defteam_timeouts_remaining`

其中：

`Diff_Time_Ratio = score_differential /
exp(-4 * ((3600 - game_seconds_remaining) / 3600))`

booster 输出 possession-team win probability。主队持球时
`p_home=p_posteam`，客队持球时 `p_home=1-p_posteam`。协议只接受
2021–2025 REG/POST、`qtr in {1,2,3,4}`、参赛双方身份一致且所需字段
非空的 rows；排序以 `order_sequence` 为主，并用 frozen raw record
ordinal 解决重复顺序而不丢行。binary label 只来自最终比赛结果。

冻结 census 为：

| Season | Raw rows | Games | Eligible rows | Eligible non-tie games |
|---|---:|---:|---:|---:|
| 2021 | 49,922 | 285 | 41,342 | 284 |
| 2022 | 49,434 | 284 | 40,937 | 282 |
| 2023 | 49,665 | 285 | 41,698 | 285 |
| 2024 | 49,492 | 285 | 41,269 | 285 |
| 2025 | 48,771 | 285 | 40,361 | 284 |
| Total | 247,284 | 1,424 | 205,607 | 1,420 |

4 场平局的 595 rows 被单独报告并排除出 binary metrics：
`2021_10_DET_PIT`、`2022_01_IND_HOU`、`2022_13_WAS_NYG` 和
`2025_04_GB_DAL`。

### 评价结果与 bootstrap

aggregate point estimates 与 95% game-cluster bootstrap CI 为：

| Metric | Estimate | 95% CI |
|---|---:|---:|
| Row-micro Brier | `0.16628494129309768` | `[0.15895365711008524, 0.17231821510203402]` |
| Row-micro log loss | `0.49142648387256166` | `[0.472089417505973, 0.5074931712586502]` |
| Calibration slope | `0.9876121419583983` | `[0.9123315338560851, 1.084643143892938]` |
| Calibration intercept | `-0.047990070224865344` | `[-0.1538096142333979, 0.07853486288879798]` |
| Game-macro Brier | `0.16550869435681267` | `[0.1583121939900199, 0.17179176587590594]` |
| Game-macro log loss | `0.48908050909729683` | `[0.4708145414742444, 0.5063001571369761]` |

每季 point estimates 也来自同一冻结运行：

| Season | Row Brier | Row log loss | Slope | Intercept | Game Brier | Game log loss |
|---|---:|---:|---:|---:|---:|---:|
| 2021 | `0.15615792457106023` | `0.4602850854654701` | `1.1115665982018994` | `-0.24741402744563404` | `0.15672033896908513` | `0.4616870233727524` |
| 2022 | `0.1887806301815843` | `0.5562342313533986` | `0.7291690671357458` | `0.08522694723728141` | `0.18758533129988508` | `0.5520805601522263` |
| 2023 | `0.1585473715972639` | `0.47317159999197067` | `1.0849656338021723` | `-0.0036918626230402754` | `0.1576648662789643` | `0.47081199294205833` |
| 2024 | `0.1569034475166553` | `0.46307638990007305` | `1.1679020075144204` | `-0.05940079635539684` | `0.1565183768771834` | `0.46199482641876416` |
| 2025 | `0.17142780586858813` | `0.5054396428757195` | `0.9353190116274308` | `-0.046547384679639946` | `0.16926930279360602` | `0.49943150281527304` |

bootstrap unit 是 `game_id`，confidence level 为 `0.95`，seed 为
`20260723`；aggregate 与各 season 都请求 200 次并得到 200 次有效
resamples，最低有效门槛为 100。没有拟合任何 calibration model。

### upstream `home_wp` 诊断边界

本地输出与 source `home_wp` 的 absolute-delta 诊断覆盖 206,202 rows，
缺失 0；p50、p95、p99 都是 `0.0`，mean absolute delta 是
`1.2328149033261145e-10`，maximum 是
`1.1920928955078125e-7`。

这是 near-exact reproduction diagnostic，说明冻结 rows 上的本地
booster path 与上游预计算列数值近乎一致。它不是独立准确率 oracle：
`home_wp` 本身是预测列，其精确 transformer/model lineage 没有作为
独立真值冻结；它既不进入特征，也不充当 label。准确率只以最终比赛
结果为 label。

### 确定性与 full-path latency

完整评价运行了两次。两次 metrics bytes 均为 15,672 bytes，逐字节
一致，digest 都是
`sha256:a5d0c7e984229909eda06f602d83ac86eca202c1b1cdd88ae159d522bab1bfe0`；
latency bytes 被明确排除在确定性比较之外。

当前 latency 的 `full_path` 边界包含：

`normalized event construction → state/event transition → official feature
projection → preloaded official booster → probability output validation`

它明确排除 I/O、network、registry loading 和 market join。50 次 warmup
后测量 1,000 次：

- minimum: `183000 ns`
- p50: `468896 ns`
- p95: `1611483 ns`
- p99: `4412516 ns`
- maximum: `10546042 ns`
- mean: `628108.34 ns`
- mean-derived throughput: `1592.082028396566 transitions/s`

这些数字只代表当前本地冻结 full-path boundary，不是网络到端、market
join 或生产 SLA。

### 证据边界

- result label: `PRELIMINARY`
- PIT status: `PIT_UNPROVEN`
- observation mode: `offline_reconstruction_not_live_PIT`
- calibrator: `none_fitted`
- market data used: `false`
- prediction-market alignment: `none`
- prediction-market symmetry: `not_evaluated`
- alpha evidence: `none`

因此“官方 no-spread booster 已受治理并复现”成立；“live PIT 已证明”、
“已完成 market 对齐”或“存在交易 alpha”均不成立。spread booster
不在 V2 scope 内，也没有通过旧 identity 或兼容层进入本次结果。

Primary sources:

- https://github.com/nflverse/fastrmodels/blob/75c7b68bc49535370236c38c9826265da075bd71/data-raw/MODELS.R
- https://github.com/nflverse/nflfastR/blob/ead5e2f9641490f692d923c04835bd3b90275b4e/R/helper_add_ep_wp.R
- https://github.com/nflverse/fastrmodels/releases/tag/model_archive

## 足球：状态重放、历史 POC 与动态模型必须分层

### 严格状态重放证据

`soccer_real_replay_validation_v2.json` 是 reducer-v3 的
`ENGINEERING_VALIDATION`，结论为
`DETERMINISTIC_STRICT_PARTIAL_REPLAY`。它不是模型证据。

冻结 StatsBomb Premier League 2015/16 source 含 380 场比赛和
1,313,773 个 native events；381 个 manifest/object 均先经 static-store
验证。两次扫描的 run objects 完全相同，所有 1,313,773 rows 都被
adapted；其中 1,090,963 次进入 reducer attempt，1,090,826 次成功
reduce。

严格门禁在每场首个 state-contract violation 处停止该场状态推进：
137 场 fail closed，包括 135 个首发 clock regression、1 个 possession
regression 和 1 个 period end 后 active event；其后的 222,810 rows 只做
源异常清点，不能再改变 state。其余 243 场到达 source end，但没有一场
具备 reducer-v3 所要求的显式终场证据，所以 `finished_games=0`，
243 场均为 `finalization_unproven`。

同源 final-score metadata 与 event export 在 380/380 场上一致，只能
称同一冻结来源的内部一致性检查，不是独立 score oracle。该 artifact
明确没有 predictive accuracy、model inference latency、live PIT、
market data 或 alpha 证据。其 reducer-only p50/p95/p99
`37917/52750/94292 ns` 也不能写成模型延迟。

### 已登记的历史 Dixon–Coles v0 是另一项 POC

`x12_real_data_poc_v0.json` 是
`MODEL-SOCCER-DIXON-COLES` 的赛前 1X2 expanding-window POC，并另含
旧 `MODEL-SOCCER-FIVE-MINUTE-TRANSITION` 静态 head。它不是当前动态
强度模型，更不是 Maia 官方实现。

该 v0 的 1X2 部分有 280 个 held-out matches：

- multiclass Brier: `0.6333086250422754`
- multiclass log loss: `1.0566358383349692`
- model-minus-simple-baseline Brier delta:
  `-0.024442623053046564`，95% CI
  `[-0.06037747745710088, 0.013990344957441411]`
- model-minus-simple-baseline log-loss delta:
  `-0.029168652881823798`，95% CI
  `[-0.08353985226881108, 0.028878288366651978]`

两个 paired CI 都跨 0。v0 仍是 `POC_NO_PIT_MARKET_PRIOR`、
`PRELIMINARY`、`POC_ONLY`；它不能证明相对简单 comparator 的稳定改进。
旧静态五分钟 head 的 5,040 observations、Brier
`0.24552780282114847` 和 log loss `0.49219732711024167` 只属于该旧
model identity，不能迁移给当前动态模型。

### 当前动态强度模型只有工程代码

论文定义的是带动态回归量的 Cox process，即 doubly stochastic Poisson
event intensity；不是 Cox proportional-hazards survival model。没有可
冻结的论文官方代码仓库，因此 SAF 不声称“官方 Maia 复现”。

当前实现的准确名称是：

`Maia-family dynamic-covariate adaptation with frozen Dixon-Coles
base-rate offset`

实现的 300-second competing-Poisson head 使用冻结 Dixon–Coles
goals-per-90 base-rate offset，以及 cutoff 时已知的：

- `second_half`
- 主客 `score_difference`
- 主客去重后的 `dismissal_difference`（Red Card / Second Yellow）

它包含 50% 完整日期组 base/dynamic fit、25% temperature calibration、
25% final test 的工程协议和相应测试，但当前治理状态没有变化：

- `registries/model_registry.csv` 没有
  `MODEL-SOCCER-DYNAMIC-INTENSITY` row；
- X-12 仍无 amendments，`results_ref=[]`；
- X-12 的现有授权 scope 仍绑定
  `MODEL-SOCCER-DIXON-COLES` 与
  `MODEL-SOCCER-FIVE-MINUTE-TRANSITION`；
- X-12 registration locks 仍 unresolved；
- 没有已登记的动态模型真实 empirical result；
- 没有当前动态模型 latency artifact。

`model_latency_v0.json` 中的足球数字绑定旧静态五分钟 model identity，
已不属于当前动态 path。动态模型的 latency harness 会在缺少“已登记且
已校准的动态 empirical evidence”时 fail closed，所以不能报告旧速度，
也不能用 synthetic fitted instance 代替当前治理结果。

StatsBomb commit 仍冻结为
`b0bc9f22dd77c206ddedc1d742893b3bbe64baec`，license status 仍为
`research_only`；offline archive 不能证明 live PIT availability，也没有
PIT market prior。

Primary sources:

- https://arxiv.org/abs/2312.04338v1
- https://doi.org/10.1016/j.ijforecast.2025.10.006
- https://github.com/hudl/open-data/commit/b0bc9f22dd77c206ddedc1d742893b3bbe64baec

## 治理结论

NFL 的最短治理链已经闭合：V1 历史保留但永久退役；V2 在评价前通过
append-only amendment 原子绑定新 model identity、官方 asset、code、
data、protocol 和 reproduction spec；评价后才追加
`PRELIMINARY/PIT_UNPROVEN` result。这里没有 registry row 覆盖、历史
重写或兼容层。

足球仍是开放项。状态 reducer-v3 的严格重放证据可以独立保留，但不能
为模型准确率背书；Dixon–Coles v0 也必须继续按原 identity 只读保留。
若要产生动态强度 empirical result，Team H 仍须先用 append-only
reproduction amendment 登记新的 model row、scope、code/data/protocol/
spec hashes，且评价时间必须严格晚于该 amendment。正式 promotion 在
X-12 中仍是 permanent NO-GO。

在此之前，足球动态强度只能称 engineering implementation；不得称官方
Maia reproduction，不得引用旧 Dixon–Coles/静态五分钟准确率，也不得
报告旧 latency 为当前速度。
