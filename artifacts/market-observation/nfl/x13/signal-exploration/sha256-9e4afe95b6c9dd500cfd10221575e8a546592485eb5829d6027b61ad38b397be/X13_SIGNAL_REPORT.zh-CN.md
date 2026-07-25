# X-13 NFL 信号探索报告

- 结果标签：`PRELIMINARY_SOURCE_TIME_ONLY`
- 分析状态：`POST_HOC_DISCOVERY_HOLDOUT_REQUIRED`
- 比赛数：20
- 证据边界：仅 source-time 事后发现；需要独立 holdout 验证。
- 不主张因果、实时延迟、可执行性或可交易性。
- 不主张 OFI、depth 或 execution 证据。

## 逐场冻结因子

- delay=0 审计单元：8104152 行；其中真实成交观测 64058 行、缺失或无效单元 8040094 行。
- 观察窗口固定为 `1/2/5/10/30/60` 秒；缺失单元显式保留且不 forward-fill。
- observation policy：`ACTUAL_ONLY_NO_FORWARD_FILL`。
- 逐场 factor table：10277 行；cross-game candidate ranking：63 行。
- 无独立结构字段：fumble、turnover_on_downs。
- 结构字段不可用：third_fourth_down_failure、two_minute_drive、fumble、pat、scoring_reversal、turnover_on_downs、two_point_conversion、recent_activity_band、drive_progression、recent_activity_band、timeouts_remaining、win_probability_band；未从 description 推断。

## Top 10 discovery contrasts

| rank | factor contrast | effect | 95% CI | BH q | Kalshi | Polymarket | venue sign | support |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | STATE / distance_band / LONG_GT_10 vs ALL_OTHER_LEVELS | -0.004910614051 | [-0.008172017688, -0.00176173873] | 0.08639136086 | -0.00435423824 | -0.005910223105 | True | True |
| 2 | MARKET_REGIME / pre_event_probability_band / MID vs ALL_OTHER_LEVELS | 0.005217373834 | [0.001098770187, 0.009812816993] | 0.1058294171 | 0.01335345512 | 0.00359751995 | True | True |
| 3 | STATE / distance_band / STANDARD_8_TO_10 vs ALL_OTHER_LEVELS | 0.002081008602 | [0.000101355902, 0.004197657927] | 0.3054551688 | 0.001776662731 | 0.002712011933 | True | True |
| 4 | MARKET_REGIME / pre_event_probability_band / LOW vs ALL_OTHER_LEVELS | -0.002250632922 | [-0.005368819103, 0.0004367589044] | 0.7545745425 | -0.001760608787 | -0.002334772424 | True | True |
| 5 | STATE / score_margin_band / MULTI_SCORE vs ALL_OTHER_LEVELS | -0.002695174342 | [-0.007747101433, 0.001112148953] | 0.9144058567 | -0.008186112499 | -0.001249439699 | True | True |
| 6 | STATE / quarter / Q4 vs ALL_OTHER_LEVELS | -0.002058397718 | [-0.005278121797, 0.001320005929] | 0.9144058567 | -0.004596264846 | -0.0007512978119 | True | True |
| 7 | MARKET_REGIME / pre_event_probability_band / HIGH vs ALL_OTHER_LEVELS | -0.001862159988 | [-0.00539944662, 0.002145421277] | 0.9144058567 | -0.001453221615 | -0.001176536648 | True | True |
| 8 | STATE / field_position_band / OWN_TERRITORY vs ALL_OTHER_LEVELS | -0.001242521906 | [-0.003987732664, 0.001407766573] | 0.9144058567 | -0.002034390609 | -0.001914173926 | True | True |
| 9 | COMBINATION / event_x_score_margin / routine|MULTI_SCORE vs ALL_OTHER_LEVELS | -0.001096188387 | [-0.003662881992, 0.00151083615] | 0.9144058567 | -0.001271825397 | -0.001154035167 | True | True |
| 10 | STATE / distance_band / MEDIUM_4_TO_7 vs ALL_OTHER_LEVELS | 0.0007546039532 | [-0.001871615389, 0.003668872161] | 0.9144058567 | 0.0002219151355 | 0.0004523307236 | True | True |
- 当前最强 discovery-only 对比为 distance > 10；独立 holdout reaction 尚未运行。
- touchdown、interception、field goal 与 sequence 均未形成可晋升的跨 venue 稳健候选。
- 冻结 holdout selection 不等于 holdout result；本报告未读取 holdout reaction。

## 冻结推断表

- Primary contrasts：3 行；CASE_ONLY=1、DESCRIPTIVE=2
- 10–60 秒路径：3 行；CASE_ONLY=3
- PRE-state moderators：15 行；CASE_ONLY=10、DESCRIPTIVE=5
- 配对 venue differences：3 行；CASE_ONLY=2、DESCRIPTIVE=1

## Primary contrasts

| comparison | estimate | 95% CI | BH q | label |
|---|---:|---:|---:|---|
| score_minus_routine | -0.005333305833 | [-0.009666647207, -0.001388852907] | 0.01849568434 | DESCRIPTIVE |
| score_minus_turnover | NA | [NA, NA] | NA | CASE_ONLY |
| turnover_minus_routine | NA | [NA, NA] | NA | DESCRIPTIVE |

本文件由 runner 确定性生成，并受 bundle manifest 与 checksums.sha256 约束。
