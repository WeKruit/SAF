# NFL Sports Factor Lab Workbench

这是 X-13 的权威研究工作台，不是只读取最终结论的展示页。Notebook 不下载数据，
但会逐阶段调用与生产物化完全相同的 fail-closed Python 函数，并展示真实
DataFrame、行数、排除原因、外键与 hash：

```text
S3/local immutable input → hash验证 → sports清理 → replay/episode
→ PIT context → market清理 → G/M时间区间 → 纯时间轴
→ reaction association → factor observation → 单场结果 → 20场汇总
```

正式计算由 Python module 完成；Notebook 只负责按顺序调用和呈现，不能在 cell
里另写一套清理或统计逻辑。

## 从这里开始

总入口是
[`NFL_Factor_Lab_Master.ipynb`](NFL_Factor_Lab_Master.ipynb)。它像一份 Kaggle
solution 一样按故事顺序展示研究边界、20 场覆盖、方法卡、完整 factor universe、
具体 play、实际市场路径、逐场结果、跨场统计、候选、数据缺口、专家审核和
Dashboard deep link。过去的 `00–12` 已移到 `drilldowns/`，只保留为专题参考材料；
它们不是入口，也不再要求读者靠自己把 13 份页面拼成全貌。

Master 从 V2 run index 开始，随后按 Section 0–16 展示：

- source/stage manifest、S3/local 状态和 SHA-256；
- raw PBP、participation 与 game binding；
- sports row disposition、canonical events、episodes 与 stat ledger；
- PIT team/player context；
- contract inventory、market cleaning audit 与 observations；
- Layer G、Layer M、纯 source-time timeline；
- association attrition、actual reaction paths；
- factor registry、逐行 factor observations 与 exclusion audit；
- 当前单场结果、20 场跨场结果、方法卡与专家审核队列。

每张表都直接展示 path、shape、columns、head、grain 和 semantic hash。PyArrow
负责 Parquet metadata 与过滤读取，Pandas 负责人类可读展示，DuckDB 负责只读查询。
单场模式只读取该场 bundle；20 场模式只组合 20 个已验证 single-game bundle，
不会重扫 48.6m association universe。

总入口严格依赖 `registries/factors/nfl_factor_lab_run_index_v2.json`。该索引必须
精确绑定冻结的 20 场、17×20 个单场 stage 和 5 个跨场 stage；如果索引或正式产物
尚未发布，Notebook 会明确失败，不展示旧 V1 结果、空表或假结果。

默认启动采用 `SHALLOW_COMMIT_VERIFIED`：验证 canonical run-index、21 个 bundle
manifest 的 SHA-256/大小、路径边界、治理字段，以及 20 场/345 stage 的引用和
byte-length 一致性；不会为了打开 Notebook 再读取和 hash 约 3.58 GiB 的全部
Parquet。Section 1 会明确显示 `stage_content_verification=NOT_READ` 和
`latest_deep_batch_status=NOT_RECORDED_IN_RUN_INDEX`，因此浅验证不会冒充内容深验。

如需重新执行全部 stage 的 SHA-256、semantic rows 和跨表约束验证，显式运行：

```bash
SAF_FACTOR_LAB_DEEP_VERIFY=1 ./notebooks/nfl-factor-lab/open_factor_lab.command
```

S3 表只显示 bucket/prefix 配置，状态固定为
`CONFIGURED_NOT_REMOTE_VERIFIED`；Notebook 不访问网络，也不会把“配置存在”误写成
远端对象已经重新验证。默认单场仍为 `2025_14_DAL_DET`，各 Section 只读取该场
已发布 bundle 的 bounded slices。

本机启动：

```bash
./notebooks/nfl-factor-lab/open_factor_lab.command
```

也可以双击 macOS Finder 中的 `open_factor_lab.command`。启动后 Jupyter Lab 直接打开
总 Notebook。

运行边界：

- 输入必须先通过 manifest/hash/license/experiment lock 校验。
- 所有页面均为 `PRELIMINARY_SOURCE_TIME_ONLY`，不作因果、真实 latency、执行性或
  可交易 alpha 声明。
- `12_candidate_decision_and_holdout` 默认不会运行 holdout。只有存在人工审核完成且
  已冻结的 `ShortlistLockV1` 时才调用 locked holdout 接口。
- Dashboard case 链接只使用稳定的 game、episode、logical market、delay 与 horizon
  身份。
- 源 Notebook 保持无输出、无 execution count；用户运行后的输出留在本地工作副本。

报告源固定面向 Quarto 1.9.38。当前机器若没有固定版本 Quarto/Jupyter runtime，
本目录只交付确定性的 Notebook/QMD/YAML 源和结构验证，不生成伪造的 rendered output。

阅读只从 Master 开始；`drilldowns/` 是旧的 immutable 专题参考，不再作为维护入口。
Holdout 仍关闭，直到专家审核完成并产生正式 shortlist lock。

固定的 Python 研究运行时由 `pyproject.toml` 的 `research` dependency group 与
`uv.lock` 提供：

```bash
uv sync --locked --group research
uv run --locked --group research jupyter lab notebooks/nfl-factor-lab
```

Quarto 不作为 Python wheel 安装。下载 method catalog 锁定的官方
`quarto-1.9.38-macos.tar.gz` 后，必须先核对 SHA-256
`47089a5020cfb41981ba0d4b46e110edfa608722aea45ef248e14efba6d6b18a`，
再通过 `SAF_QUARTO=/absolute/path/to/quarto-1.9.38/bin/quarto` 调用正式
`render_quarto_reports(...)` 接口。不得使用其他版本或跳过 checksum。
