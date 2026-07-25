# X-13 Research Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private-first dashboard that lets a researcher replay 20 NFL games, query game events and market paths, and inspect hash-bound event × market association candidates without loading the 2.4 GB canonical batch into the browser.

**Architecture:** A Python exporter verifies the immutable X-13 batch and emits content-addressed presentation slices. Supabase S3 stores those immutable slices while a three-table Supabase Postgres schema indexes batches, games, and assets. A small Vinext site fetches private objects through a server-side S3 gateway and renders the 20-game index and single-game research workspace.

**Tech Stack:** Python 3.12, pytest, canonical JSON/SHA-256, boto3/Supabase S3, PostgreSQL SQL migration, Vinext/React/TypeScript, Vitest, Playwright.

---

## File structure

- `src/prediction_market/sports/nfl_x13_source_bundles.py`: deterministic per-dataset manifest bundles for X-13 provenance.
- `src/prediction_market/sports/nfl_x13_dashboard.py`: verified batch-to-presentation projection.
- `src/prediction_market/dashboard_store.py`: private S3 publication and byte/hash verification, independent of PMXT.
- `src/prediction_market/cli/publish_x13_dashboard.py`: export, upload, and publish-pointer CLI.
- `registries/dataset_manifest_bundles/`: content-addressed source bundle artifacts.
- `supabase/migrations/202607250001_x13_dashboard_index.sql`: three-table rebuildable index schema.
- `tests/sports/test_nfl_x13_source_bundles.py`: provenance bundle red/green tests.
- `tests/sports/test_nfl_x13_dashboard.py`: projection and tamper tests.
- `tests/test_dashboard_store.py`: S3 publication tests.
- `apps/saf-dashboard/`: dashboard site and server-side private-asset gateway.

### Task 1: Publish exact X-13 per-source manifest bundles

**Files:**
- Create: `src/prediction_market/sports/nfl_x13_source_bundles.py`
- Create: `tests/sports/test_nfl_x13_source_bundles.py`
- Create: `registries/dataset_manifest_bundles/sha256-40a886ddb9463720a5f54ad8e3a212216a9a0098ae2d17c8230b0d8f33ad2530.json`
- Create: `registries/dataset_manifest_bundles/sha256-7714b0132a89a1f8e812e5a8bc78a20e2fb6dfd5c1689c08cee9ec9c6dc376d8.json`
- Modify: `registries/dataset_registry.csv`
- Modify: `tests/test_research_registries.py`
- Modify: `tests/test_experiment_registry.py`

- [ ] **Step 1: Write failing bundle identity tests**

```python
def test_x13_source_bundles_are_exact_and_reopenable(program_root: Path) -> None:
    bundles = build_x13_source_manifest_bundles(
        program_root / "var/raw/capture-receipts/f08c3a175cc6a91c6eeec92de2cac745977b6f90aa53df4e59054ba8d1e6a53d.json"
    )
    assert bundles["DS-KALSHI-HISTORICAL"].artifact_sha256 == (
        "sha256:40a886ddb9463720a5f54ad8e3a212216a9a0098ae2d17c8230b0d8f33ad2530"
    )
    assert bundles["DS-POLYMARKET-PUBLIC"].artifact_sha256 == (
        "sha256:7714b0132a89a1f8e812e5a8bc78a20e2fb6dfd5c1689c08cee9ec9c6dc376d8"
    )
    assert len(bundles["DS-KALSHI-HISTORICAL"].raw_manifest_sha256s) == 9_345
    assert len(bundles["DS-POLYMARKET-PUBLIC"].raw_manifest_sha256s) == 840
```

- [ ] **Step 2: Run the test and confirm the feature is missing**

Run:

```bash
uv run pytest tests/sports/test_nfl_x13_source_bundles.py -q
```

Expected: collection fails because `nfl_x13_source_bundles` does not exist.

- [ ] **Step 3: Implement canonical bundle construction**

Define `X13SourceManifestBundleV1`, strict receipt parsing, exact dataset/version
filters, sorted unique manifest IDs, canonical JSON bytes, and artifact
SHA-256. Reject duplicate path/object identities, unknown datasets, altered
capture IDs, and mismatched expected counts.

- [ ] **Step 4: Materialize and reopen the two content-addressed artifacts**

Run:

```bash
uv run python -m prediction_market.sports.nfl_x13_source_bundles \
  --receipt /Users/wekruitclaw1/Desktop/prediction-market/var/raw/capture-receipts/f08c3a175cc6a91c6eeec92de2cac745977b6f90aa53df4e59054ba8d1e6a53d.json \
  --output registries/dataset_manifest_bundles
```

Expected: two files are created with the exact filenames declared above, then
reopened and byte-hash verified.

- [ ] **Step 5: Update research-only registry bindings**

Change only:

- `DS-POLYMARKET-PUBLIC`: `pending` → `research_only`, manifest →
  `sha256:7714...376d8`;
- `DS-KALSHI-HISTORICAL`: `pending` → `research_only`, manifest →
  `sha256:40a8...2530`;
- `DS-NFLVERSE-PARTICIPATION`: retain `research_only`, manifest →
  `sha256:c5b2...e291`.

Do not change Team I review IDs, due gates, or any status to GREEN/approved.

- [ ] **Step 6: Verify registry and authorization**

Run:

```bash
uv run pytest \
  tests/sports/test_nfl_x13_source_bundles.py \
  tests/test_research_registries.py \
  tests/test_x13_registry.py \
  tests/test_experiment_registry.py -q
```

Expected: all selected tests pass and `validate_result_ref` accepts the
research-only X-13 dataset bindings.

- [ ] **Step 7: Commit**

```bash
git add src/prediction_market/sports/nfl_x13_source_bundles.py \
  tests/sports/test_nfl_x13_source_bundles.py \
  registries/dataset_manifest_bundles \
  registries/dataset_registry.csv \
  tests/test_research_registries.py \
  tests/test_experiment_registry.py
git commit -m "feat: bind X-13 source manifest bundles"
```

### Task 2: Export bounded, hash-bound dashboard slices

**Files:**
- Create: `src/prediction_market/sports/nfl_x13_dashboard.py`
- Create: `tests/sports/test_nfl_x13_dashboard.py`

- [ ] **Step 1: Write failing projection tests**

```python
def test_export_dashboard_projection_is_bounded_and_deterministic(
    verified_batch: Path,
    tmp_path: Path,
) -> None:
    first = export_x13_dashboard(verified_batch, tmp_path / "first")
    second = export_x13_dashboard(verified_batch, tmp_path / "second")
    assert first.publish_manifest_sha256 == second.publish_manifest_sha256
    assert first.game_count == 20
    assert first.maximum_asset_bytes <= 8 * 1024 * 1024
    assert all(asset.source_sha256 for asset in first.assets)
```

Also add tests for batch tampering, missing embedded presentation JSON, duplicate
logical-market IDs, unsafe paths, observed/derived semantics, and absent
Polymarket BBO.

- [ ] **Step 2: Run the tests and confirm RED**

```bash
uv run pytest tests/sports/test_nfl_x13_dashboard.py -q
```

Expected: collection fails because `nfl_x13_dashboard` does not exist.

- [ ] **Step 3: Implement verified projection**

Implement:

```python
@dataclass(frozen=True, slots=True)
class DashboardAssetV1:
    role: str
    relative_path: str
    schema: str
    source_sha256: str
    object_sha256: str
    byte_length: int

def export_x13_dashboard(
    batch_root: str | Path,
    output_root: str | Path,
) -> DashboardExportV1:
    ...
```

The exporter must:

1. call `verify_published_batch`;
2. extract `nfl_x13_html_presentation_v1` from each existing standalone HTML;
3. emit catalog, core, contracts, one file per logical market, associations,
   and a publish manifest;
4. use canonical JSON plus one trailing newline;
5. publish only content-addressed paths;
6. reject any asset above 8 MiB;
7. atomically rename the completed staging directory.

- [ ] **Step 4: Run projection tests GREEN**

```bash
uv run pytest tests/sports/test_nfl_x13_dashboard.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Export the formal batch twice**

```bash
uv run python -m prediction_market.sports.nfl_x13_dashboard \
  --batch artifacts/market-observation/nfl/x13/batches/sha256-3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85 \
  --output output/x13-dashboard-a
uv run python -m prediction_market.sports.nfl_x13_dashboard \
  --batch artifacts/market-observation/nfl/x13/batches/sha256-3ddafb13fe58faa3caf01fa720a60f6c2efc4e87d9977b129b3c3824b43c1d85 \
  --output output/x13-dashboard-b
```

Expected: both exports report the same publish-manifest SHA-256 and 20 games.

- [ ] **Step 6: Commit**

```bash
git add src/prediction_market/sports/nfl_x13_dashboard.py \
  tests/sports/test_nfl_x13_dashboard.py
git commit -m "feat: export bounded X-13 dashboard assets"
```

### Task 3: Add private S3 publication and rebuildable index rows

**Files:**
- Create: `src/prediction_market/dashboard_store.py`
- Create: `src/prediction_market/cli/publish_x13_dashboard.py`
- Create: `tests/test_dashboard_store.py`
- Modify: `pyproject.toml`
- Create: `supabase/migrations/202607250001_x13_dashboard_index.sql`

- [ ] **Step 1: Write failing object-store tests**

```python
def test_publish_uploads_content_objects_before_latest_pointer(
    fake_s3: FakeS3,
    dashboard_export: Path,
) -> None:
    result = publish_dashboard_export(dashboard_export, fake_s3.config)
    assert fake_s3.put_keys[-1] == "dashboard/x13/published/latest.json"
    assert all(call.checksum_sha256 for call in fake_s3.put_calls[:-1])
    assert result.verified_object_count == len(fake_s3.put_calls) - 1
```

Add tests for remote hash mismatch, existing identical objects, conflicting
objects, prefix escape, missing env variables, and no pointer update after a
failed upload.

- [ ] **Step 2: Run the tests and confirm RED**

```bash
uv run pytest tests/test_dashboard_store.py -q
```

Expected: collection fails because `dashboard_store` does not exist.

- [ ] **Step 3: Implement the independent S3 store**

Implement `DashboardS3Config`, `load_dashboard_s3_config`, upload/head/download
verification, immutable object publication, and final pointer publication.
Read the existing `PM_DATA_S3_*` variables server-side but do not import
`prediction_market.pmxt.data_store` or use the PMXT object prefix.

- [ ] **Step 4: Add CLI**

Add:

```toml
publish-x13-dashboard = "prediction_market.cli.publish_x13_dashboard:main"
```

CLI modes:

- `export`: local verified projection only;
- `publish-s3`: upload an existing verified projection;
- `verify-s3`: re-read manifest and verify every remote object.

- [ ] **Step 5: Add the exact three-table migration**

Create tables:

```sql
create table dashboard_batches (
  batch_id text primary key,
  experiment_id text not null,
  status text not null,
  manifest_sha256 text not null unique,
  builder_version text not null,
  claim_boundary jsonb not null,
  published_at timestamptz not null
);

create table dashboard_games (
  batch_id text not null references dashboard_batches(batch_id),
  game_id text not null,
  away_team text not null,
  home_team text not null,
  away_score integer not null,
  home_score integer not null,
  event_count integer not null,
  episode_count integer not null,
  contract_count integer not null,
  audit_status text not null,
  primary key (batch_id, game_id)
);

create table dashboard_assets (
  batch_id text not null references dashboard_batches(batch_id),
  asset_path text not null,
  game_id text,
  role text not null,
  media_type text not null,
  schema_version text not null,
  source_sha256 text not null,
  object_sha256 text not null,
  byte_length bigint not null,
  primary key (batch_id, asset_path)
);
```

Enable RLS and create no anonymous read policy in version one.

- [ ] **Step 6: Verify tests and CLI**

```bash
uv run pytest tests/test_dashboard_store.py -q
uv run publish-x13-dashboard --help
```

Expected: tests pass and help lists `export`, `publish-s3`, `verify-s3`.

- [ ] **Step 7: Commit**

```bash
git add src/prediction_market/dashboard_store.py \
  src/prediction_market/cli/publish_x13_dashboard.py \
  tests/test_dashboard_store.py pyproject.toml uv.lock \
  supabase/migrations/202607250001_x13_dashboard_index.sql
git commit -m "feat: publish private dashboard assets"
```

### Task 4: Scaffold the SAF dashboard site

**Files:**
- Create: `apps/saf-dashboard/**`

- [ ] **Step 1: Initialize the site exactly once**

```bash
mkdir -p apps/saf-dashboard
cd apps/saf-dashboard
/Users/wekruitclaw1/.codex/plugins/cache/openai-bundled/sites/0.1.31/scripts/init-site.sh "$PWD"
```

Expected: Vinext starter, package lock, and `.openai/hosting.json` are created.
Do not run the initializer again.

- [ ] **Step 2: Write failing data-client and route tests**

Tests must prove:

- catalog fetch occurs before game assets;
- selecting a game fetches its core only;
- market data loads only after selecting a logical market;
- asset responses are rejected when SHA-256 differs from the manifest;
- browser code contains no `PM_DATA_S3_SECRET_ACCESS_KEY`;
- unknown batch/game/asset paths return 404.

Run:

```bash
npm test
```

Expected: tests fail because the dashboard client and private gateway are not
implemented.

- [ ] **Step 3: Implement the server-side private asset gateway**

Expose:

```text
GET /api/dashboard/catalog
GET /api/dashboard/assets/:encodedPath
```

The gateway reads S3 credentials from runtime environment variables, permits
only paths present in the verified publish manifest, checks returned bytes
against `object_sha256`, and sets private cache headers. No credential is
serialized into client props or JavaScript.

- [ ] **Step 4: Implement the 20-game index**

Replace the starter skeleton with the research index showing matchup, final
score, events, episodes, contracts, venue coverage, audit gate, and the
`PRELIMINARY_SOURCE_TIME_ONLY` boundary. Provide keyboard-accessible game
selection and responsive layout.

- [ ] **Step 5: Implement the single-game research workspace**

Render:

- field position, possession, direction, down/distance, score, and play type;
- play scrubber and source-time event list;
- offense/defense personnel and cumulative state;
- venue/family/logical-market selectors;
- both outcomes, observed/derived semantics, and Kalshi BBO when available;
- association filters for delay, horizon, ambiguity, contamination, and status;
- audit and lineage drawer with copyable stable IDs.

Use CSS shapes and typography; do not add generated football imagery or an
external chart CDN.

- [ ] **Step 6: Run tests and build**

```bash
npm test
npm run build
```

Expected: both exit zero; no starter metadata or preview skeleton remains.

- [ ] **Step 7: Commit**

```bash
git add apps/saf-dashboard
git commit -m "feat: add SAF X-13 research dashboard"
```

### Task 5: Publish the private presentation and validate end to end

**Files:**
- Modify: `README.md`
- Create: `artifacts/market-observation/nfl/x13/dashboard/x13-dashboard-publication-report-v1.json`

- [ ] **Step 1: Upload presentation assets to the existing private bucket**

Run from the repository root:

```bash
uv run publish-x13-dashboard publish-s3 \
  --env-file /Users/wekruitclaw1/Desktop/prediction-market/.env.s3 \
  --export-root output/x13-dashboard-a
```

Expected: content-addressed objects upload under the dashboard prefix, remote
SHA-256 verification passes, and `published/latest.json` is written last.

- [ ] **Step 2: Re-verify remote publication**

```bash
uv run publish-x13-dashboard verify-s3 \
  --env-file /Users/wekruitclaw1/Desktop/prediction-market/.env.s3
```

Expected: every published presentation object is found and hash-verified; no
PMXT key is read or written.

- [ ] **Step 3: Write the publication report**

Record batch ID, publish-manifest SHA-256, object count, total presentation
bytes, maximum object bytes, remote verification count, site source commit,
license boundary, and a statement that no raw/full canonical/PMXT object was
published by this command.

- [ ] **Step 4: Run focused and regression tests**

```bash
uv run pytest \
  tests/sports/test_nfl_x13_source_bundles.py \
  tests/sports/test_nfl_x13_dashboard.py \
  tests/test_dashboard_store.py \
  tests/test_x13_registry.py \
  tests/test_research_registries.py -q
cd apps/saf-dashboard
npm test
npm run build
```

Expected: all commands exit zero.

- [ ] **Step 5: Browser acceptance**

Open the site and verify:

- index lists 20 games;
- DAL–DET replay and market paths render;
- BUF–DEN high-volume market loads lazily;
- PHI–LAC overtime, GB–DAL tie, and KC–LV no-touchdown state render;
- no external network request other than the registered private asset gateway;
- no console error;
- source-time, ambiguous, contaminated, observed/derived, BBO unavailable, and
  claim-boundary labels are visible where applicable.

- [ ] **Step 6: Update README**

Document what the dashboard is for, its preliminary claim boundary, the local
start command, and the private-data requirement. Do not include credentials or
claim the registered X-13 estimand has run.

- [ ] **Step 7: Commit**

```bash
git add README.md \
  artifacts/market-observation/nfl/x13/dashboard/x13-dashboard-publication-report-v1.json
git commit -m "docs: record X-13 dashboard publication"
```

### Task 6: Final review and delivery

**Files:**
- Review all files changed by Tasks 1–5.

- [ ] **Step 1: Verify no secret or giant artifact is staged**

```bash
git diff --cached --stat
git grep -n "PM_DATA_S3_SECRET_ACCESS_KEY=" -- . ':!*.example'
git status --short
```

Expected: no credential value, canonical game JSON, full association Parquet,
or PMXT object is staged.

- [ ] **Step 2: Run final Python regression**

```bash
uv run pytest -q
```

Expected: all non-environment-gated tests pass.

- [ ] **Step 3: Run final site verification**

```bash
cd apps/saf-dashboard
npm test
npm run build
```

Expected: tests and production build pass.

- [ ] **Step 4: Review claim boundaries**

Confirm the site and report say:

- `PRELIMINARY_SOURCE_TIME_ONLY`;
- no real latency;
- no causality;
- no execution;
- no tradable-alpha claim;
- Polymarket historical BBO unavailable where not observed;
- Kalshi BBO is one-minute candle only.

- [ ] **Step 5: Commit any review-only corrections and push**

```bash
git push origin codex/hft-time-evidence-audit-v1
```

Expected: the remote branch advances to the verified local head.
