# SAF — Sports Alpha Finding

SAF is a governed research repository for sports game-state models and
prediction-market data. It does not claim validated trading alpha.

Program status: **CONDITIONAL_GO**.

Current evidence and limitations are summarized in the
[Chinese validation report](artifacts/game-state/current_validation_report_zh_v0.md).

The program-level sources of truth are only:

- [`charter/research_program_charter_v0.2.md`](charter/research_program_charter_v0.2.md)
- [`charter/catalog_registry.csv`](charter/catalog_registry.csv)
- [`charter/catalog_team_assignments.csv`](charter/catalog_team_assignments.csv)

The charter's [NO-GO list](charter/research_program_charter_v0.2.md#05-program-decision2026-07-22-外部评审后) remains binding. Performance or return claims in any README are never evidence and must not inform a go/no-go decision.

Validate the governed sources with:

```shell
uv run python tools/validate_governance.py
```

## Repository map

- `charter/`: the three program-level sources of truth; changes require program governance.
- `contracts/`: Team A versioned event, identity, model-output, rule-snapshot, replay, raw-capture, and TCA contracts.
- `registries/`: Team H experiment/artifact registries and Team I compliance/license matrices.
- `src/prediction_market/`: the governed monolith containing adapters, immutable storage, PMXT audit/reconstruction, research POCs, simulators, and replay harnesses.
- `artifacts/`: versioned specifications, evidence-bound reports, the week-one backlog, program status, and blocker register.
- `data/raw/`: append-only raw-segment convention; a manifest is the publication commit point and every segment is content-hashed.
- `var/raw/manifests/`: publishable immutable source manifests; raw source bytes remain local and ignored.
- `tests/`: contract, registry, fail-closed, deterministic replay, and integration verification.

The current owner/version/due-gate inventory is
[`registries/artifact_registry.csv`](registries/artifact_registry.csv). Current execution status and external inputs are in
[`artifacts/architecture/program_status_v0.md`](artifacts/architecture/program_status_v0.md) and
[`artifacts/architecture/risk_blocker_register_v0.csv`](artifacts/architecture/risk_blocker_register_v0.csv).

## Local verification

```shell
uv sync
uv run pytest -q
uv run audit-pmxt --help
uv run record-markets --help
```

`audit-pmxt` performs bounded public archive inventory/sample audits. `record-markets polymarket` performs bounded public WebSocket capture into an explicit raw root. Kalshi runtime activation intentionally fails closed until read-only API credentials are supplied through references documented in `artifacts/venue-connectivity/kalshi_credentials.md`.

## Evidence boundary

Passing repository tests establishes contract and harness behavior only. It does not complete X-01 through X-12, authorize a formal result, satisfy a promotion gate, establish legal eligibility, or authorize real-money execution. Synthetic fixtures are never empirical evidence. Numerical X-07 outputs remain `PRELIMINARY`; the X-09 fixture may establish `HARNESS_PASS` while the formal experiment remains `EXPERIMENT_BLOCKED`.

The following remain prohibited: real-money trading, live maker, exact queue-fill claims from PMXT L2, multi-venue simultaneous live arbitrage, live copy trading, LLMs in the hot path, reinforcement learning, large-scale microservices, unregistered quick backtests, and using README returns as evidence.
