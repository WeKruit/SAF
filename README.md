# Prediction Market Research Program

Program status: **CONDITIONAL_GO**.

The program-level sources of truth are only:

- [`charter/research_program_charter_v0.2.md`](charter/research_program_charter_v0.2.md)
- [`charter/catalog_registry.csv`](charter/catalog_registry.csv)
- [`charter/catalog_team_assignments.csv`](charter/catalog_team_assignments.csv)

The charter's [NO-GO list](charter/research_program_charter_v0.2.md#05-program-decision2026-07-22-外部评审后) remains binding. Performance or return claims in any README are never evidence and must not inform a go/no-go decision.

Validate the governed sources with:

```shell
uv run python tools/validate_governance.py
```
