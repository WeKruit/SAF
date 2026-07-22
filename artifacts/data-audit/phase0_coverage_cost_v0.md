# PMXT Phase 0 Coverage and Cost Audit v0

- **Owner:** Team C
- **Observed at:** 2026-07-22T23:28:16.971Z
- **Scope:** PMXT Polymarket v2 hourly Parquet index only
- **Source:** <https://archive.pmxt.dev/Polymarket/v2>
- **Machine evidence:** `phase0_inventory.json`
- **Status:** measured inventory; not an X-01, X-02, or X-03 result

## Method

The audit fetched all 49 pages advertised by the public index, parsed the hourly filename as UTC, deduplicated by filename, and retained the source URL and reported size for every object. It does not mix PMXT v1 into the evidence set.

The archive labels sizes as KB/MB/GB. Comparison with downloaded originals shows that the listing formatter uses binary powers: the listed 416.1 MB object is 436,334,049 bytes, or 416.1206 MiB. The inventory therefore converts the labels with 1024-based multipliers. Because the listing rounds to one decimal place, all inventory byte totals remain estimates rather than exact object lengths.

## Coverage result

| Measure | Observation |
|---|---:|
| First listed hour | 2026-04-13T19:00:00.000Z |
| Last listed hour | 2026-07-22T22:00:00.000Z |
| Inclusive expected hours | 2,404 |
| Listed unique hours/files | 2,401 |
| Listing coverage | 99.875208% |
| Missing hourly objects | 3 |

The missing hourly filenames are:

- `polymarket_orderbook_2026-06-11T04.parquet`
- `polymarket_orderbook_2026-06-11T05.parquet`
- `polymarket_orderbook_2026-06-11T06.parquet`

These are missing archive objects, not proof of lost exchange messages. PMXT has no sequence field and may skip empty hours, so the cause cannot be inferred from the listing alone.

## Reported-size cost envelope

| Measure | Observation |
|---|---:|
| Files with a reported size | 2,401 |
| Approximate listed total | 945,792,482,370 bytes (945.792 GB) |
| Approximate minimum file | 1,572,864 bytes |
| Approximate mean file | 393,916,069 bytes |
| Approximate maximum file | 764,936,192 bytes |
| Mean-size projection for 24 hours | 9.453986 GB/day |

A full local mirror at this cutoff therefore needs about 946 GB for one compressed copy before filesystem overhead, future hours, or backup copies. This is a byte envelope, not a cloud-price estimate; no vendor or retention policy was supplied.

DuckDB was exercised against both a 1.61 MB original and a 436.33 MB original containing 75,587,694 rows. It completed schema/event/timestamp aggregation and a pushed-down single-condition query. That demonstrates the single-hour audit path, not whole-archive scan feasibility. Full-mirror and full-day runtime, memory, transfer, and dollar cost remain unmeasured.

## Decision boundary

PMXT v2 is usable for bounded, content-addressed sampling from 2026-04-13T19Z onward. It is not accepted as a gap-free sequence source, and the three absent objects must remain explicit in any date-range selection. A complete formal experiment still needs preregistered market/day selection and immutable metadata joins.

## Reproduction

```bash
uv run python -m prediction_market.cli.audit_pmxt inventory \
  --output artifacts/data-audit/phase0_inventory.json
```

The planned project script entry point is `audit-pmxt = "prediction_market.cli.audit_pmxt:main"`; root integration owns that `pyproject.toml` change.
