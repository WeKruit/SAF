# NFL Factor Lab method registry

This directory documents how external research is allowed to influence the
X-13 factor lab.  It does not import external data or publish an external
author's result as SAF evidence.

The authoritative registry is
`registries/methods/method_catalog_v1.json`.  Every card separates:

- source identity and precise version;
- code license from data license;
- inputs, target, and validation;
- the idea SAF may reuse;
- what SAF has independently reproduced;
- data gaps and the claim boundary.

The loader in `catalog.py` validates exact keys, unique sorted IDs, license
gates, per-card hashes, and the catalog hash.  Unknown rights fail closed: a
card with an unknown code or data license cannot be marked
`READY_FOR_LOCAL_REPRODUCTION` or integrated.

Run the focused validation with:

```bash
uv run pytest -q tests/research/test_nfl_factor_method_catalog.py
```

No loader in this directory performs a network request.  Notebooks must use
the formal local feature pipeline and must not call an external research
package to download data.

## Integration decisions

| Method | Decision |
|---|---|
| nflfastR model outputs | Read already-computed fields from frozen `DS-NFLVERSE`; treat them as diagnostic football value, not market truth. |
| fastrmodels | Reuse the frozen no-spread WP asset and existing X-11 reproduction protocols. |
| nflreadpy | Reuse the explicit Polars/cache loader interface idea only; keep SAF's manifest verification and local reads. |
| xpass / pass_oe | Use the frozen fields as play-choice surprise; never reinterpret them as result quality. |
| nfl4th | Run later as a pinned, isolated R build and publish hashed Parquet; no result exists until that build passes fixtures. |
| Quarto | Use a pinned renderer for parameterized offline reports; it cannot define factors or compute formal statistics. |
| Big Data Bowl / Kaggle cases | Use only their research decomposition and presentation patterns until tracking access and both licenses are green. |

See [PRIMARY_SOURCE_REVIEW.md](PRIMARY_SOURCE_REVIEW.md) for the source-by-source
review.

