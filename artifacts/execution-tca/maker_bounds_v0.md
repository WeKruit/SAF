# Maker Fill Bounds v0

- **Owner:** Team F2
- **Version:** v0
- **Due gate:** `2026-08-05_W2_review`
- **Status:** `SPECIFICATION_ONLY`

Phase 1 contains **no trained queue model**, no maker point estimate, and no live maker path. PMXT L2 cannot identify own queue position or exact queue fill.

Any research-only maker discussion is limited to three separately reported bounds:

- **optimistic:** displayed size ahead is treated as fully canceling before the simulated order and all eligible traded volume can fill it;
- **base:** a preregistered share of displayed size ahead remains and only observable eligible volume after that bound can fill it;
- **pessimistic:** all displayed size ahead remains and ambiguous cancels or trades provide no fill credit.

The three cases are scenario bounds, not probabilities. They must never be averaged into a point fill rate. Promotion requires own order acknowledgements, queue-relevant lifecycle data, and Team H approval in a later phase.
