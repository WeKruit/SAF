# Normalized data boundary

Normalized data is derived state, never the source of truth. Every normalized
artifact must be possible to reconstruct from committed raw manifests and
objects plus the pinned canonical-ID snapshot, venue-rule stream,
configuration, and dependency lockfile.

Rebuilds publish a new version instead of modifying raw evidence. A normalized
record retains raw object SHA-256 and raw record ordinal lineage so replay can
return to the byte-exact observation. Present-day metadata or venue rules must
not be used to backfill historical state.
