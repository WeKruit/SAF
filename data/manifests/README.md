# Raw segment manifests

A manifest is the commit point for one sealed raw object. The writer publishes
and fsyncs the content-addressed raw object first, then atomically publishes the
immutable manifest. Consumers enumerate manifests only, so an object without a
manifest is not a consumable segment.

Manifests mirror the raw UTC-hour partition:

```text
manifests/source=<source>/stream=<stream>/date=YYYY-MM-DD/hour=HH/<object-sha256>.manifest.json
```

Each sidecar follows `contracts/raw-capture/v0.schema.yaml` and pins the exact
object path, SHA-256, byte size, capture session, ordinal bounds, receive-time
bounds, compression, and record count. `manifest_sha256` hashes the canonical
manifest record with that field removed, binding all other fields including the
object path and `sealed_at`. Verification rejects malformed paths, symlinks,
hash or count mismatches, invalid records, non-contiguous ordinals, and an
object inode that changes during verification. Existing manifest paths are
never overwritten by this writer.

Production immutability remains `OPEN`. It requires an external WORM policy
such as Object Lock with a registered retention period, deny-delete and
deny-overwrite permissions for the capture identity, plus publication of each
manifest hash to an independent anchor. Local file permissions do not satisfy
that gate.
