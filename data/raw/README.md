# Raw data boundary

This prefix contains sealed, byte-exact capture objects. The local writer
protocol is append-only: it never reopens, replaces, truncates, or deletes a
committed object. Any correction is a new object and manifest.

Objects use the UTC-hour partition convention:

```text
raw/source=<source>/stream=<stream>/date=YYYY-MM-DD/hour=HH/<object-sha256>.jsonl.zst
```

The filename is the lowercase SHA-256 of the exact compressed object. Each
Zstandard object contains canonical JSON lines with the original payload in
base64, its payload SHA-256, capture session, ordinal, and local receive time.
Duplicate payloads remain separate ordinal records; the raw layer does not
silently deduplicate observations.

Consumers must discover committed objects through `data/manifests`, not by
scanning this prefix. Temporary files live outside this prefix and an existing
final path is always an error.

The local no-overwrite protocol, no-follow traversal, hashes, and read-only mode
are integrity and accidental-modification controls. They are not a WORM control
and cannot resist the filesystem owner or an administrator.
