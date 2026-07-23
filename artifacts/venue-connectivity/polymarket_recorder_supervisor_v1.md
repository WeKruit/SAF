# Polymarket public recorder supervisor v1

Owner: Team B+C  
Version: v1  
Due gate: X08 prospective recorder continuity

The supervisor records the unauthenticated Polymarket public market WebSocket
raw-first. It runs for a genuinely elapsed, bounded window, stores local UTC
receive time with every exact frame, seals content-addressed raw objects, and
publishes the manifest only after the object is durable.

```text
uv run record-markets polymarket \
  --discover-sports \
  --run-seconds 604800 \
  --raw-root /durable/path/var/raw \
  --output /durable/path/health/polymarket-<run-id>.json
```

`--max-frames` is optional and is intended only for bounded smoke runs.
Production-style elapsed-window runs omit it. Reconnects are unlimited within
the requested window unless `--max-reconnects` is explicitly set.

One raw segment cannot cross a UTC hour. The first frame observed in a new UTC
hour seals the prior segment before the new frame is appended. Every reconnect
also starts a new segment and increments both `gaps` and
`continuity_unknown`; the recorder never synthesizes missing deltas or joins
connection epochs.

The health report keeps `required_elapsed_days: 7` separate from the measured
`observed_elapsed_days`. It always labels itself
`operational_observation_not_formal_x08_result`; fixtures and short smoke runs
cannot satisfy the seven-day gate. The reported uptime ratio is an operational
observation only while X-08's exact uptime denominator remains locked.

The bounded 2026-07-23 UTC smoke report is
`polymarket_recorder_smoke_20260722_v1.json`. It ran for 15 real seconds,
captured five frames with zero parse errors, sealed two verified segments, and
observed one reconnect/unknown-continuity boundary. This does not complete the
seven-day uptime gate.

This recorder performs no order placement or trading. Kalshi historical REST
contains markets, trades, and candles; this artifact makes no claim that
historical Kalshi L2 exists.
