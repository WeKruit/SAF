# Kalshi recorder credential gate

Status: **blocked pending user-provisioned credentials**. No Kalshi network capture
or authenticated connection has been attempted by this repository.

The recorder accepts references only:

- `KALSHI_API_KEY_ID` — the Kalshi-issued key identifier.
- `KALSHI_PRIVATE_KEY_PATH` — an absolute or workspace-resolved path to an
  unencrypted RSA private-key PEM file held outside Git.

Inline private-key environment values are not consumed. Private-key material,
signatures, and authentication headers must never be logged or committed. Demo
and production credentials are environment-specific and must not be reused
across endpoints.

Provisioning these references unlocks authenticated WebSocket testing; it does
not authorize real-money orders or production trading. The current scope is
read-only orderbook capture and protocol verification.

Official evidence checked 2026-07-22:

- https://docs.kalshi.com/getting_started/api_keys
- https://docs.kalshi.com/getting_started/quick_start_websockets
- https://docs.kalshi.com/websockets/orderbook-updates
