# Secret Management v0

- **Owner:** Team I
- **Version:** v0
- **Evidence date:** 2026-07-22
- **Due gate:** `Team_I_compliance_green`
- **Status:** `NOT_GREEN_OPEN`

This contract covers credential handling for research and recorder infrastructure. It does not authorize live trading. The program-level NO-GO on real-money execution remains in force.

## Boundary

Secrets are never committed to Git, written to raw event payloads, included in experiment cards, emitted in logs, or accepted as command-line values. Code accepts only environment references to secret material. The current references are:

| Reference | Value | Consumer | Current authority |
|---|---|---|---|
| `KALSHI_KEY_ID` | Kalshi public key identifier | native Kalshi adapter | read-only market-data authentication after Team I review |
| `KALSHI_PRIVATE_KEY_PATH` | absolute path to an RSA private-key file outside the repository | native Kalshi adapter | read-only market-data authentication after Team I review |
| Polymarket signing/wallet references | not defined in v0 | none | prohibited in this phase |

The loader must reject missing files, symlinks, group/world-readable key files, non-RSA keys, and repository-contained key paths. It must not print key bytes or signatures in an exception.

## Least privilege and lifecycle

1. Use a dedicated account and key for each environment and adapter.
2. Grant only the read-only scope required for public market recording. Trading and withdrawal authority are prohibited in this phase.
3. Store key material in the operating system or approved secret manager; the repository contains only variable names.
4. Record issuer, owner, scope, creation time, expiry, rotation due date, last-use time, and revocation time in the external secret inventory.
5. Rotate on personnel change, suspected exposure, privilege change, or provider requirement. A replacement key is verified before the old key is revoked.
6. Redact authorization headers, signatures, private paths, wallet addresses when linked to identity, cookies, and response bodies that may contain account data.

## Incident runbook

On suspected exposure: stop the affected recorder; revoke the key at the venue; preserve non-secret audit logs and immutable raw segment hashes; open an incident record; determine affected time and permissions; issue a scoped replacement; review repository history and log sinks; and obtain Team I approval before restart. Deleting audit evidence or rewriting sealed raw data is prohibited.

## Promotion checks

Production readiness remains false until an external secret inventory exists, access and rotation tests pass, the exact operating context is green in `compliance_matrix.csv`, and Team I issues an approval reference. No green state is claimed by this document.
