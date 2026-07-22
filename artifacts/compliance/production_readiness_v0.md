# Production Readiness v0

- **Owner:** Team I
- **Version:** v0
- **Evidence date:** 2026-07-22
- **Due gate:** `Team_I_compliance_green`
- **Decision:** `NOT_READY`

The repository is authorized for architecture/schema work, data audit, public market recording, game-state baselines, label specification, taker simulator specification, validation registry work, and compliance/license review. It is not authorized for real-money execution.

## Hard-gate decision

`registries/compliance_matrix.csv` has no `GREEN` row and contains only unspecified operating contexts. Therefore every real-money request must fail closed. A future green decision requires one exact venue × jurisdiction × account-type row with all three reviews marked `VERIFIED` and a Team I approval reference. Unknown or unmatched contexts are denied.

## Required evidence before Team I green

| Area | Required evidence | Current blocker |
|---|---|---|
| Operator | legal entity or individual identity; physical and operating jurisdictions | not provided |
| Account | venue; account type; KYC/KYB status; beneficial owner; permitted users | not provided |
| Venue | current terms; eligibility; regulatory/account restrictions; API agreement | review open |
| Data | commercial use; redistribution; attribution; retention for each O-001–O-008 source | no row green; O-005 blocked for gambling use absent permission |
| Secrets | external inventory; least privilege; rotation and revocation test | external system and credentials not provided |
| Operations | incident owner; monitoring; recovery drill; immutable audit evidence | not executed |
| Research | required promotion gates and registered experiment evidence | incomplete |

## NO-GO audit

The following remain prohibited: real-money trading; live maker; precise queue-fill claims from PMXT L2; simultaneous multi-venue live arbitrage; live copy trading; LLM in the hot path; RL; large-scale microservices; unregistered quick backtests; and treating README returns as evidence.

## Resources needed from the user

To begin an eligibility decision, Team I needs the intended operating jurisdiction, legal entity or individual account context, intended Polymarket platform (international or US), Kalshi account type and completed agreement, authorized API scopes, and the organization’s approved secret-storage mechanism. Supplying these inputs starts review; it does not itself produce green status.
