# X-05 Barrier-Hit Label Specification v0

- **Owner:** Team E; approval owner Team H
- **Version:** v0
- **Due gate:** `2026-08-05_W2_review` (`R-022` first artifact)
- **Experiment:** `X-05`
- **Status:** `PRELIMINARY_SPEC_DRAFT`

This document defines label semantics only. The X-05 registry card currently has `label_generation` and `formal_result` unauthorized. No label dataset or numerical result is authorized by this draft.

## Locked parameters with no defaults

Every generation run must explicitly supply and preregister:

- `U`: positive upper return fraction;
- `L`: positive lower return fraction;
- `h`: positive horizon measured from the event anchor;
- maximum source-to-receive quote age;
- same-timestamp opposite-touch rule: `UPPER_FIRST`, `LOWER_FIRST`, or `AMBIGUOUS`;
- overlap rule: `KEEP_GROUPED` or `DROP_LATER`;
- purge duration and embargo duration.

v0 supplies no numerical values. The unresolved `barrier_values`, `purge_and_embargo`, `same_time_touch_rule`, quote-manifest, resume-rule, and H-signature locks in the registered X-05 card remain authoritative.

## Executable-price rule

For a long label, the first eligible quote at or after the anchor is the entry and its **ask** is `p_entry`. The upper barrier is `p_entry × (1 + U)` and the lower barrier is `p_entry × (1 - L)`. A barrier is observed only on an executable **bid**. The horizon exit also uses the first eligible bid at or after the deadline. Midpoint, last trade, indicative price, one-sided quote, and stale quote are forbidden as entry, barrier, or exit prices.

The barrier window begins after the entry quote. A touch at exactly `anchor + h` is evaluated before the horizon exit. All arithmetic is decimal fixed-point; binary float input is rejected.

## Suspension and resume

A suspension observation makes every quote with the same receive timestamp ineligible. While suspended there is no executable label price. Resume occurs at the first later non-stale, two-sided quote; that quote ID is recorded in `resume_quote_ids`. If suspension spans the horizon, the first executable quote after resume supplies the horizon bid. A stale or one-sided observation cannot end suspension for label purposes.

## Simultaneous touches

Quotes are ordered by `(received_at, quote_id)` after unique quote-ID validation. If different executable quotes at the same receive timestamp imply both upper and lower touches, the preregistered rule applies. `AMBIGUOUS` emits an explicit ambiguous outcome with no fabricated exit price. The rule cannot be selected after inspecting labels.

## Overlap, purge, and embargo

`KEEP_GROUPED` retains overlapping windows for later game-clustered evaluation; downstream splitting must keep a game together and apply the preregistered purge and embargo. `DROP_LATER` deterministically keeps the earlier anchor and excludes later anchors through `prior_window_end + purge + embargo`. Results must report the retained count, overlap degree, pre/post-purge metric difference, and game-cluster effective sample size.

## Authorization and lineage

The formal generator loads the experiment registry and refuses X-05 unless the `label_generation` scope is authorized, every required lock is resolved, code and data hashes are preregistered for that scope, and X-01 is complete. Generated result references must subsequently pass Team H's append-only registration and temporal validation. Midpoint use or unregistered parameter selection invalidates the result.
