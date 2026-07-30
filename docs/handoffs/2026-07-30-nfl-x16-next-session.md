# NFL X-16 Next-Session Handoff

## Authoritative X-16 Plan Location

The complete, authoritative X-16 implementation plan is:

```text
Repository-relative:
docs/superpowers/plans/2026-07-30-nfl-x16-reaction-trajectory.md

Local absolute path:
/Users/wekruitclaw1/Desktop/prediction-market/.worktrees/hft-time-evidence-audit-v1/docs/superpowers/plans/2026-07-30-nfl-x16-reaction-trajectory.md

GitHub:
https://github.com/WeKruit/SAF/blob/codex/hft-time-evidence-audit-v1/docs/superpowers/plans/2026-07-30-nfl-x16-reaction-trajectory.md
```

This handoff is only the next-session startup prompt. It does not replace,
summarize, or override the X-16 plan. The next session must read the complete
plan before inspecting implementation files or delegating work.

Copy everything below the divider into a new Codex session.

---

You are continuing the SAF repository's NFL research program.

GitHub repository:

```text
https://github.com/WeKruit/SAF
```

Required branch:

```text
codex/hft-time-evidence-audit-v1
```

Local repository:

```text
/Users/wekruitclaw1/Desktop/prediction-market
```

Canonical implementation worktree:

```text
/Users/wekruitclaw1/Desktop/prediction-market/.worktrees/hft-time-evidence-audit-v1
```

First checkout and verify the required branch:

```bash
git fetch origin
git checkout codex/hft-time-evidence-audit-v1
git pull --ff-only origin codex/hft-time-evidence-audit-v1
git status --short --branch
git rev-parse HEAD
```

The authoritative X-16 plan is:

```text
docs/superpowers/plans/2026-07-30-nfl-x16-reaction-trajectory.md
```

Read that complete file first. Then read the three program-level sources of
truth before editing:

```text
charter/research_program_charter_v0.2.md
charter/catalog_registry.csv
charter/catalog_team_assignments.csv
```

Task:

> Implement the full X-16 plan. Do not redesign the target, do not reopen
> X-15, and do not read the 81-game holdout before a passing development model
> and immutable selection lock exist.

Current facts:

- X-15 is complete and terminal `NO_MODEL_ADVANCE`; preserve all X-15 objects.
- X-16 has been designed but is not implemented.
- Development contains 153 games.
- Holdout contains 81 sealed games; reaction read count must remain zero.
- PIA V2 contains 25,250 anchors and 25,070 primary-supported anchors.
- Market inputs contain 4,773,299 actual observations.
- Kalshi is primary; Polymarket is frozen replication only.
- No upstream download is needed.
- Existing `.env.s3` contains storage credentials; never print or commit them.
- Missing local immutable objects may be hydrated only by manifest SHA from the
  existing Supabase S3 store.

Execution order:

1. Register X-16 and freeze source/hash/holdout authorities.
2. Build the decision-time feature snapshot and Stage A nflfastR reference
   provenance.
3. Build exact-window trajectory targets one game at a time:
   - 752,100 horizon candidates;
   - 1,905,320 continuation candidates.
4. Publish reconciliation and attrition before fitting models.
5. Run Kalshi B0/M1 rolling-origin OOF models and all advancement gates.
6. Freeze Polymarket replication only after Kalshi definitions are locked.
7. Update the Master Notebook and publish the offline Chinese report.
8. Open the 81-game holdout once only if every development gate passes;
   otherwise publish `NO_INCREMENTAL_MODEL_ADVANCE` and leave it sealed.

Parallel ownership is allowed and recommended:

- Agent A: X-16 governance, registry, source authority, and holdout guard.
- Agent B: feature snapshot and Stage A reference provenance.
- Agent C: market trajectory targets, attrition, and deterministic publication.
- Main agent: model integration, exact-153 run, report/notebook, and final
  verification.

Agents are not alone in the codebase. Give each agent explicit file ownership,
tell them not to revert other work, and integrate only after targeted tests.

Implementation rules:

- Write focused failing tests before production code.
- Do not create a compatibility wrapper around PanelV4.
- Do not reuse old `mark_L/mark_H`, old direction labels, exact-45 selection
  results, or cumulative X-13 reaction paths.
- Formal model features must satisfy `known_at <= T`.
- Use the actual native home contract only.
- No trade means missing/observed-trade propensity, never zero/no-move.
- Operational is the only formal prediction target.
- Isolated and continuation are retrospective diagnostics only.
- Do not pool venues or treat two venue rows as independent games.
- All inference, bootstrap, folds, and losses are game-clustered/equal-game.
- No L2, BBO, execution, P&L, Sharpe, causality, latency, or alpha claims.

Required checkpoints:

1. Target builder fixtures pass.
2. Development candidate counts match exactly.
3. Two exact-153 builds produce the same content hash.
4. Future-field mutation leaves feature hashes and predictions unchanged.
5. Holdout reaction access counter remains zero before lock.
6. Notebook/report consume only published manifests.
7. Final report prominently includes the claim boundary in the plan.

When complete, provide:

- commit and branch;
- exact artifact/report/notebook paths;
- development gate result;
- whether holdout remained sealed or was consumed;
- targeted test commands and full pass/fail counts;
- runtime and peak RSS;
- remaining data gaps and claim boundaries.

Do not stop after scaffolding. Continue through the complete 153-game
development publication, Stage A, Stage B, replication, report, and—only if
authorized by the locked gates—the one-time holdout.
