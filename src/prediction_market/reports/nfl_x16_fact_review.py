"""Self-contained one-game NFL fact-review report."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from prediction_market.sports.nfl_x16_fact_extraction import GameFactTables


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _json_for_script(value: object) -> str:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _market_payload(
    market_observations: pd.DataFrame | None,
    events: pd.DataFrame,
) -> dict[str, object]:
    """Build a compact, observed-trade-only market overlay for the game book.

    This deliberately uses one point per venue/outcome/UTC second for the
    chart.  It is a display reduction of the frozen actual-trade rows; it is
    not a new reaction estimate and never forward-fills missing seconds.
    """
    if not isinstance(market_observations, pd.DataFrame) or market_observations.empty:
        return {
            "available": False,
            "semantics": "NO_HISTORICAL_MARKET_OBJECT_ATTACHED",
            "points": [],
            "start_time": None,
            "duration_s": None,
            "raw_trade_count": 0,
            "source_bundle_sha256": None,
        }
    required = {"kind", "logical_market_id", "outcome", "venue", "native_source_time"}
    if not required.issubset(market_observations.columns):
        return {
            "available": False,
            "semantics": "MARKET_SCHEMA_MISSING_REQUIRED_COLUMNS",
            "points": [],
            "start_time": None,
            "duration_s": None,
            "raw_trade_count": 0,
            "source_bundle_sha256": None,
        }
    market = market_observations.copy()
    market = market[
        market["kind"].astype(str).eq("trade")
        & market["logical_market_id"].astype(str).str.endswith(":winner")
        & market["outcome"].astype(str).isin(
            events[["home_team", "away_team"]].iloc[0].astype(str).tolist()
        )
    ].copy()
    market["source_time"] = pd.to_datetime(
        market["native_source_time"], utc=True, errors="coerce"
    )
    market["price_num"] = pd.to_numeric(market.get("price"), errors="coerce")
    market = market.dropna(subset=["source_time", "price_num"])
    if market.empty:
        return {
            "available": False,
            "semantics": "NO_OBSERVED_WINNER_TRADES",
            "points": [],
            "start_time": None,
            "duration_s": None,
            "raw_trade_count": 0,
            "source_bundle_sha256": None,
        }
    event_times = pd.to_datetime(events["source_time_utc"], utc=True, errors="coerce")
    event_times = event_times.dropna()
    start = event_times.min() if not event_times.empty else market["source_time"].min()
    end = event_times.max() if not event_times.empty else market["source_time"].max()
    # Keep a small pre/post context but do not silently use it as an endpoint.
    clipped = market[
        (market["source_time"] >= start - pd.Timedelta(seconds=60))
        & (market["source_time"] <= end + pd.Timedelta(seconds=60))
    ].copy()
    clipped["source_second"] = clipped["source_time"].dt.floor("s")
    clipped = clipped.sort_values("source_time", kind="mergesort")
    # Last actual trade in each second is a deterministic chart downsample.
    points = (
        clipped.groupby(["venue", "outcome", "source_second"], as_index=False)
        .tail(1)
        .sort_values(["source_time", "venue", "outcome"], kind="mergesort")
    )
    records: list[dict[str, object]] = []
    for row in points.itertuples(index=False):
        point_time = getattr(row, "source_time")
        records.append(
            {
                "venue": str(getattr(row, "venue")),
                "outcome": str(getattr(row, "outcome")),
                "source_time": point_time.isoformat().replace("+00:00", "Z"),
                "relative_s": round(float((point_time - start).total_seconds()), 3),
                "price": round(float(getattr(row, "price_num")), 6),
                "size": str(getattr(row, "size", "")),
            }
        )
    return {
        "available": True,
        "semantics": (
            "ACTUAL_WINNER_TRADES_ONLY; ONE_LAST_TRADE_PER_UTC_SECOND; "
            "NO_FORWARD_FILL; NO_HISTORICAL_BBO"
        ),
        "points": records,
        "start_time": start.isoformat().replace("+00:00", "Z"),
        "duration_s": round(float((end - start).total_seconds()), 3),
        "raw_trade_count": int(len(clipped)),
        "chart_point_count": int(len(records)),
        "source_bundle_sha256": market_observations.attrs.get("source_bundle_sha256"),
        "venues": sorted(str(v) for v in points["venue"].unique()),
        "outcomes": sorted(str(v) for v in points["outcome"].unique()),
    }


def render_game_fact_review(
    tables: GameFactTables,
    market_observations: pd.DataFrame | None = None,
) -> str:
    """Render an offline game-book-style review from extracted fact tables."""

    if tables.events.empty:
        raise ValueError("fact review requires at least one event")
    events = tables.events.sort_values(
        ["order_sequence", "play_id"], kind="mergesort"
    )
    first = events.iloc[0]
    last = events.iloc[-1]
    game_id = str(first["game_id"])
    home_team = str(first["home_team"])
    away_team = str(first["away_team"])
    final_home = last.get("post_home_score")
    final_away = last.get("post_away_score")
    if pd.isna(final_home):
        final_home = events["post_home_score"].dropna().iloc[-1]
    if pd.isna(final_away):
        final_away = events["post_away_score"].dropna().iloc[-1]
    preserved = int(tables.reconciliation["raw_row_preserved"].sum())
    raw_rows = int(len(tables.reconciliation))
    action_counts = Counter(str(value) for value in events["primary_action"])
    payload = {
        "game": {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "home_score": int(final_home),
            "away_score": int(final_away),
            "event_count": int(len(events)),
            "raw_rows_preserved": preserved,
            "raw_row_count": raw_rows,
            "participation_present": int(
                events["participation_status"].ne("MISSING").sum()
            ),
            "source_time_present": int(events["source_time_utc"].notna().sum()),
            "injury_evidence_count": int(len(tables.injury_evidence)),
            "active_factor_count": int(
                tables.factor_coverage["status"].eq("ACTIVE").sum()
                if not tables.factor_coverage.empty
                else 0
            ),
            "registry_sha256": tables.registry_sha256,
            "action_counts": dict(sorted(action_counts.items())),
        },
        "events": _records(events),
        "players": _records(tables.players),
        "injuries": _records(tables.injury_evidence),
        "availability": _records(tables.availability),
        "factor_hits": _records(tables.factor_hits),
        "factor_coverage": _records(tables.factor_coverage),
        "reconciliation": _records(tables.reconciliation),
        "market": _market_payload(market_observations, events),
    }
    data = _json_for_script(payload)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>NFL 单场事实复盘 · {game_id}</title>
<style>
:root {{
  color-scheme: light;
  --paper: #f4f1e8;
  --ink: #182018;
  --muted: #667064;
  --rule: #c9c5b9;
  --field: #667a55;
  --field-line: #dce6cf;
  --dal: #041e42;
  --det: #0076b6;
  --live: #29673e;
  --admin: #8b6d36;
  --turnover: #9b3a2f;
  --selection: #e3dfd2;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Avenir Next", "Gill Sans", sans-serif;
}}
button, input, select {{ font: inherit; }}
header {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 2rem;
  align-items: end;
  padding: clamp(1.25rem, 3vw, 3rem);
  border-bottom: 1px solid var(--ink);
}}
.eyebrow {{ letter-spacing: .12em; text-transform: uppercase; font-size: .75rem; }}
h1, h2, h3 {{ font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif; font-weight: 500; }}
h1 {{ margin: .35rem 0 0; font-size: clamp(2rem, 5vw, 4.8rem); line-height: .95; }}
.score {{ display: flex; gap: 1.25rem; align-items: baseline; white-space: nowrap; }}
.score strong {{ font-family: "Iowan Old Style", Georgia, serif; font-size: clamp(2rem, 5vw, 4rem); font-weight: 500; }}
.score .dal {{ color: var(--dal); }}
.score .det {{ color: var(--det); }}
.scope {{
  padding: .7rem clamp(1.25rem, 3vw, 3rem);
  border-bottom: 1px solid var(--rule);
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  gap: .6rem 1.5rem;
  font-size: .83rem;
}}
.scope b {{ color: var(--ink); font-weight: 500; }}
.toolbar {{
  display: grid;
  grid-template-columns: minmax(12rem, 1fr) repeat(3, minmax(8rem, 12rem));
  gap: .75rem;
  padding: 1rem clamp(1.25rem, 3vw, 3rem);
  border-bottom: 1px solid var(--rule);
}}
.toolbar input, .toolbar select {{
  width: 100%;
  border: 1px solid var(--rule);
  border-radius: 0;
  background: transparent;
  color: var(--ink);
  padding: .72rem .8rem;
}}
.impact-flag {{ display: inline-block; margin-left: .35rem; padding: .12rem .28rem; border: 1px solid #d97706; color: #9a4f05; font-size: .66rem; letter-spacing: .03em; white-space: nowrap; }}
.event-row.impact-row {{ background: rgba(217,119,6,.08); }}
.event-row.impact-row[aria-pressed="true"] {{ background: rgba(217,119,6,.16); }}
.timeline-wrap {{ padding: 1.2rem clamp(1.25rem, 3vw, 3rem) .6rem; }}
.timeline {{
  position: relative;
  height: 5.2rem;
  border-top: 1px solid var(--ink);
  border-bottom: 1px solid var(--rule);
}}
.timeline[tabindex="0"]:focus {{ outline: 3px solid var(--ink); outline-offset: 3px; }}
.quarter-line {{
  position: absolute; top: 0; bottom: 0; border-left: 1px solid var(--rule);
}}
.quarter-label {{ position: absolute; bottom: .3rem; color: var(--muted); font-size: .72rem; }}
.timeline button {{
  position: absolute;
  top: 1.25rem;
  width: .56rem;
  height: .56rem;
  transform: translateX(-50%);
  padding: 0;
  border: 1px solid var(--paper);
  border-radius: 50%;
  background: var(--live);
  cursor: pointer;
}}
.timeline button.score-event {{ width: .8rem; height: .8rem; top: 1.12rem; background: var(--det); }}
.timeline button.turnover-event {{ width: .8rem; height: .8rem; top: 1.12rem; background: var(--turnover); }}
.timeline button.admin-event {{ background: var(--admin); }}
.timeline button[aria-pressed="true"] {{ outline: 3px solid var(--ink); outline-offset: 2px; }}
.timeline button.impact-event {{ box-shadow: 0 0 0 3px #d97706; }}
.scrubber {{ display: grid; grid-template-columns: auto minmax(0, 1fr) minmax(0, 24rem); gap: .8rem 1.2rem; align-items: center; height: 3.25rem; margin-top: .75rem; padding: .7rem .8rem; border: 1px solid var(--rule); background: rgba(255,255,255,.22); }}
.scrub-label {{ color: var(--ink); font-size: .8rem; white-space: nowrap; }}
.scrubber output {{ display: block; min-width: 0; width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink); font-size: .8rem; font-variant-numeric: tabular-nums; }}
.scrubber .hint {{ color: var(--muted); font-size: .75rem; white-space: nowrap; }}
.market-overview {{ padding: 1rem clamp(1.25rem, 3vw, 3rem) 1.4rem; border-bottom: 1px solid var(--ink); }}
.market-overview h2 {{ margin: .3rem 0 .25rem; font-size: 1.55rem; }}
.market-meta {{ display: flex; flex-wrap: wrap; gap: .45rem 1rem; color: var(--muted); font-size: .78rem; margin-bottom: .8rem; }}
.market-chart-wrap {{ overflow-x: auto; border: 1px solid var(--rule); background: #faf8f1; }}
.market-chart {{ width: 100%; min-width: 760px; height: 310px; display: block; }}
.market-chart .grid {{ stroke: var(--rule); stroke-width: 1; }}
.market-chart .axis {{ stroke: var(--ink); stroke-width: 1; }}
.market-chart .event-line {{ stroke: var(--admin); stroke-width: 1; stroke-dasharray: 3 4; opacity: .55; }}
.market-chart .selected-line {{ stroke: var(--ink); stroke-width: 2; stroke-dasharray: 5 3; opacity: .85; }}
.market-chart .event-label {{ fill: var(--ink); font-size: 10px; }}
.market-chart .series-poly-dal {{ stroke: #1d4ed8; }}
.market-chart .series-poly-det {{ stroke: #60a5fa; }}
.market-chart .series-kalshi-dal {{ stroke: #b91c1c; }}
.market-chart .series-kalshi-det {{ stroke: #f97316; }}
.market-chart .series {{ fill: none; stroke-width: 1.8; }}
.market-chart .series-poly-dal, .market-chart .series-poly-det {{ stroke-dasharray: 5 3; stroke-width: 2.4; }}
.market-chart .point {{ stroke: #faf8f1; stroke-width: 1; }}
.market-legend {{ display: flex; flex-wrap: wrap; gap: .45rem 1rem; margin-top: .6rem; color: var(--muted); font-size: .76rem; }}
.market-legend span::before {{ content: ""; display: inline-block; width: 1.1rem; height: .16rem; margin: 0 .35rem .15rem 0; background: currentColor; }}
.market-legend .poly-dal {{ color: #1d4ed8; }} .market-legend .poly-det {{ color: #60a5fa; }}
.market-legend .kalshi-dal {{ color: #b91c1c; }} .market-legend .kalshi-det {{ color: #f97316; }}
.market-note {{ color: var(--muted); font-size: .78rem; margin: .7rem 0 0; max-width: 80rem; }}
.market-local {{ margin-top: 1.25rem; }}
.market-local h3 {{ margin-bottom: .35rem; }}
.market-no-data {{ border: 1px dashed var(--rule); padding: 1rem; color: var(--muted); }}
.overview-disclosure {{ border-bottom: 1px solid var(--ink); }}
.overview-disclosure > summary {{
  cursor: pointer;
  list-style: none;
  padding: .9rem clamp(1.25rem, 3vw, 3rem);
  color: var(--ink);
  font-size: .84rem;
  letter-spacing: .02em;
  background: rgba(255,255,255,.2);
}}
.overview-disclosure > summary::-webkit-details-marker {{ display: none; }}
.overview-disclosure > summary::before {{ content: "＋"; display: inline-block; width: 1.25rem; color: var(--muted); }}
.overview-disclosure[open] > summary::before {{ content: "−"; }}
.overview-disclosure > summary:hover {{ background: var(--selection); }}
.current-event-strip {{
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) auto;
  align-items: center;
  gap: .8rem 1.2rem;
  margin: 1rem clamp(1.25rem, 3vw, 3rem);
  padding: .85rem 1rem;
  border: 1px solid var(--ink);
  border-left: 5px solid var(--det);
  background: #faf8f1;
}}
.current-event-strip .event-number {{ color: var(--muted); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; white-space: nowrap; }}
.current-event-strip strong {{ font-weight: 600; }}
.current-event-strip .current-description {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.current-event-strip .current-meta {{ color: var(--muted); font-size: .78rem; white-space: nowrap; }}
.current-event-strip button, .nav-toggle {{ border: 1px solid var(--rule); background: transparent; color: var(--ink); padding: .42rem .65rem; cursor: pointer; }}
.current-event-strip button:hover, .nav-toggle:hover {{ background: var(--selection); }}
.workspace {{
  display: grid;
  grid-template-columns: minmax(18rem, 34rem) minmax(0, 1fr);
  min-height: 52rem;
  border-top: 1px solid var(--ink);
}}
.workspace.navigator-collapsed {{ grid-template-columns: minmax(0, 1fr); }}
.event-index {{ border-right: 1px solid var(--ink); }}
.workspace.navigator-collapsed .event-index {{ border-right: 0; border-bottom: 1px solid var(--ink); }}
.workspace.navigator-collapsed .event-list {{ display: none; }}
.event-index-head {{
  padding: 1rem 1.25rem;
  display: flex;
  align-items: center;
  gap: .8rem;
  justify-content: space-between;
  border-bottom: 1px solid var(--rule);
}}
.event-index-head > b {{ font-weight: 500; }}
.event-index-head > span {{ color: var(--muted); font-size: .78rem; margin-left: auto; }}
.event-list {{ max-height: 52rem; overflow-y: auto; }}
.event-row {{
  appearance: none;
  width: 100%;
  display: grid;
  grid-template-columns: 4.5rem 4rem minmax(0, 1fr);
  gap: .7rem;
  text-align: left;
  border: 0;
  border-bottom: 1px solid var(--rule);
  background: transparent;
  color: var(--ink);
  padding: .78rem 1.25rem;
  cursor: pointer;
}}
.event-row:hover {{ background: var(--selection); }}
.event-row[aria-pressed="true"] {{
  background: var(--selection);
  box-shadow: inset 5px 0 0 var(--ink);
  padding-left: calc(1.25rem + 5px);
}}
.event-row .clock {{ font-variant-numeric: tabular-nums; }}
.event-row .action {{ color: var(--muted); font-size: .75rem; }}
.event-row .desc {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.detail {{ min-width: 0; }}
.detail-tabs {{ position: sticky; top: 0; z-index: 4; background: var(--paper); }}
.detail-tabs {{
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--ink);
}}
.detail-tabs button {{
  border: 0;
  border-right: 1px solid var(--rule);
  border-radius: 0;
  background: transparent;
  color: var(--ink);
  padding: .85rem 1.15rem;
  cursor: pointer;
}}
.detail-tabs button[aria-selected="true"] {{ background: var(--ink); color: var(--paper); }}
.panel {{ display: none; padding: clamp(1.25rem, 3vw, 2.5rem); }}
.panel.active {{ display: block; }}
#panel-replay.active {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1rem 1.5rem; align-items: start; }}
#panel-replay.active > .event-heading {{ grid-column: 1 / -1; }}
#panel-replay.active > .football-field, #panel-replay.active > .legend, #panel-replay.active > .raw-description, #panel-replay.active > .tags {{ grid-column: 1; }}
#panel-replay.active > .state-grid {{ grid-column: 1 / -1; }}
.inline-market-card {{ grid-column: 2; grid-row: 2 / span 5; align-self: stretch; min-height: 100%; padding: .9rem; border: 1px solid var(--rule); background: #faf8f1; }}
.inline-market-card h3 {{ margin: 0 0 .35rem; font-size: 1.1rem; }}
.inline-market-card .market-chart-wrap {{ overflow: hidden; }}
.inline-market-card .market-chart {{ min-width: 0; height: 280px; }}
.inline-market-card .market-legend {{ display: grid; grid-template-columns: 1fr 1fr; gap: .35rem .5rem; font-size: .7rem; }}
.inline-market-card .market-note {{ font-size: .7rem; line-height: 1.35; }}
.event-heading {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 1rem;
  align-items: start;
  margin-bottom: 1.4rem;
}}
.event-heading h2 {{ margin: 0; font-size: clamp(1.6rem, 3vw, 2.8rem); }}
.event-heading .clock-big {{ font-size: 1.25rem; font-variant-numeric: tabular-nums; }}
.raw-description {{
  font-family: "Iowan Old Style", Georgia, serif;
  font-size: 1.1rem;
  line-height: 1.55;
  margin: 1.25rem 0 1.6rem;
  max-width: 70rem;
}}
.football-field {{
  width: 100%;
  min-height: 15rem;
  display: block;
  background: var(--field);
  border: 1px solid var(--ink);
}}
.legend {{ display: flex; flex-wrap: wrap; gap: .45rem 1rem; margin-top: .55rem; font-size: .78rem; color: var(--muted); }}
.legend span::before {{ content: ""; display: inline-block; width: .65rem; height: .65rem; margin-right: .35rem; vertical-align: -.05rem; border-radius: 50%; background: currentColor; }}
.legend .dal {{ color: var(--dal); }} .legend .det {{ color: var(--det); }}
.state-grid {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  border-top: 1px solid var(--ink);
  border-bottom: 1px solid var(--ink);
  margin-top: 1.5rem;
}}
.state-grid section {{ padding: 1.2rem; border-right: 1px solid var(--rule); }}
.state-grid section:last-child {{ border-right: 0; }}
.state-grid h3 {{ margin: 0 0 .7rem; }}
.facts {{ margin: 0; display: grid; grid-template-columns: minmax(9rem, 15rem) minmax(0, 1fr); gap: .35rem .9rem; }}
.facts dt {{ color: var(--muted); }} .facts dd {{ margin: 0; overflow-wrap: break-word; word-break: normal; min-width: 0; }}
.tags {{ display: flex; flex-wrap: wrap; gap: .35rem; margin: .8rem 0; }}
.tag {{ border: 1px solid var(--rule); padding: .28rem .45rem; font-size: .75rem; }}
table {{ width: 100%; border-collapse: collapse; font-size: .84rem; }}
th, td {{ text-align: left; vertical-align: top; padding: .55rem .5rem; border-bottom: 1px solid var(--rule); }}
th {{ color: var(--muted); font-weight: 500; }}
code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: .76rem; overflow-wrap: anywhere; }}
.empty {{ color: var(--muted); font-style: italic; }}
.audit-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); border-top: 1px solid var(--ink); }}
.audit-grid div {{ padding: 1rem; border-right: 1px solid var(--rule); border-bottom: 1px solid var(--rule); }}
.audit-grid strong {{ display: block; font-family: "Iowan Old Style", Georgia, serif; font-size: 1.8rem; font-weight: 500; }}
.claim {{ border-top: 1px solid var(--ink); margin-top: 2rem; padding-top: 1rem; color: var(--muted); }}
@media (max-width: 900px) {{
  header {{ grid-template-columns: 1fr; }}
  .toolbar {{ grid-template-columns: 1fr; }}
  .workspace {{ grid-template-columns: 1fr; }}
  .event-index {{ border-right: 0; border-bottom: 1px solid var(--ink); }}
  .event-list {{ max-height: 24rem; }}
  .current-event-strip {{ grid-template-columns: 1fr; gap: .35rem; }}
  .current-event-strip .current-description {{ white-space: normal; }}
  .current-event-strip .current-meta {{ white-space: normal; }}
  .state-grid, .audit-grid {{ grid-template-columns: 1fr; }}
  .state-grid section, .audit-grid div {{ border-right: 0; border-bottom: 1px solid var(--rule); }}
  .scrubber {{ grid-template-columns: auto minmax(0, 1fr); height: auto; min-height: 3.25rem; }}
  .scrubber .hint {{ grid-column: 1 / -1; }}
  #panel-replay.active {{ display: block; }}
  .inline-market-card {{ margin: 1rem 0; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior: auto !important; }} }}
</style>
</head>
<body>
<header>
  <div>
    <div class="eyebrow">NFL fact review / NFL 事实复盘 · draft registry / 草案注册 · source-time market overlay / 源时间市场叠加</div>
    <h1>{away_team} at {home_team}</h1>
  </div>
  <div class="score" aria-label="Final score">
    <span class="dal">{away_team} <strong>{int(final_away)}</strong></span>
    <span>—</span>
    <span class="det"><strong>{int(final_home)}</strong> {home_team}</span>
  </div>
</header>
<div class="scope">
  <span><b>{preserved} / {raw_rows} raw rows preserved / 原始行保留</b></span>
  <span>One canonical event per source row / 每个源行一个规范事件</span>
  <span>Participation is research-only / 球员参与层仅研究用途</span>
  <span><b>Historical trades aligned by source time / 历史成交按源时间对齐</b></span>
  <span>Market data deliberately excluded from causal claims / 市场数据不用于因果结论</span>
</div>
<div class="toolbar">
  <label><span class="eyebrow">Search event / 搜索事件</span><input id="event-search" type="search" placeholder="player, action, tag, raw description / 球员、动作、标签、原文"></label>
  <label><span class="eyebrow">Quarter / 节次</span><select id="quarter-filter"><option value="ALL">All quarters / 全部节</option><option value="1">Q1 / 第一节</option><option value="2">Q2 / 第二节</option><option value="3">Q3 / 第三节</option><option value="4">Q4 / 第四节</option><option value="5">OT / 加时</option></select></label>
  <label><span class="eyebrow">Action / 动作</span><select id="action-filter"><option value="ALL">All actions / 全部动作</option></select></label>
  <label><span class="eyebrow">Impact threshold / 影响阈值</span><select id="impact-threshold-filter"><option value="0.01">≥1pp diagnostic WP Δ</option><option value="0.02">≥2pp diagnostic WP Δ</option><option value="0.05" selected>≥5pp diagnostic WP Δ</option><option value="0.10">≥10pp diagnostic WP Δ</option></select></label>
</div>
<section class="timeline-wrap" aria-label="Full game timeline">
  <div id="timeline" class="timeline" tabindex="0" role="slider" aria-label="Keyboard event timeline / 键盘事件时间轴" aria-valuemin="0" aria-valuemax="3600"></div>
  <div class="scrubber" aria-label="Keyboard event timeline / 键盘事件时间轴">
    <span class="scrub-label">Use ↑ ↓ / 使用上下键</span><output id="scrub-readout">—</output>
    <span class="hint">previous/next event; replay and market follow / 上一个或下一个事件，复盘和市场同步</span>
  </div>
</section>
<div id="current-event-strip" class="current-event-strip" aria-live="polite"></div>
<details class="overview-disclosure">
<summary>Full-game market overview / 全场市场总览（点击展开；当前复核优先显示局部窗口）</summary>
<section class="market-overview" aria-label="Source-time market timeline / 源时间市场时间轴">
  <h2>Source-time timeline / 源时间时间轴 × winner probability / 胜者概率</h2>
  <div id="market-meta" class="market-meta"></div>
  <div id="game-market-chart" class="market-chart-wrap"></div>
  <div class="market-legend"><span class="poly-dal">Polymarket {away_team}</span><span class="poly-det">Polymarket {home_team}</span><span class="kalshi-dal">Kalshi {away_team}</span><span class="kalshi-det">Kalshi {home_team}</span></div>
  <p class="market-note">Only observed winner trades are shown / 只显示实际成交的 winner 合约。每个 venue/outcome/UTC 秒取该秒最后一笔成交；缺失秒保持空白，不 forward-fill。历史没有完整 BBO/L2，因此这里不是 bid/ask 或可执行价格。</p>
</section>
 </details>
<main id="workspace" class="workspace">
  <aside class="event-index">
    <div class="event-index-head"><button id="event-index-toggle" class="nav-toggle" type="button" aria-expanded="true">Hide navigator / 收起事件导航</button><span id="visible-count"></span></div>
    <div id="event-list" class="event-list"></div>
  </aside>
  <article class="detail">
    <nav class="detail-tabs" aria-label="Fact review sections">
      <button type="button" data-tab="replay" aria-selected="true">Replay</button>
      <button type="button" data-tab="market" aria-selected="false">Market / 市场</button>
      <button type="button" data-tab="players" aria-selected="false">Players / 球员</button>
      <button type="button" data-tab="registry" aria-selected="false">Factor hits / 因子命中</button>
      <button type="button" data-tab="audit" aria-selected="false">Audit / 审计</button>
    </nav>
    <section id="panel-replay" class="panel active"></section>
    <section id="panel-market" class="panel"></section>
    <section id="panel-players" class="panel"></section>
    <section id="panel-registry" class="panel"></section>
    <section id="panel-audit" class="panel"></section>
  </article>
</main>
<script>
const D={data};
const M=D.market||{{available:false,points:[]}};
const $=s=>document.querySelector(s);
const esc=v=>String(v??"—").replace(/[&<>"']/g,m=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[m]));
const tags=e=>{{try{{return JSON.parse(e.outcome_tags||"[]")}}catch{{return []}}}};
const val=v=>v===null||v===undefined||Number.isNaN(v)?"—":v;
const pp=v=>v===null||v===undefined?"—":`${{(Number(v)*100).toFixed(2)}}pp`;
let impactThreshold=0.05;
const diagnosticDelta=e=>Number(e?.home_wp_delta_diagnostic);
const hasMaterialImpact=e=>Number.isFinite(diagnosticDelta(e))&&Math.abs(diagnosticDelta(e))>=impactThreshold;
const impactCopy=e=>hasMaterialImpact(e)?`ΔWP ${{diagnosticDelta(e)>=0?"+":""}}${{(diagnosticDelta(e)*100).toFixed(2)}}pp · diagnostic / 诊断` : "";
const gameClock=e=>`Q${{e.quarter??"?"}} · ${{e.game_clock||"—"}}`;
const directionCopy=v=>({{
  SINGLE_PHASE_TEAM_NORMALIZED:"single-phase · team-normalized",
  KICK_AND_POSSIBLE_RETURN_NET_STATE_ONLY:"kick + possible return · net state",
  OFFENSE_THEN_POSSIBLE_RETURN_NET_STATE_ONLY:"offense + possible return · net state",
  UNAVAILABLE:"unavailable"
}}[v]||v);
const endpointCopy=v=>({{
  DERIVED_FROM_STRUCTURED_YARDS_GAINED_ON_TEAM_NORMALIZED_FIELD:"derived from structured yards gained",
  NEXT_OBSERVED_STRUCTURED_SCRIMMAGE_STATE_NOT_EXACT_BALL_TRAJECTORY:"next observed scrimmage state · not exact trajectory",
  UNAVAILABLE:"unavailable"
}}[v]||v);
function marketPointColor(point){{
  const away=D.game.away_team, home=D.game.home_team;
  if(point.venue==="polymarket") return point.outcome===away?"#1d4ed8":"#60a5fa";
  return point.outcome===away?"#b91c1c":"#f97316";
}}
function marketSeriesClass(point){{
  const team=point.outcome===D.game.away_team?"dal":"det";
  return `series-${{point.venue}}-${{team}}`;
}}
function marketEventRelative(e){{
  if(!M.start_time||!e.source_time_utc) return null;
  const t=Date.parse(e.source_time_utc), s=Date.parse(M.start_time);
  return Number.isFinite(t)&&Number.isFinite(s)?(t-s)/1000:null;
}}
function marketSvg(centerEvent, local=false, sourceEvents=D.events){{
  if(!M.available||!M.points.length) return `<div class="market-no-data">No observed winner trades / 没有可用的 winner 实际成交。</div>`;
  const W=1100,H=310,L=54,R=18,T=24,B=34;
  const center=local?marketEventRelative(centerEvent):0;
  const x0=local?Math.max(-30,(center??0)-30):0;
  const x1=local?Math.min((M.duration_s||60)+60,(center??0)+60):(M.duration_s||60);
  const span=Math.max(1,x1-x0);
  const x=s=>L+(Number(s)-x0)/span*(W-L-R);
  const y=p=>T+(1-Math.max(0,Math.min(1,Number(p))))*(H-T-B);
  const grid=[0,.25,.5,.75,1].map(p=>`<line class="grid" x1="${{L}}" x2="${{W-R}}" y1="${{y(p)}}" y2="${{y(p)}}"/><text x="${{L-8}}" y="${{y(p)+4}}" text-anchor="end" fill="var(--muted)" font-size="11">${{Math.round(p*100)}}%</text>`).join("");
  const groups={{}};
  M.points.forEach(point=>{{
    const s=Number(point.relative_s), p=Number(point.price);
    if(!Number.isFinite(s)||!Number.isFinite(p)||s<x0-1||s>x1+1) return;
    const key=`${{point.venue}}|${{point.outcome}}`;
    (groups[key] ||= []).push(point);
  }});
  let paths="";
  Object.values(groups).forEach(points=>{{
    points.sort((a,b)=>Number(a.relative_s)-Number(b.relative_s));
    let d="", last=null;
    points.forEach(point=>{{
      const s=Number(point.relative_s);
      const gap=last===null?Infinity:s-last;
      d+=`${{(!d||gap>10)?"M":"L"}} ${{x(s).toFixed(2)}} ${{y(point.price).toFixed(2)}} `;
      last=s;
    }});
    if(d) paths+=`<path class="series ${{marketSeriesClass(points[0])}}" d="${{d}}"/>`;
    if(local) points.forEach(point=>{{ paths+=`<circle class="point" cx="${{x(point.relative_s)}}" cy="${{y(point.price)}}" r="2.4" fill="${{marketPointColor(point)}}"><title>${{point.venue}} · ${{point.outcome}} · ${{(Number(point.price)*100).toFixed(1)}}% · ${{point.source_time}}</title></circle>`; }});
  }});
  const eventLines=sourceEvents.map(e=>{{
    const s=marketEventRelative(e); if(s===null||s<x0||s>x1) return "";
    const selected=centerEvent&&e.event_id===centerEvent.event_id;
    const salient=tags(e).some(t=>["SCORE","TOUCHDOWN","TURNOVER","FIELD_GOAL"].includes(t));
    if(!local&&!salient&&!selected) return "";
    return `<line class="${{selected?"selected-line":"event-line"}}" x1="${{x(s)}}" x2="${{x(s)}}" y1="${{T}}" y2="${{H-B}}"/>${{salient?`<text class="event-label" transform="translate(${{x(s)+3}},${{T+12}}) rotate(-70)">${{esc(e.primary_action)}} · Q${{e.quarter}}</text>`:""}}`;
  }}).join("");
  const ticks=Array.from({{length:Math.min(9,Math.floor(span/30)+2)}},(_,i)=>x0+i*span/(Math.min(8,Math.floor(span/30)+1))).filter(v=>v<=x1+1).map(v=>`<text x="${{x(v)}}" y="${{H-10}}" text-anchor="middle" fill="var(--muted)" font-size="11">${{v<0?"":`T+${{Math.round(v)}}s`}}</text>`).join("");
  const title=local?`Selected event window / 当前事件窗口 · ${{centerEvent?gameClock(centerEvent):"—"}}`:`Full game source-time axis / 全场源时间轴`;
  return `<svg class="market-chart" viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="${{title}}"><text x="${{L}}" y="15" fill="var(--muted)" font-size="12">${{title}}</text>${{grid}}<line class="axis" x1="${{L}}" x2="${{L}}" y1="${{T}}" y2="${{H-B}}"/><line class="axis" x1="${{L}}" x2="${{W-R}}" y1="${{H-B}}" y2="${{H-B}}"/>${{eventLines}}${{paths}}${{ticks}}</svg>`;
}}
function renderMarketOverview(){{
  const meta=$("#market-meta");
  if(!M.available){{ meta.innerHTML="No market object attached / 未附加市场对象"; $("#game-market-chart").innerHTML=marketSvg(null,false); return; }}
  meta.innerHTML=`<span><b>${{M.raw_trade_count.toLocaleString()}}</b> raw winner trades / 原始 winner 成交</span><span><b>${{M.chart_point_count.toLocaleString()}}</b> chart points / 图表点</span><span>start / 起点 ${{M.start_time}}</span><span>venues / venue：${{M.venues.join(" · ")}}</span>`;
  $("#game-market-chart").innerHTML=marketSvg(null,false,filtered);
}}
function renderMarket(e){{
  const root=$("#panel-market");
  const center=marketEventRelative(e);
  const start=center===null?"—":`T${{center>=0?"+":""}}${{center.toFixed(3)}}s`;
  const localPoints=M.points.filter(p=>center!==null&&Number(p.relative_s)>=center-30&&Number(p.relative_s)<=center+60).sort((a,b)=>Number(a.relative_s)-Number(b.relative_s));
  root.innerHTML=`<div class="event-heading"><div><div class="eyebrow">Source-time market alignment / 源时间市场对齐</div><h2>Winner probability path / 胜者概率路径</h2></div><div class="clock-big">${{esc(gameClock(e))}}<br>${{esc(start)}}</div></div><p class="raw-description">${{esc(e.description)}}</p><div class="market-meta"><span>event source time / 事件源时间：<b>${{esc(e.source_time_utc||"unavailable / 不可用")}}</b></span><span>window / 窗口：T−30s → T+60s</span><span>observed trade only / 仅实际成交</span></div><div class="market-chart-wrap">${{marketSvg(e,true,D.events)}}</div><div class="market-legend"><span class="poly-dal">Polymarket ${{D.game.away_team}}</span><span class="poly-det">Polymarket ${{D.game.home_team}}</span><span class="kalshi-dal">Kalshi ${{D.game.away_team}}</span><span class="kalshi-det">Kalshi ${{D.game.home_team}}</span></div><p class="market-note">The lines are source-time trend alignment, not causal attribution / 这些线是源时间趋势对齐，不代表事件因果。市场没有成交的秒保持空白；midpoint、bid/ask、L2 和 execution 不在此页。</p><div class="market-local"><h3>Nearest observed trades / 最近实际成交</h3>${{localPoints.length?`<table><thead><tr><th>Venue</th><th>Outcome / 结果</th><th>Source time / 源时间</th><th>Relative / 相对秒</th><th>Price / 概率</th><th>Size</th></tr></thead><tbody>${{localPoints.slice(0,120).map(p=>`<tr><td>${{esc(p.venue)}}</td><td>${{esc(p.outcome)}}</td><td><code>${{esc(p.source_time)}}</code></td><td>${{(Number(p.relative_s)-center).toFixed(3)}}s</td><td>${{(Number(p.price)*100).toFixed(2)}}%</td><td>${{esc(p.size)}}</td></tr>`).join("")}}</tbody></table>`:'<p class="empty">No observed winner trade in this window / 此窗口无 winner 实际成交。</p>'}}</div>`;
}}
const elapsed=e=>{{
  if(Number.isFinite(Number(e.game_seconds_remaining))) return Math.max(0,Math.min(3600,3600-Number(e.game_seconds_remaining)));
  const q=Number(e.quarter||1), parts=String(e.game_clock||"0:00").split(":").map(Number);
  return Math.min(3600,(q-1)*900+(900-(parts[0]*60+(parts[1]||0))));
}};
let filtered=[...D.events], selected=D.events.find(e=>e.factor_eligible)||D.events[0];
function syncScrubber(){{
  const readout=$("#scrub-readout");
  if(!readout||!selected) return;
  readout.textContent=`${{gameClock(selected)}} · ${{selected.primary_action}} / ${{selected.description||""}}`;
  const timeline=$("#timeline");
  if(timeline){{
    const elapsedValue=Math.round(elapsed(selected));
    timeline.setAttribute("aria-valuenow",String(elapsedValue));
    timeline.setAttribute("aria-valuetext",`${{gameClock(selected)}} · ${{selected.primary_action}}`);
  }}
}}
function renderCurrentStrip(e){{
  const root=$("#current-event-strip");
  if(!root||!e) return;
  const index=D.events.findIndex(row=>row.event_id===e.event_id);
  const eventNo=index<0?"—":`${{index+1}} / ${{D.events.length}}`;
  const marketPosition=marketEventRelative(e);
  const marketCopy=marketPosition===null?"market time unavailable / 市场时间不可用":`market T${{marketPosition>=0?"+":""}}${{marketPosition.toFixed(3)}}s · observed trades / 实际成交`;
  const downDistance=e.down?`${{e.down}} & ${{e.distance}}`:"down/distance —";
  const stateCopy=`state / 状态：${{e.possession_before||"—"}} · score ${{e.pre_away_score}}–${{e.pre_home_score}} · ${{downDistance}} · ${{e.pre_yardline||"—"}}`;
  const impactBadge=hasMaterialImpact(e)?`<span class="impact-flag">${{esc(impactCopy(e))}}</span>`:"";
  root.innerHTML=`<span class="event-number">Event / 事件 ${{eventNo}}</span><span class="current-description"><strong>${{esc(gameClock(e))}} · ${{esc(e.primary_action)}}</strong> ${{impactBadge}} · ${{esc(e.description||"Administrative / 管理事件")}}</span><span class="current-meta">${{esc(stateCopy)}}<br>${{esc(marketCopy)}}</span><button id="current-market-button" type="button">Open event market / 打开事件市场</button>`;
  $("#current-market-button").onclick=()=>{{
    document.querySelectorAll(".detail-tabs button").forEach(b=>b.setAttribute("aria-selected",String(b.dataset.tab==="market")));
    document.querySelectorAll(".panel").forEach(p=>p.classList.toggle("active",p.id==="panel-market"));
    $("#panel-market")?.scrollIntoView({{behavior:"smooth",block:"start"}});
  }};
}}
function selectEvent(e){{
  if(!e) return;
  selected=e;
  renderIndex(); renderTimeline(); renderDetail(); renderMarketOverview(); syncScrubber();
}}
function scrubTo(value){{
  const target=Number(value);
  if(!filtered.length||!Number.isFinite(target)) return;
  selected=filtered.reduce((best,e)=>Math.abs(elapsed(e)-target)<Math.abs(elapsed(best)-target)?e:best,filtered[0]);
  renderIndex(); renderTimeline(); renderDetail(); renderMarketOverview(); syncScrubber();
}}
function moveSelected(step){{
  if(!filtered.length) return;
  const currentIndex=Math.max(0,filtered.findIndex(e=>e.event_id===selected?.event_id));
  const nextIndex=Math.max(0,Math.min(filtered.length-1,currentIndex+step));
  selectEvent(filtered[nextIndex]);
}}

function eventClass(e){{
  const t=tags(e);
  let base="";
  if(t.includes("TURNOVER")) base="turnover-event";
  else if(t.includes("SCORE")) base="score-event";
  else if(["ADMIN","TIMEOUT","PENALTY"].includes(e.primary_action)) base="admin-event";
  return hasMaterialImpact(e)?`${{base}} impact-event`.trim():base;
}}
function populateActions(){{
  const sel=$("#action-filter");
  [...new Set(D.events.map(e=>e.primary_action))].sort().forEach(action=>{{
    const o=document.createElement("option"); o.value=action; o.textContent=action; sel.appendChild(o);
  }});
}}
function applyFilters(){{
  const q=$("#quarter-filter").value, action=$("#action-filter").value, needle=$("#event-search").value.trim().toLowerCase();
  impactThreshold=Number($("#impact-threshold-filter").value)||0.05;
  filtered=D.events.filter(e=>{{
    const hay=[e.description,e.primary_action,e.possession_before,...tags(e)].join(" ").toLowerCase();
    return (q==="ALL"||String(e.quarter)===q)&&(action==="ALL"||e.primary_action===action)&&(!needle||hay.includes(needle));
  }});
  if(!filtered.some(e=>e.event_id===selected.event_id)&&filtered.length) selected=filtered[0];
  renderIndex(); renderTimeline(); renderDetail(); renderMarketOverview(); syncScrubber();
}}
function renderTimeline(){{
  const root=$("#timeline");
  root.innerHTML=[1,2,3].map(q=>`<span class="quarter-line" style="left:${{q*25}}%"></span>`).join("")+
    [1,2,3,4].map(q=>`<span class="quarter-label" style="left:${{(q-1)*25+.6}}%">Q${{q}}</span>`).join("");
  filtered.forEach(e=>{{
    const b=document.createElement("button"); b.type="button"; b.className=eventClass(e);
    b.style.left=`${{(elapsed(e)/3600)*100}}%`; b.title=`${{gameClock(e)}} · ${{e.primary_action}} · ${{e.description||""}}`;
    b.setAttribute("aria-label",b.title); b.setAttribute("aria-pressed",String(e.event_id===selected.event_id));
    b.onclick=()=>selectEvent(e);
    root.appendChild(b);
  }});
}}
function renderIndex(){{
  const impactCount=filtered.filter(hasMaterialImpact).length;
  $("#visible-count").textContent=`${{filtered.length}} / ${{D.events.length}} · impact ${{impactCount}}`;
  $("#event-list").innerHTML=filtered.map(e=>`<button class="event-row ${{hasMaterialImpact(e)?"impact-row":""}}" type="button" data-event="${{esc(e.event_id)}}" aria-pressed="${{e.event_id===selected.event_id}}">
    <span class="clock">${{esc(gameClock(e))}}</span><span class="action">${{esc(e.primary_action)}}${{hasMaterialImpact(e)?`<b class="impact-flag">${{esc(impactCopy(e))}}</b>`:""}}</span><span class="desc">${{esc(e.description)}}</span>
  </button>`).join("");
  document.querySelectorAll(".event-row").forEach(b=>b.onclick=()=>selectEvent(D.events.find(e=>e.event_id===b.dataset.event)));
  document.querySelector('.event-row[aria-pressed="true"]')?.scrollIntoView({{block:"nearest"}});
}}
function fieldSvg(e){{
  const start=Number(e.pre_field_coordinate_0_100), end=Number(e.visual_end_field_coordinate_0_100);
  const hasStart=e.pre_field_coordinate_0_100!==null&&e.pre_field_coordinate_0_100!==undefined&&Number.isFinite(start);
  const hasEnd=e.visual_end_field_coordinate_0_100!==null&&e.visual_end_field_coordinate_0_100!==undefined&&Number.isFinite(end);
  const sx=hasStart?50+start*9:500, ex=hasEnd?50+end*9:sx;
  const direction=e.offense_direction==="LEFT"?"←":e.offense_direction==="RIGHT"?"→":e.offense_direction==="BIDIRECTIONAL"?"↔":"";
  const lines=Array.from({{length:11}},(_,i)=>`<line x1="${{50+i*90}}" y1="0" x2="${{50+i*90}}" y2="240" stroke="var(--field-line)" opacity="${{i===0||i===10?.8:.38}}"/><text x="${{50+i*90}}" y="228" text-anchor="middle" fill="var(--field-line)" font-size="13">${{i<=5?i*10:(10-i)*10}}</text>`).join("");
  const path=hasStart&&hasEnd?`<line x1="${{sx}}" y1="120" x2="${{ex}}" y2="120" stroke="#f4f1e8" stroke-width="5"/><polygon points="${{ex}},120 ${{ex+(ex>=sx?-15:15)}},111 ${{ex+(ex>=sx?-15:15)}},129" fill="#f4f1e8"/>`:"";
  const startMarker=hasStart?`<circle cx="${{sx}}" cy="120" r="10" fill="var(--ink)" stroke="#f4f1e8" stroke-width="3"/>`:"";
  const endMarker=hasEnd?`<circle cx="${{ex}}" cy="120" r="8" fill="#f4f1e8" stroke="var(--ink)" stroke-width="3"/>`:"";
  return `<svg class="football-field" viewBox="0 0 1000 240" role="img" aria-label="Team-normalized field, ${{esc(e.pre_yardline)}} to ${{esc(e.next_observed_yardline)}}">${{lines}}<rect x="0" y="0" width="50" height="240" fill="var(--dal)"/><rect x="950" y="0" width="50" height="240" fill="var(--det)"/><text x="25" y="125" text-anchor="middle" fill="#f4f1e8" transform="rotate(-90 25 125)">DAL</text><text x="975" y="125" text-anchor="middle" fill="#f4f1e8" transform="rotate(90 975 125)">DET</text>${{path}}${{startMarker}}${{endMarker}}<text x="500" y="28" text-anchor="middle" fill="#f4f1e8" font-size="22">${{esc(e.possession_before)}} ${{direction}} · ${{esc(directionCopy(e.transition_direction_semantics))}}</text></svg>`;
}}
function facts(items){{return `<dl class="facts">${{items.map(([k,v])=>`<dt>${{esc(k)}}</dt><dd>${{esc(val(v))}}</dd>`).join("")}}</dl>`}}
function renderReplay(e){{
  const transition=[["Action",e.primary_action],["Yards",e.yards_gained],["Air / YAC",`${{val(e.air_yards)}} / ${{val(e.yards_after_catch)}}`],["Return yards",e.return_yards],["Pass",e.pass_length&&e.pass_location?`${{e.pass_length}} ${{e.pass_location}}`:"—"],["Run",e.run_location?`${{e.run_location}} ${{e.run_gap||""}}`:"—"],["Kick distance",e.kick_distance],["Penalty",e.penalty_type?`${{e.penalty_team||""}} ${{e.penalty_type}} ${{val(e.penalty_yards)}} yd`:"—"],["Review",e.review_result]];
  const impactTag=hasMaterialImpact(e)?`<span class="impact-flag">${{esc(impactCopy(e))}}</span>`:"";
  $("#panel-replay").innerHTML=`<div class="event-heading"><div><div class="eyebrow">${{esc(e.event_id)}} · ${{esc(e.row_disposition)}} / ${{esc(e.primary_action)}}</div><h2>Replay / 比赛复盘 · ${{esc(e.possession_before||"Administrative / 管理事件")}}</h2></div><div class="clock-big">${{esc(gameClock(e))}}<br>${{esc(e.pre_away_score)}}–${{esc(e.pre_home_score)}}</div></div>
  <aside class="inline-market-card" aria-label="Compact market trend / 紧凑市场趋势"><h3>Market trend / 市场走向</h3><div class="eyebrow">T−30s → T+60s · observed trades / 实际成交</div><div class="market-chart-wrap">${{marketSvg(e,true,D.events)}}</div><div class="market-legend"><span class="poly-dal">Poly ${{D.game.away_team}}</span><span class="poly-det">Poly ${{D.game.home_team}}</span><span class="kalshi-dal">Kalshi ${{D.game.away_team}}</span><span class="kalshi-det">Kalshi ${{D.game.home_team}}</span></div><p class="market-note">Trend alignment only / 仅趋势对齐；无历史 BBO/L2。</p></aside>
  ${{fieldSvg(e)}}<div class="legend"><span class="dal">DAL normalized end</span><span class="det">DET normalized end</span><span>solid ball = source start</span><span>open ball = derived/next observed display end</span><span>not physical stadium orientation</span></div>
  <p class="raw-description">${{esc(e.description)}}</p><div class="tags">${{tags(e).map(t=>`<span class="tag">${{esc(t)}}</span>`).join("")||'<span class="empty">No outcome tag / 无结果标签</span>'}} ${{impactTag}}</div>
  <div class="state-grid"><section><h3>Pre-state / 事件前状态</h3>${{facts([["Score / 比分",`${{e.pre_away_score}}–${{e.pre_home_score}}`],["Source posteam / 源进攻方",e.possession_before],["Posteam semantics / 进攻方语义",e.posteam_semantics],["Down / distance / 档数距离",e.down?`${{e.down}} & ${{e.distance}}`:"—"],["Field / 场地",e.pre_yardline],["Normalized direction / 规范方向",e.offense_direction],["Direction semantics / 方向语义",directionCopy(e.transition_direction_semantics)],["Timeouts / 暂停",`${{e.away_timeouts_remaining}} / ${{e.home_timeouts_remaining}}`]])}}</section><section><h3>Transition / 状态转移</h3>${{facts(transition)}}</section><section><h3>Post / next observed / 事件后与下一观测</h3>${{facts([["Score / 比分",`${{e.post_away_score}}–${{e.post_home_score}}`],["Next posteam / 下一进攻方",e.next_observed_possession],["Next structured field / 下一结构化位置",e.next_observed_yardline],["Display endpoint / 显示终点",e.visual_end_field_coordinate_0_100],["Endpoint semantics / 终点语义",endpointCopy(e.visual_end_semantics)],["Participation / 参与层",e.participation_status],["Diagnostic WP before / 诊断胜率前",pp(e.home_wp_before_diagnostic)],["Diagnostic WP Δ / 诊断胜率变化",pp(e.home_wp_delta_diagnostic)]])}}</section></div>`;
}}
function renderPlayers(e){{
  const rows=D.players.filter(p=>p.event_id===e.event_id);
  const injuries=D.injuries.filter(p=>p.event_id===e.event_id);
  $("#panel-players").innerHTML=`<h2>Players on this fact</h2><p class="raw-description">${{esc(e.description)}}</p>${{rows.length?`<table><thead><tr><th>Role</th><th>Player</th><th>Team</th><th>Position / #</th><th>Evidence</th></tr></thead><tbody>${{rows.map(p=>`<tr><td>${{esc(p.role)}}</td><td>${{esc(p.player_name)}}<br><code>${{esc(p.player_id)}}</code></td><td>${{esc(p.team)}}</td><td>${{esc(p.position)}} / ${{esc(p.jersey_number)}}</td><td>${{esc(p.evidence_class)}}</td></tr>`).join("")}}</tbody></table>`:'<p class="empty">No participation or explicit player role on this administrative row.</p>'}}<h2>Injury evidence on this event</h2>${{injuries.length?`<table><tbody>${{injuries.map(i=>`<tr><td>${{esc(i.evidence_type)}}</td><td>${{esc(i.player_name||i.source_player_name)}}</td><td>${{esc(i.status)}}</td><td>${{esc(i.evidence_grade)}}</td></tr>`).join("")}}</tbody></table>`:'<p class="empty">No source-declared injury evidence on this event. Participation changes are not relabeled as injuries.</p>'}}`;
}}
function renderRegistry(e){{
  const hits=D.factor_hits.filter(h=>h.event_id===e.event_id);
  $("#panel-registry").innerHTML=`<h2>Deterministic factor hits</h2><p>These are registry predicates applied to the canonical fact—not hand-written labels in the report.</p>${{hits.length?`<table><thead><tr><th>Factor</th><th>Version</th><th>Predicate evidence</th></tr></thead><tbody>${{hits.map(h=>`<tr><td>${{esc(h.factor_id)}}</td><td>${{esc(h.factor_version)}}</td><td><code>${{esc(h.predicate_evidence)}}</code></td></tr>`).join("")}}</tbody></table>`:'<p class="empty">No active factor matched this row.</p>'}}<h2>Registry coverage in this game</h2><table><thead><tr><th>Factor</th><th>Status</th><th>Events</th><th>Availability</th></tr></thead><tbody>${{D.factor_coverage.map(f=>`<tr><td>${{esc(f.factor_id)}}</td><td>${{esc(f.status)}}</td><td>${{esc(f.event_count)}}</td><td>${{esc(f.availability_status)}}</td></tr>`).join("")}}</tbody></table>`;
}}
function renderAudit(e){{
  const g=D.game;
  $("#panel-audit").innerHTML=`<h2>Full-game reconciliation</h2><div class="audit-grid"><div><strong>${{g.raw_rows_preserved}} / ${{g.raw_row_count}}</strong>raw rows preserved</div><div><strong>${{g.participation_present}}</strong>rows with participation</div><div><strong>${{g.source_time_present}}</strong>rows with source time</div><div><strong>${{g.injury_evidence_count}}</strong>source-declared injury facts</div></div><h2>Source-declared injuries and returns</h2>${{D.injuries.length?`<table><thead><tr><th>Clock</th><th>Player</th><th>Evidence</th><th>Identity</th></tr></thead><tbody>${{D.injuries.map(i=>{{const ev=D.events.find(e=>e.event_id===i.event_id);return `<tr><td>${{esc(gameClock(ev))}}</td><td>${{esc(i.player_name||i.source_player_name)}} · ${{esc(i.team)}} #${{esc(i.jersey_number)}}</td><td>${{esc(i.evidence_type)}}<br>${{esc(i.raw_evidence_text)}}</td><td>${{esc(i.identity_status)}}</td></tr>`}}).join("")}}</tbody></table>`:'<p class="empty">No source-declared injury evidence.</p>'}}<h2>Selected event lineage</h2>${{facts([["Event",e.event_id],["Source time",e.source_time_utc],["Time semantics",e.source_time_semantics],["PBP hash",e.pbp_source_sha256],["Participation hash",e.participation_source_sha256],["Registry hash",g.registry_sha256]])}}<p class="claim"><b>Claim boundary:</b> sports fact description only. No market reaction, latency, causality, execution, or alpha conclusion appears in this report. Observed roster changes only bound participation differences between snaps; they are not exact substitution or injury times.</p>`;
}}
function renderDetail(){{
  if(!selected) return;
  renderCurrentStrip(selected);
  renderReplay(selected); renderMarket(selected); renderPlayers(selected); renderRegistry(selected); renderAudit(selected);
}}
document.querySelectorAll(".detail-tabs button").forEach(button=>button.onclick=()=>{{
  document.querySelectorAll(".detail-tabs button").forEach(b=>b.setAttribute("aria-selected",String(b===button)));
  document.querySelectorAll(".panel").forEach(p=>p.classList.toggle("active",p.id===`panel-${{button.dataset.tab}}`));
}});
["event-search","quarter-filter","action-filter","impact-threshold-filter"].forEach(id=>$("#"+id).addEventListener(id==="event-search"?"input":"change",applyFilters));
$("#event-index-toggle").addEventListener("click",()=>{{
  const workspace=$("#workspace"), button=$("#event-index-toggle");
  const collapsed=workspace.classList.toggle("navigator-collapsed");
  button.setAttribute("aria-expanded",String(!collapsed));
  button.textContent=collapsed?"Open navigator / 打开事件导航":"Hide navigator / 收起事件导航";
}});
document.addEventListener("keydown",event=>{{
  const tag=event.target?.tagName;
  if(["INPUT","SELECT","TEXTAREA"].includes(tag)||event.target?.isContentEditable) return;
  if(event.key==="ArrowDown"||event.key==="ArrowRight"){{
    event.preventDefault(); moveSelected(1);
  }} else if(event.key==="ArrowUp"||event.key==="ArrowLeft"){{
    event.preventDefault(); moveSelected(-1);
  }} else if(event.key==="Home"){{
    event.preventDefault(); if(filtered.length) selectEvent(filtered[0]);
  }} else if(event.key==="End"){{
    event.preventDefault(); if(filtered.length) selectEvent(filtered[filtered.length-1]);
  }}
}});
populateActions(); applyFilters(); renderMarketOverview(); syncScrubber();
</script>
</body>
</html>"""


__all__ = ["render_game_fact_review"]
