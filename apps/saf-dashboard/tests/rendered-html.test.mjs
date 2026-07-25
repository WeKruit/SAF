import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

async function requestSite(path = "/", { authenticated = true } = {}) {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request(`http://localhost${path}`, {
      headers: {
        accept: "text/html",
        ...(authenticated
          ? { "oai-authenticated-user-email": "researcher@example.test" }
          : {}),
      },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

async function render(path = "/") {
  return requestSite(path);
}

test("page redirects unauthenticated users and both data APIs reject them", async () => {
  const page = await requestSite("/", { authenticated: false });
  assert.ok([302, 303, 307, 308].includes(page.status));
  assert.match(page.headers.get("location") ?? "", /signin-with-chatgpt/);

  for (const path of [
    "/api/dashboard/catalog",
    "/api/dashboard/assets/Y2F0YWxvZy9jYXRhbG9nLmpzb24",
  ]) {
    const response = await requestSite(path, { authenticated: false });
    assert.ok([401, 403].includes(response.status));
  }
});

test("server-renders the SAF evidence workspace, not starter metadata", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>SAF \| X-13 Research Workspace<\/title>/i);
  assert.match(html, /Game State × Market Reaction/);
  assert.match(html, /PRELIMINARY_SOURCE_TIME_ONLY/);
  assert.match(html, /20-game evidence index/i);
  assert.doesNotMatch(html, /Your site is taking shape|Building your site/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton/i);
});

test("starter preview and dependency are removed", async () => {
  const [page, layout, packageJson, appFiles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readdir(new URL("../app/", import.meta.url)),
  ]);

  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.doesNotMatch(page, /SkeletonPreview|codex-preview/);
  assert.doesNotMatch(layout, /Starter Project|Geist|favicon\.svg/);
  assert.ok(!appFiles.includes("_sites-preview"));
});

test("browser source does not contain the S3 secret variable", async () => {
  const browserFiles = [
    "../app/page.tsx",
    "../app/dashboard-workspace.tsx",
    "../lib/dashboard-client.mjs",
  ];
  const source = (
    await Promise.all(
      browserFiles.map((path) => readFile(new URL(path, import.meta.url), "utf8")),
    )
  ).join("\n");
  assert.doesNotMatch(source, /PM_DATA_S3_SECRET_ACCESS_KEY/);
  assert.doesNotMatch(source, /localStorage|sessionStorage|caches\.open|CacheStorage/);
});

test("every critical control is represented in the rendered workspace", async () => {
  const response = await render();
  const html = await response.text();
  for (const label of [
    "Game index",
    "Play scrubber",
    "Market family",
    "Venue",
    "Market state",
    "Logical market",
    "Delay scenario",
    "Horizon",
    "Include ambiguous",
    "Include contaminated",
    "Audit & lineage",
  ]) {
    const renderedLabel = label.replace("&", "(?:&|&amp;)");
    assert.match(html, new RegExp(renderedLabel, "i"));
  }
});

test("closed audit drawer does not leave hidden controls focusable", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /aria-expanded="false"/);
  assert.doesNotMatch(html, />Copy<\/button>/);
  assert.doesNotMatch(html, /class="stable-id"/);
});

test("market semantics have a screen-reader description and structured tables", async () => {
  const response = await render();
  const html = await response.text();
  assert.match(html, /aria-describedby="market-path-description"/);
  assert.match(html, /<table[^>]*class="accessible-evidence-table"/);
  assert.match(html, /<caption>Trade range evidence<\/caption>/);
  assert.match(html, /<caption>Kalshi BBO OHLC evidence<\/caption>/);
  for (const heading of ["Source time", "Outcome", "Value", "Status", "Bid OHLC", "Ask OHLC"]) {
    assert.match(html, new RegExp(`>${heading}<`));
  }
});

test("workspace uses abortable generation guards for game, market, and association requests", async () => {
  const source = await readFile(
    new URL("../app/dashboard-workspace.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /gameRequestGeneration/);
  assert.match(source, /marketRequestGeneration/);
  assert.match(source, /associationRequestGeneration/);
  assert.ok((source.match(/new AbortController\(\)/g) ?? []).length >= 3);
});

test("BBO source-order ambiguity is explicit in summary, plot, and accessible table", async () => {
  const source = await readFile(
    new URL("../app/dashboard-workspace.tsx", import.meta.url),
    "utf8",
  );
  assert.match(source, /ambiguousPointCount/);
  assert.match(source, /point\.status === "ambiguous"/);
  assert.match(source, /source order ambiguous/i);
  assert.match(source, /no within-interval order inferred/i);
});

test("critical touch controls declare a coarse-pointer 44px target", async () => {
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(css, /@media\s*\(pointer:\s*coarse\)/);
  assert.match(css, /min-height:\s*44px/);
  assert.match(css, /min-width:\s*44px/);
});

test("no external chart or image CDN is referenced", async () => {
  const files = await Promise.all(
    [
      "../app/page.tsx",
      "../app/dashboard-workspace.tsx",
      "../app/globals.css",
    ].map((path) => readFile(new URL(path, import.meta.url), "utf8")),
  );
  assert.doesNotMatch(files.join("\n"), /cdn|chart\.js|recharts|<svg|\.svg\b/i);
});
