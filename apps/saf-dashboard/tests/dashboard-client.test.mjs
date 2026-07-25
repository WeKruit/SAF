import assert from "node:assert/strict";
import test from "node:test";

import {
  DashboardDataClient,
  encodeAssetPath,
} from "../lib/dashboard-client.mjs";

const catalog = {
  batch_id: `sha256:${"1".repeat(64)}`,
  payload: {
    games: [
      {
        game_id: "2025_14_DAL_DET",
        assets: {
          core: {
            relative_path: "games/2025_14_DAL_DET/core/core.json",
            object_sha256: `sha256:${"2".repeat(64)}`,
          },
          contracts: {
            relative_path: "games/2025_14_DAL_DET/contracts/contracts.json",
            object_sha256: `sha256:${"3".repeat(64)}`,
          },
          associations: {
            relative_path:
              "games/2025_14_DAL_DET/associations/associations.json",
            object_sha256: `sha256:${"4".repeat(64)}`,
          },
        },
      },
    ],
  },
};
const manifestSha256 = `sha256:${"9".repeat(64)}`;

function fakeFetch(responses, calls) {
  return async (url) => {
    calls.push(String(url));
    const response = responses.shift();
    if (!response) throw new Error("unexpected request");
    return new Response(JSON.stringify(response), {
      status: 200,
      headers: {
        "content-type": "application/json",
        "x-dashboard-manifest-sha256": manifestSha256,
      },
    });
  };
}

test("catalog is fetched before a selected game's core and nothing else", async () => {
  const calls = [];
  const core = { payload: { game_id: "2025_14_DAL_DET", events: [] } };
  const client = new DashboardDataClient({
    fetchImpl: fakeFetch([catalog, core], calls),
  });

  const selection = await client.selectGame("2025_14_DAL_DET");

  assert.equal(selection.payload.game_id, "2025_14_DAL_DET");
  assert.deepEqual(calls, [
    "/api/dashboard/catalog",
    `/api/dashboard/assets/${encodeAssetPath(
      "games/2025_14_DAL_DET/core/core.json",
    )}?batch=${encodeURIComponent(catalog.batch_id)}&manifest=${encodeURIComponent(
      manifestSha256,
    )}`,
  ]);
});

test("slow game A cannot overwrite fast game B or its stable selection", async () => {
  const raceCatalog = {
    batch_id: catalog.batch_id,
    payload: {
      games: ["A", "B"].map((game_id) => ({
        game_id,
        assets: {
          core: { relative_path: `games/${game_id}/core.json` },
          contracts: { relative_path: `games/${game_id}/contracts.json` },
        },
      })),
    },
  };
  const resolvers = new Map();
  const fetchImpl = async (url, options = {}) => {
    if (url === "/api/dashboard/catalog") {
      return new Response(JSON.stringify(raceCatalog), {
        headers: { "x-dashboard-manifest-sha256": manifestSha256 },
      });
    }
    return new Promise((resolve, reject) => {
      resolvers.set(String(url), { resolve, reject, signal: options.signal });
    });
  };
  const client = new DashboardDataClient({ fetchImpl });
  const slowA = client.selectGame("A");
  await new Promise((resolve) => setTimeout(resolve, 0));
  const fastB = client.selectGame("B");
  await new Promise((resolve) => setTimeout(resolve, 0));
  const bEntry = [...resolvers.entries()].find(([url]) =>
    url.includes(encodeAssetPath("games/B/core.json")),
  )[1];
  bEntry.resolve(
    new Response(JSON.stringify({ payload: { game_id: "B" } }), {
      headers: { "x-dashboard-manifest-sha256": manifestSha256 },
    }),
  );
  assert.equal((await fastB).payload.game_id, "B");
  const aEntry = [...resolvers.entries()].find(([url]) =>
    url.includes(encodeAssetPath("games/A/core.json")),
  )[1];
  aEntry.resolve(
    new Response(JSON.stringify({ payload: { game_id: "A" } }), {
      headers: { "x-dashboard-manifest-sha256": manifestSha256 },
    }),
  );
  await assert.rejects(slowA, /superseded/i);
  assert.equal(client.game.game_id, "B");
});

test("market asset loads only after the logical market is selected", async () => {
  const calls = [];
  const core = { payload: { game_id: "2025_14_DAL_DET", events: [] } };
  const contracts = {
    payload: {
      contracts: [
        {
          logical_market_id: "polymarket:dal-det:winner",
          market_asset: {
            relative_path: "games/2025_14_DAL_DET/markets/winner.json",
          },
        },
      ],
    },
  };
  const market = { payload: { logical_market_id: "polymarket:dal-det:winner" } };
  const client = new DashboardDataClient({
    fetchImpl: fakeFetch([catalog, core, contracts, market], calls),
  });

  await client.selectGame("2025_14_DAL_DET");
  assert.equal(calls.length, 2);
  await client.loadContracts();
  assert.equal(calls.length, 3);
  await client.selectMarket("polymarket:dal-det:winner");
  assert.equal(calls.length, 4);
  assert.match(
    calls.at(-1),
    new RegExp(encodeAssetPath("games/2025_14_DAL_DET/markets/winner.json")),
  );
});

test("a verified asset is reused only inside the current client session", async () => {
  const calls = [];
  const core = { payload: { game_id: "2025_14_DAL_DET", events: [] } };
  const contracts = {
    payload: {
      contracts: [
        {
          logical_market_id: "polymarket:dal-det:winner",
          market_asset: {
            relative_path: "games/2025_14_DAL_DET/markets/winner.json",
            object_sha256: `sha256:${"5".repeat(64)}`,
          },
        },
      ],
    },
  };
  const market = { payload: { logical_market_id: "polymarket:dal-det:winner" } };
  const client = new DashboardDataClient({
    fetchImpl: fakeFetch([catalog, core, contracts, market], calls),
  });

  await client.selectGame("2025_14_DAL_DET");
  await client.loadContracts();
  const first = await client.selectMarket("polymarket:dal-det:winner");
  const second = await client.selectMarket("polymarket:dal-det:winner");

  assert.equal(second, first);
  assert.equal(calls.length, 4, "second selection must not repeat the asset fetch");
  assert.equal(globalThis.localStorage, undefined);
  assert.equal(globalThis.caches, undefined);
});

test("unknown games and logical markets are rejected before fetch", async () => {
  const calls = [];
  const client = new DashboardDataClient({
    fetchImpl: fakeFetch([catalog], calls),
  });
  await client.loadCatalog();
  await assert.rejects(() => client.selectGame("unknown"), /Unknown game/);
  await assert.rejects(
    () => client.selectMarket("unknown"),
    /Select a game|Load contracts/,
  );
  assert.equal(calls.length, 1);
});

test("slow market and association responses are superseded by newer requests", async () => {
  const core = { payload: { game_id: "2025_14_DAL_DET", events: [] } };
  const contracts = {
    payload: {
      contracts: ["A", "B"].map((id) => ({
        logical_market_id: id,
        market_asset: { relative_path: `markets/${id}.json` },
      })),
    },
  };
  const client = new DashboardDataClient({
    fetchImpl: fakeFetch([catalog, core, contracts], []),
  });
  await client.selectGame("2025_14_DAL_DET");
  await client.loadContracts();

  const pending = [];
  client.fetchImpl = async (_url, options = {}) =>
    new Promise((resolve) => pending.push({ resolve, signal: options.signal }));

  const marketA = client.selectMarket("A");
  const marketB = client.selectMarket("B");
  pending[1].resolve(
    new Response(JSON.stringify({ payload: { logical_market_id: "B" } }), {
      headers: { "x-dashboard-manifest-sha256": manifestSha256 },
    }),
  );
  assert.equal((await marketB).payload.logical_market_id, "B");
  pending[0].resolve(
    new Response(JSON.stringify({ payload: { logical_market_id: "A" } }), {
      headers: { "x-dashboard-manifest-sha256": manifestSha256 },
    }),
  );
  await assert.rejects(marketA, /superseded/i);

  pending.length = 0;
  const associationA = client.loadAssociations();
  const associationB = client.loadAssociations();
  pending[1].resolve(
    new Response(JSON.stringify({ payload: { marker: "B" } }), {
      headers: { "x-dashboard-manifest-sha256": manifestSha256 },
    }),
  );
  assert.equal((await associationB).payload.marker, "B");
  pending[0].resolve(
    new Response(JSON.stringify({ payload: { marker: "A" } }), {
      headers: { "x-dashboard-manifest-sha256": manifestSha256 },
    }),
  );
  await assert.rejects(associationA, /superseded/i);
  assert.equal(client.game.game_id, "2025_14_DAL_DET");
});
