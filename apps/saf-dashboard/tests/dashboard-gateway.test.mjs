import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import {
  createDashboardGateway,
  canonicalS3Uri,
  dashboardObjectKey,
  latestPointerKey,
  readBounded,
} from "../lib/dashboard-gateway.mjs";

function canonicalJson(value) {
  return Buffer.from(`${JSON.stringify(value)}\n`);
}

function sha256(bytes) {
  return `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
}

function fixtureStore() {
  const asset = canonicalJson({ schema: "catalog_v1", payload: { games: [] } });
  const assetSha = sha256(asset);
  const manifest = canonicalJson({
    schema: "nfl_x13_dashboard_publish_manifest_v1",
    batch_id: `sha256:${"1".repeat(64)}`,
    asset_count: 1,
    assets: [
      {
        relative_path: "catalog/catalog.json",
        logical_role: "catalog",
        object_sha256: assetSha,
        byte_length: asset.byteLength,
      },
    ],
  });
  const manifestSha = sha256(manifest);
  const manifestKey = dashboardObjectKey("", manifestSha);
  const pointer = canonicalJson({
    schema: "x13_dashboard_latest_pointer_v1",
    batch_id: `sha256:${"1".repeat(64)}`,
    manifest_sha256: manifestSha,
    manifest_key: manifestKey,
  });
  const objects = new Map([
    [latestPointerKey(""), pointer],
    [manifestKey, manifest],
    [dashboardObjectKey("", assetSha), asset],
  ]);
  return { asset, assetSha, manifestSha, objects };
}

function makeGateway(objects) {
  return createDashboardGateway({
    config: {
      endpoint: "https://s3.example.test",
      region: "us-east-1",
      bucket: "private-test",
      accessKeyId: "access",
      secretAccessKey: "secret",
      prefix: "",
    },
    fetchImpl: async (request) => {
      const key = decodeURIComponent(new URL(request.url).pathname)
        .replace(/^\/private-test\//, "");
      const body = objects.get(key);
      return body
        ? new Response(body, { status: 200 })
        : new Response("missing", { status: 404 });
    },
  });
}

test("catalog and assets are served only from the verified manifest", async () => {
  const fixture = fixtureStore();
  const gateway = makeGateway(fixture.objects);

  const response = await gateway.catalog();
  assert.equal(response.status, 200);
  assert.deepEqual(Buffer.from(await response.arrayBuffer()), fixture.asset);
  assert.equal(
    response.headers.get("cache-control"),
    "private, no-store",
  );

  const assetResponse = await gateway.asset("catalog/catalog.json", {
    batchId: `sha256:${"1".repeat(64)}`,
    manifestSha256: fixture.manifestSha,
  });
  assert.equal(assetResponse.status, 200);
  assert.equal(
    assetResponse.headers.get("cache-control"),
    "private, no-store",
  );
});

test("asset bytes are rejected when SHA-256 differs from manifest", async () => {
  const fixture = fixtureStore();
  fixture.objects.set(
    dashboardObjectKey("", fixture.assetSha),
    Buffer.from("tampered"),
  );
  const response = await makeGateway(fixture.objects).catalog();
  assert.equal(response.status, 502);
});

test("unknown batch, game, and asset paths return 404", async () => {
  const gateway = makeGateway(fixtureStore().objects);
  for (const [path, batchId] of [
    ["games/unknown/core.json", `sha256:${"1".repeat(64)}`],
    ["catalog/unknown.json", `sha256:${"1".repeat(64)}`],
    ["catalog/catalog.json", `sha256:${"9".repeat(64)}`],
  ]) {
    const response = await gateway.asset(path, {
      batchId,
      manifestSha256: fixtureStore().manifestSha,
    });
    assert.equal(response.status, 404);
  }
});

test("unsafe and malformed paths fail closed", async () => {
  const gateway = makeGateway(fixtureStore().objects);
  const manifestSha256 = fixtureStore().manifestSha;
  for (const path of ["../catalog/catalog.json", "/catalog/catalog.json", ""]) {
    const response = await gateway.asset(path, {
      batchId: `sha256:${"1".repeat(64)}`,
      manifestSha256,
    });
    assert.equal(response.status, 404);
  }
});

test("bounded reader cancels a lying large stream at maximum plus one", async () => {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array(6));
      controller.enqueue(new Uint8Array(6));
    },
    cancel() {
      cancelled = true;
    },
  });
  const response = new Response(stream, {
    headers: { "content-length": "1" },
  });
  await assert.rejects(() => readBounded(response, 8), /size limit/);
  assert.equal(cancelled, true);
});

test("bounded reader handles a bodyless response without an unbounded fallback", async () => {
  const bytes = await readBounded(new Response(null), 8);
  assert.equal(bytes.byteLength, 0);
  await assert.rejects(
    () =>
      readBounded(
        new Response(null, { headers: { "content-length": "9" } }),
        8,
      ),
    /size limit/,
  );
});

test("S3 canonical URI uses RFC3986 byte encoding without collapsing slashes", () => {
  assert.equal(
    canonicalS3Uri(
      "https://s3.example.test/storage/v1/s3",
      "bucket name",
      "prefix/space !'()*~/%/雪.json",
    ),
    "/storage/v1/s3/bucket%20name/prefix/space%20%21%27%28%29%2A~/%25/%E9%9B%AA.json",
  );
});

test("catalog rejects deprecated role instead of logical_role", async () => {
  const fixture = fixtureStore();
  const manifest = JSON.parse(fixture.objects.get(
    dashboardObjectKey("", fixture.manifestSha),
  ));
  manifest.assets[0].role = manifest.assets[0].logical_role;
  delete manifest.assets[0].logical_role;
  const manifestBytes = canonicalJson(manifest);
  const replacementSha = sha256(manifestBytes);
  const replacementKey = dashboardObjectKey("", replacementSha);
  fixture.objects.set(replacementKey, manifestBytes);
  fixture.objects.set(
    latestPointerKey(""),
    canonicalJson({
      schema: "x13_dashboard_latest_pointer_v1",
      batch_id: manifest.batch_id,
      manifest_sha256: replacementSha,
      manifest_key: replacementKey,
    }),
  );
  assert.equal((await makeGateway(fixture.objects).catalog()).status, 502);
});

test("manifest-pinned A session can lazy-load A after latest moves to B", async () => {
  const a = fixtureStore();
  const bAsset = canonicalJson({ schema: "catalog_v1", payload: { games: ["B"] } });
  const bAssetSha = sha256(bAsset);
  const bManifest = canonicalJson({
    schema: "nfl_x13_dashboard_publish_manifest_v1",
    batch_id: `sha256:${"2".repeat(64)}`,
    asset_count: 1,
    assets: [{
      relative_path: "catalog/catalog.json",
      logical_role: "catalog",
      object_sha256: bAssetSha,
      byte_length: bAsset.byteLength,
    }],
  });
  const bManifestSha = sha256(bManifest);
  const bManifestKey = dashboardObjectKey("", bManifestSha);
  a.objects.set(bManifestKey, bManifest);
  a.objects.set(dashboardObjectKey("", bAssetSha), bAsset);
  const gateway = makeGateway(a.objects);
  assert.equal((await gateway.catalog()).status, 200);

  a.objects.set(
    latestPointerKey(""),
    canonicalJson({
      schema: "x13_dashboard_latest_pointer_v1",
      batch_id: `sha256:${"2".repeat(64)}`,
      manifest_sha256: bManifestSha,
      manifest_key: bManifestKey,
    }),
  );
  const oldAsset = await gateway.asset("catalog/catalog.json", {
    batchId: `sha256:${"1".repeat(64)}`,
    manifestSha256: a.manifestSha,
  });
  assert.equal(oldAsset.status, 200);
  assert.deepEqual(Buffer.from(await oldAsset.arrayBuffer()), a.asset);
});
