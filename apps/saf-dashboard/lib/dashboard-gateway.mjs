const DASHBOARD_PREFIX = "dashboard/x13";
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/;
const MAX_POINTER_BYTES = 64 * 1024;
const MAX_OBJECT_BYTES = 8 * 1024 * 1024;
const encoder = new TextEncoder();

function jsonError(status, message) {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "private, no-store",
      "x-content-type-options": "nosniff",
    },
  });
}

function cleanPrefix(prefix) {
  const value = prefix ?? "prediction-market";
  if (value === "") return "";
  if (
    typeof value !== "string" ||
    value.startsWith("/") ||
    value.includes("\\") ||
    value.split("/").some((part) => !part || part === "." || part === "..")
  ) {
    throw new Error("Invalid dashboard object prefix");
  }
  return value;
}

function joinKey(prefix, suffix) {
  const root = cleanPrefix(prefix);
  return root ? `${root}/${suffix}` : suffix;
}

export function dashboardObjectKey(prefix, objectSha256) {
  if (!SHA256_PATTERN.test(objectSha256)) {
    throw new Error("Invalid dashboard object SHA-256");
  }
  const digest = objectSha256.slice("sha256:".length);
  return joinKey(
    prefix,
    `${DASHBOARD_PREFIX}/objects/sha256/${digest.slice(0, 2)}/${digest}.json`,
  );
}

export function latestPointerKey(prefix) {
  return joinKey(prefix, `${DASHBOARD_PREFIX}/published/latest.json`);
}

function hex(bytes) {
  return Array.from(new Uint8Array(bytes), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

async function digest(bytes) {
  return `sha256:${hex(await crypto.subtle.digest("SHA-256", bytes))}`;
}

async function hmac(key, value) {
  const material = await crypto.subtle.importKey(
    "raw",
    typeof key === "string" ? encoder.encode(key) : key,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  return new Uint8Array(
    await crypto.subtle.sign("HMAC", material, encoder.encode(value)),
  );
}

function rfc3986EncodeSegment(value) {
  return Array.from(encoder.encode(value), (byte) => {
    const unreserved =
      (byte >= 0x41 && byte <= 0x5a) ||
      (byte >= 0x61 && byte <= 0x7a) ||
      (byte >= 0x30 && byte <= 0x39) ||
      byte === 0x2d ||
      byte === 0x2e ||
      byte === 0x5f ||
      byte === 0x7e;
    return unreserved
      ? String.fromCharCode(byte)
      : `%${byte.toString(16).toUpperCase().padStart(2, "0")}`;
  }).join("");
}

export function canonicalS3Uri(endpointValue, bucket, key) {
  const endpoint = new URL(endpointValue);
  const endpointPath = endpoint.pathname
    .split("/")
    .map((segment) => rfc3986EncodeSegment(decodeURIComponent(segment)))
    .join("/")
    .replace(/\/$/, "");
  return `${endpointPath}/${rfc3986EncodeSegment(bucket)}/${key
    .split("/")
    .map(rfc3986EncodeSegment)
    .join("/")}`;
}

async function signedRequest(config, key, now = new Date()) {
  const endpoint = new URL(config.endpoint);
  if (endpoint.protocol !== "https:") {
    throw new Error("Dashboard S3 endpoint must use HTTPS");
  }
  const canonicalUri = canonicalS3Uri(
    config.endpoint,
    config.bucket,
    key,
  );
  const url = new URL(endpoint.origin + canonicalUri);

  const timestamp = now
    .toISOString()
    .replace(/[:-]|\.\d{3}/g, "")
    .replace("Z", "Z");
  const shortDate = timestamp.slice(0, 8);
  const payloadHash =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
  const canonicalHeaders =
    `host:${url.host}\n` +
    `x-amz-content-sha256:${payloadHash}\n` +
    `x-amz-date:${timestamp}\n`;
  const signedHeaders = "host;x-amz-content-sha256;x-amz-date";
  const canonicalRequest = [
    "GET",
    canonicalUri,
    "",
    canonicalHeaders,
    signedHeaders,
    payloadHash,
  ].join("\n");
  const scope = `${shortDate}/${config.region}/s3/aws4_request`;
  const stringToSign = [
    "AWS4-HMAC-SHA256",
    timestamp,
    scope,
    (await digest(encoder.encode(canonicalRequest))).slice(7),
  ].join("\n");
  const dateKey = await hmac(`AWS4${config.secretAccessKey}`, shortDate);
  const regionKey = await hmac(dateKey, config.region);
  const serviceKey = await hmac(regionKey, "s3");
  const signingKey = await hmac(serviceKey, "aws4_request");
  const signature = hex(await hmac(signingKey, stringToSign));

  return new Request(url, {
    headers: {
      authorization:
        `AWS4-HMAC-SHA256 Credential=${config.accessKeyId}/${scope}, ` +
        `SignedHeaders=${signedHeaders}, Signature=${signature}`,
      "x-amz-content-sha256": payloadHash,
      "x-amz-date": timestamp,
    },
  });
}

export async function readBounded(response, maximumBytes) {
  if (!response.ok) {
    throw Object.assign(new Error("Private dashboard object is unavailable"), {
      status: response.status,
    });
  }
  const statedLengthValue = response.headers.get("content-length");
  const statedLength =
    statedLengthValue === null ? null : Number(statedLengthValue);
  if (
    statedLength !== null &&
    Number.isFinite(statedLength) &&
    statedLength > maximumBytes
  ) {
    throw new Error("Private dashboard object exceeds its size limit");
  }
  if (!response.body) return new Uint8Array(0);
  const reader = response.body.getReader();
  const chunks = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!(value instanceof Uint8Array)) {
        throw new Error("Private dashboard object returned non-byte data");
      }
      total += value.byteLength;
      if (total > maximumBytes) {
        await reader.cancel("dashboard object exceeds size limit");
        throw new Error("Private dashboard object exceeds its size limit");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

function parseObject(bytes, label) {
  try {
    const value = JSON.parse(new TextDecoder().decode(bytes));
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error();
    }
    return value;
  } catch {
    throw new Error(`${label} is not valid JSON`);
  }
}

function isSafeAssetPath(path) {
  return (
    typeof path === "string" &&
    path.length > 0 &&
    !path.startsWith("/") &&
    !path.includes("\\") &&
    path.split("/").every((part) => part && part !== "." && part !== "..")
  );
}

function responseFor(bytes, manifestSha256) {
  return new Response(bytes, {
    status: 200,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "private, no-store",
      "x-content-type-options": "nosniff",
      "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
      "x-dashboard-manifest-sha256": manifestSha256,
    },
  });
}

export function createDashboardGateway({ config, fetchImpl = fetch }) {
  const verifiedManifests = new Map();

  async function getObject(key, maximumBytes) {
    const request = await signedRequest(config, key);
    return readBounded(await fetchImpl(request), maximumBytes);
  }

  async function loadManifestByHash(manifestSha256, expectedBatchId) {
    if (!SHA256_PATTERN.test(manifestSha256)) {
      throw new Error("Dashboard manifest SHA-256 is invalid");
    }
    const cached = verifiedManifests.get(manifestSha256);
    if (cached) {
      if (
        expectedBatchId &&
        cached.manifest.batch_id !== expectedBatchId
      ) {
        throw Object.assign(new Error("Dashboard batch is unavailable"), {
          status: 404,
        });
      }
      return cached;
    }
    const manifestKey = dashboardObjectKey(config.prefix, manifestSha256);
    const manifestBytes = await getObject(manifestKey, MAX_OBJECT_BYTES);
    if ((await digest(manifestBytes)) !== manifestSha256) {
      throw new Error("Dashboard manifest SHA-256 mismatch");
    }
    const manifest = parseObject(manifestBytes, "Dashboard publish manifest");
    if (expectedBatchId && manifest.batch_id !== expectedBatchId) {
      throw Object.assign(new Error("Dashboard batch is unavailable"), {
        status: 404,
      });
    }
    if (
      manifest.schema !== "nfl_x13_dashboard_publish_manifest_v1" ||
      !SHA256_PATTERN.test(manifest.batch_id) ||
      !Array.isArray(manifest.assets) ||
      manifest.asset_count !== manifest.assets.length
    ) {
      throw new Error("Dashboard publish manifest identity is invalid");
    }
    const assets = new Map();
    for (const descriptor of manifest.assets) {
      if (
        !descriptor ||
        !isSafeAssetPath(descriptor.relative_path) ||
        typeof descriptor.logical_role !== "string" ||
        !descriptor.logical_role ||
        !SHA256_PATTERN.test(descriptor.object_sha256) ||
        !Number.isInteger(descriptor.byte_length) ||
        descriptor.byte_length < 0 ||
        descriptor.byte_length > MAX_OBJECT_BYTES ||
        assets.has(descriptor.relative_path)
      ) {
        throw new Error("Dashboard publish manifest asset is invalid");
      }
      assets.set(descriptor.relative_path, descriptor);
    }
    const state = { manifest, assets, manifestSha256 };
    verifiedManifests.set(manifestSha256, state);
    return state;
  }

  async function loadLatestManifest() {
    const pointerBytes = await getObject(
      latestPointerKey(config.prefix),
      MAX_POINTER_BYTES,
    );
    const pointer = parseObject(pointerBytes, "Dashboard latest pointer");
    if (
      pointer.schema !== "x13_dashboard_latest_pointer_v1" ||
      !SHA256_PATTERN.test(pointer.manifest_sha256) ||
      !SHA256_PATTERN.test(pointer.batch_id)
    ) {
      throw new Error("Dashboard latest pointer identity is invalid");
    }
    const expectedManifestKey = dashboardObjectKey(
      config.prefix,
      pointer.manifest_sha256,
    );
    if (pointer.manifest_key !== expectedManifestKey) {
      throw new Error("Dashboard manifest key is outside the immutable store");
    }
    return loadManifestByHash(pointer.manifest_sha256, pointer.batch_id);
  }

  async function asset(path, { batchId, manifestSha256 } = {}) {
    if (!isSafeAssetPath(path)) return jsonError(404, "Asset not found");
    if (!SHA256_PATTERN.test(manifestSha256 ?? "")) {
      return jsonError(404, "Manifest snapshot not found");
    }
    try {
      const state = await loadManifestByHash(manifestSha256, batchId);
      if (batchId && batchId !== state.manifest.batch_id) {
        return jsonError(404, "Batch not found");
      }
      const descriptor = state.assets.get(path);
      if (!descriptor) return jsonError(404, "Asset not found");
      const bytes = await getObject(
        dashboardObjectKey(config.prefix, descriptor.object_sha256),
        descriptor.byte_length,
      );
      if (
        bytes.byteLength !== descriptor.byte_length ||
        (await digest(bytes)) !== descriptor.object_sha256
      ) {
        return jsonError(502, "Dashboard asset integrity check failed");
      }
      return responseFor(bytes, manifestSha256);
    } catch (error) {
      if (error?.status === 404) return jsonError(404, "Asset not found");
      return jsonError(502, "Dashboard evidence store verification failed");
    }
  }

  async function catalog() {
    try {
      const state = await loadLatestManifest();
      const catalogs = [...state.assets.values()].filter(
        (asset) => asset.logical_role === "catalog",
      );
      if (catalogs.length !== 1) {
        return jsonError(502, "Dashboard catalog identity is invalid");
      }
      return asset(catalogs[0].relative_path, {
        batchId: state.manifest.batch_id,
        manifestSha256: state.manifestSha256,
      });
    } catch {
      return jsonError(502, "Dashboard evidence store verification failed");
    }
  }

  return { asset, catalog };
}

export function dashboardConfigFromEnvironment(environment = process.env) {
  const required = {
    endpoint: environment.PM_DATA_S3_ENDPOINT,
    region: environment.PM_DATA_S3_REGION,
    bucket: environment.PM_DATA_S3_BUCKET,
    accessKeyId: environment.PM_DATA_S3_ACCESS_KEY_ID,
    secretAccessKey: environment.PM_DATA_S3_SECRET_ACCESS_KEY,
  };
  if (Object.values(required).some((value) => !value)) {
    throw new Error("Private dashboard S3 configuration is incomplete");
  }
  return {
    ...required,
    prefix: environment.PM_DATA_S3_PREFIX ?? "prediction-market",
  };
}

let environmentGateway;
export function gatewayFromEnvironment() {
  environmentGateway ??= createDashboardGateway({
    config: dashboardConfigFromEnvironment(),
  });
  return environmentGateway;
}
