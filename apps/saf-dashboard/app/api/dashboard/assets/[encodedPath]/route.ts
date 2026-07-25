import { gatewayFromEnvironment } from "@/lib/dashboard-gateway.mjs";
import {
  ChatGPTApiAuthenticationError,
  requireChatGPTUser,
} from "@/app/chatgpt-auth";

export const dynamic = "force-dynamic";

function decodeAssetPath(value: string) {
  if (!/^[A-Za-z0-9_-]+$/.test(value)) return null;
  try {
    const padded = value.replace(/-/g, "+").replace(/_/g, "/").padEnd(
      Math.ceil(value.length / 4) * 4,
      "=",
    );
    const binary = atob(padded);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return null;
  }
}

export async function GET(
  request: Request,
  context: { params: Promise<{ encodedPath: string }> },
) {
  try {
    await requireChatGPTUser(new URL(request.url).pathname, { api: true });
    const { encodedPath } = await context.params;
    const path = decodeAssetPath(encodedPath);
    if (!path) {
      return Response.json(
        { error: "Asset not found" },
        { status: 404, headers: { "cache-control": "private, no-store" } },
      );
    }
    const batchId = new URL(request.url).searchParams.get("batch") ?? undefined;
    const manifestSha256 =
      new URL(request.url).searchParams.get("manifest") ?? undefined;
    return await gatewayFromEnvironment().asset(path, {
      batchId,
      manifestSha256,
    });
  } catch (error) {
    if (error instanceof ChatGPTApiAuthenticationError) {
      return Response.json(
        { error: "Authentication required" },
        { status: 401, headers: { "cache-control": "private, no-store" } },
      );
    }
    return Response.json(
      { error: "Private dashboard S3 configuration is unavailable" },
      {
        status: 503,
        headers: { "cache-control": "private, no-store" },
      },
    );
  }
}
