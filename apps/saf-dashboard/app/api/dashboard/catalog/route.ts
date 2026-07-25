import { gatewayFromEnvironment } from "@/lib/dashboard-gateway.mjs";
import {
  ChatGPTApiAuthenticationError,
  requireChatGPTUser,
} from "@/app/chatgpt-auth";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    await requireChatGPTUser("/api/dashboard/catalog", { api: true });
    return await gatewayFromEnvironment().catalog();
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
