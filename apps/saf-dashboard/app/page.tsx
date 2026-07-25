import type { Metadata } from "next";
import { requireChatGPTUser } from "./chatgpt-auth";
import { DashboardWorkspace } from "./dashboard-workspace";

export const metadata: Metadata = {
  title: "SAF | X-13 Research Workspace",
  description:
    "Evidence-first NFL game-state and prediction-market source-time research workspace.",
};

export const dynamic = "force-dynamic";

export default async function Home() {
  await requireChatGPTUser("/");
  return <DashboardWorkspace />;
}
