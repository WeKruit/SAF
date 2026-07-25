import type { Metadata } from "next";
import "./globals.css";

const title = "SAF | X-13 Research Workspace";
const description =
  "Replay game state, inspect market paths, and audit immutable X-13 evidence.";

export async function generateMetadata(): Promise<Metadata> {
  const configuredOrigin = process.env.SAF_CANONICAL_ORIGIN;
  let image: string | null = null;
  if (configuredOrigin) {
    try {
      const origin = new URL(configuredOrigin);
      if (
        origin.protocol === "https:" &&
        origin.username === "" &&
        origin.password === "" &&
        origin.pathname === "/" &&
        origin.search === "" &&
        origin.hash === ""
      ) {
        image = new URL("/og.png", origin).toString();
      }
    } catch {
      image = null;
    }
  }

  return {
    title,
    description,
    openGraph: image
      ? {
          title,
          description,
          images: [{ url: image, width: 1536, height: 1024, alt: title }],
        }
      : undefined,
    twitter: image
      ? {
          card: "summary_large_image",
          title,
          description,
          images: [image],
        }
      : undefined,
  };
}

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
