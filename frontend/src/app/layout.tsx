import type { Metadata } from "next";
import "./globals.css";
import AppShell from "@/components/AppShell";

export const metadata: Metadata = {
  title: "RAGForge — Knowledge Engineering Workspace",
  description:
    "Automated domain-specific RAG knowledge base builder: source intelligence, provenance-rich chunks, real retrieval evaluation.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
