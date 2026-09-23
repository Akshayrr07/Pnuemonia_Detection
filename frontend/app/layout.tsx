import { Suspense } from "react";
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pneumonia Detection — AI Chest X-ray Analysis",
  description:
    "Upload a chest X-ray image for hierarchical pneumonia analysis: Normal vs Pneumonia, then Bacterial vs Viral subtype when pneumonia is detected.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🫁</text></svg>"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
