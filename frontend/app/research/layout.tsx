import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Research Results — Pneumonia Detection",
  description:
    "Model evaluation results for the hierarchical pneumonia detection system: three-class, binary pneumonia, and bacterial-vs-viral subtype classification.",
};

export default function ResearchLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
