import { PredictResponse } from "./types";

/**
 * Backend API base URL.
 * Set NEXT_PUBLIC_BACKEND_URL to point at your running backend.
 * When unset, requests go to the same origin (useful behind a proxy).
 */
function baseUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (explicit && explicit.trim().length > 0) {
    return explicit.replace(/\/+$/, "");
  }
  return "";
}

export async function healthCheck(): Promise<{
  status: string;
  pipeline_ready: boolean;
}> {
  const url = `${baseUrl()}/health`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return res.json();
}

export async function predict(
  file: File,
  onProgress?: (progress: number, stage: string) => void
): Promise<PredictResponse> {
  onProgress?.(0, "Uploading image...");

  const url = `${baseUrl()}/predict`;
  const formData = new FormData();
  formData.append("file", file);

  onProgress?.(30, "Sending to backend...");

  const res = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = `Request failed: ${res.status}`;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // response body may not be JSON
    }
    throw new Error(detail);
  }

  onProgress?.(90, "Analyzing result...");
  const data = (await res.json()) as PredictResponse;
  return data;
}
