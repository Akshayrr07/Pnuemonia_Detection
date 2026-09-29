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

/**
 * Default request timeout in milliseconds.
 *
 * Without a client-side bound, a wedged or unreachable backend leaves the
 * upload UI spinning on "Sending to backend..." forever with no recovery short
 * of a page reload. Set NEXT_PUBLIC_API_TIMEOUT_MS to override.
 */
const DEFAULT_REQUEST_TIMEOUT_MS = 60_000;

/**
 * Timeout applied to each request.
 *
 * Read per request (not at module load) so tests can set the env var before
 * issuing a request, and so a Next.js runtime env change is picked up.
 */
function requestTimeoutMs(): number {
  const raw = Number.parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT_MS ?? "", 10);
  return Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_REQUEST_TIMEOUT_MS;
}

/**
 * Error raised when a request exceeds the configured timeout.
 */
export class ApiTimeoutError extends Error {
  constructor(url: string, timeoutMs: number) {
    super(`Request to ${url} timed out after ${timeoutMs}ms.`);
    this.name = "ApiTimeoutError";
  }
}

/**
 * `fetch` bounded by requestTimeoutMs().
 *
 * The signal is cleared as soon as the request settles so the timer never keeps
 * a Node process (or test runner) alive after the response arrives.
 */
async function fetchWithTimeout(
  url: string,
  init: RequestInit = {}
): Promise<Response> {
  const timeoutMs = requestTimeoutMs();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    // Distinguish our timeout from a caller-supplied abort or a network error.
    if (controller.signal.aborted) {
      throw new ApiTimeoutError(url, timeoutMs);
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function healthCheck(): Promise<{
  status: string;
  pipeline_ready: boolean;
}> {
  const url = `${baseUrl()}/health`;
  const res = await fetchWithTimeout(url, { cache: "no-store" });
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

  const res = await fetchWithTimeout(url, {
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
