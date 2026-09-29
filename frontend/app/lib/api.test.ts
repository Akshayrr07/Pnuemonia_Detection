/**
 * Regression tests for the frontend API client's request timeout (audit finding M1).
 *
 * `app/lib/api.ts` issued fetches with no timeout and no AbortController, so a
 * wedged or unreachable backend left the UI spinning on "Sending to backend..."
 * indefinitely with no way for the user to recover short of a page reload.
 *
 * These run on Node's built-in test runner (no new dependencies): the real
 * `api.ts` is compiled to JS with the repo's own tsc, then exercised against a
 * stubbed global fetch.
 */
import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import { healthCheck, predict } from "./api.js";

/**
 * Originals captured so each test can restore the global stub.
 *
 * The default stub THROWS, so any code path that forgets to stub `fetch` fails
 * loudly instead of reaching the real network and hanging until the OS-level
 * connect timeout -- which would stall the whole test run.
 */
const realFetch = globalThis.fetch;
const realEnv = process.env.NEXT_PUBLIC_BACKEND_URL;

// Hermetic default: an explicit, non-routable base URL plus a throwing fetch.
process.env.NEXT_PUBLIC_BACKEND_URL = "http://127.0.0.1:1";
globalThis.fetch = (async () => {
  throw new Error("unstubbed fetch: this test path must not hit the network");
}) as unknown as typeof fetch;

afterEach(() => {
  globalThis.fetch = realFetch;
  if (realEnv === undefined) {
    delete process.env.NEXT_PUBLIC_BACKEND_URL;
  } else {
    process.env.NEXT_PUBLIC_BACKEND_URL = realEnv;
  }
});

describe("healthCheck", () => {
  it("resolves the parsed body on a healthy response", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ status: "healthy", pipeline_ready: true }), {
        status: 200,
        headers: { "content-type": "application/json" },
      })) as unknown as typeof fetch;

    const result = await healthCheck();
    assert.equal(result.status, "healthy");
    assert.equal(result.pipeline_ready, true);
  });

  it("rejects when the backend reports a failure status", async () => {
    globalThis.fetch = (async () =>
      new Response("nope", { status: 503 })) as unknown as typeof fetch;

    await assert.rejects(() => healthCheck(), /Health check failed: 503/);
  });
});

describe("predict timeout", () => {
  it("passes an AbortSignal to fetch so the request can be cancelled", async () => {
    let seenSignal: AbortSignal | undefined;
    globalThis.fetch = (async (_url: string, init?: RequestInit) => {
      seenSignal = init?.signal ?? undefined;
      return new Response(JSON.stringify({}), { status: 200 });
    }) as unknown as typeof fetch;

    await predict(new File([], "x.png", { type: "image/png" })).catch(() => {});
    assert.ok(seenSignal, "fetch must receive an AbortSignal");
    assert.equal(seenSignal!.aborted, false, "signal must not be pre-aborted");
  });

  it("aborts and surfaces a timeout error when the backend never responds", async () => {
    // Use a short timeout so the test observes a real abort quickly instead of
    // waiting out the 60s production default.
    process.env.NEXT_PUBLIC_API_TIMEOUT_MS = "150";

    // A fetch that only ever settles via the signal: exactly the wedged-backend case.
    globalThis.fetch = ((_url: string, init?: RequestInit) =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(init.signal!.reason ?? new Error("aborted"));
        });
      })) as unknown as typeof fetch;

    // Without a client-side timeout the promise never settles, which would hang
    // the suite rather than fail it. Race a sentinel so the missing timeout is
    // reported as a real failure.
    const outcome = await Promise.race([
      predict(new File([], "x.png", { type: "image/png" })).then(
        () => ({ kind: "resolved" as const }),
        (err: unknown) => ({ kind: "rejected" as const, err })
      ),
      new Promise<{ kind: "hung" }>((resolve) => {
        setTimeout(() => resolve({ kind: "hung" }), 2_000).unref?.();
      }),
    ]);

    assert.notEqual(outcome.kind, "resolved", "predict must not resolve without a response");
    if (outcome.kind === "hung") {
      assert.fail(
        "predict() never settled: no request timeout is implemented, so a wedged " +
          "backend leaves the UI spinning indefinitely (audit finding M1)"
      );
    }

    const { err } = outcome as { kind: "rejected"; err: unknown };
    assert.ok(err instanceof Error, "timeout must surface as an Error");
    assert.match(err.message, /timed out|timeout/i);
    assert.match(err.message, /150ms/, "the configured timeout should be reported");
  });

  it("ignores a non-numeric or non-positive timeout override", async () => {
    process.env.NEXT_PUBLIC_API_TIMEOUT_MS = "not-a-number";
    globalThis.fetch = (async (_url: string, init?: RequestInit) => {
      // A well-behaved client always supplies a live signal.
      assert.ok(init?.signal, "fetch must receive an AbortSignal");
      return new Response(JSON.stringify({}), { status: 200 });
    }) as unknown as typeof fetch;

    await predict(new File([], "x.png", { type: "image/png" }));
  });

  it("propagates the backend error detail rather than a generic message", async () => {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ detail: "Unsupported image type." }), {
        status: 415,
        headers: { "content-type": "application/json" },
      })) as unknown as typeof fetch;

    await assert.rejects(
      () => predict(new File([], "x.png", { type: "image/png" })),
      /Unsupported image type\./
    );
  });
});
