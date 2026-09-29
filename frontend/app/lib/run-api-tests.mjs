#!/usr/bin/env node
/**
 * Zero-dependency test runner for the frontend API client.
 *
 * Compiles `app/lib/api.ts` and `app/lib/api.test.ts` with the repo's own
 * TypeScript compiler, then runs them on Node's built-in test runner. Adding a
 * test framework (Vitest/Jest) would pull a large dependency tree into this
 * repo for a handful of unit tests, so this script does the minimal work:
 * transpile with tsc, then run.
 *
 * The build directory is created *inside* the frontend folder (not in the OS
 * temp dir) so Node's normal upward `node_modules` resolution finds
 * `@types/node` and tsc without copying the dependency tree.
 *
 * Usage: node app/lib/run-api-tests.mjs
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const frontendRoot = join(here, "..", "..");
const buildDir = join(frontendRoot, ".api-test-build");

try {
  mkdirSync(buildDir, { recursive: true });
  const tsconfigPath = join(buildDir, "tsconfig.test.json");
  const files = [join(frontendRoot, "app", "lib", "api.ts"), join(frontendRoot, "app", "lib", "api.test.ts")];

  // Minimal tsconfig: emit JS for the two modules under test only. The project
  // config is noEmit + Next plugin + path aliases, none of which apply here.
  writeFileSync(
    tsconfigPath,
    JSON.stringify(
      {
        compilerOptions: {
          target: "ES2022",
          module: "ESNext",
          moduleResolution: "bundler",
          lib: ["ES2022", "DOM"],
          strict: true,
          esModuleInterop: true,
          skipLibCheck: true,
          types: ["node"],
          typeRoots: [join(frontendRoot, "node_modules", "@types")],
          outDir: buildDir,
          rootDir: join(frontendRoot, "app", "lib"),
        },
        files,
      },
      null,
      2
    )
  );

  // Invoke the repo's local tsc binary directly; `npx tsc` may try to reach the
  // network, which hangs in sandboxed/offline CI environments.
  const tscBin = join(frontendRoot, "node_modules", "typescript", "bin", "tsc");
  if (!existsSync(tscBin)) {
    console.error("TypeScript not installed. Run `npm ci` in frontend/ first.");
    process.exit(1);
  }
  const tsc = spawnSync(process.execPath, [tscBin, "-p", tsconfigPath], {
    cwd: frontendRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  if (tsc.status !== 0) {
    console.error("tsc failed:\n" + (tsc.stdout ?? "") + (tsc.stderr ?? ""));
    process.exit(1);
  }

  // The compiled output is ESM (module: ESNext); mark the build dir as ESM so
  // `node --test` loads the .js files as modules rather than CommonJS.
  writeFileSync(join(buildDir, "package.json"), JSON.stringify({ type: "module" }, null, 2));

  const compiled = join(buildDir, "api.test.js");
  if (!existsSync(compiled)) {
    console.error(`tsc reported success but ${compiled} is missing`);
    process.exit(1);
  }

  // Bound the run: a regression that reintroduces an unbounded request would
  // otherwise hang CI until the job timeout.
  const result = spawnSync(process.execPath, ["--test-timeout=30000", "--test", compiled], {
    cwd: frontendRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  process.stdout.write(result.stdout ?? "");
  process.stderr.write(result.stderr ?? "");
  process.exit(result.status ?? 1);
} finally {
  rmSync(buildDir, { recursive: true, force: true });
}
