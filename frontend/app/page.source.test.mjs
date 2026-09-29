import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const pagePath = new URL("./page.tsx", import.meta.url);
const source = await readFile(pagePath, "utf8");

test("upload uses a native label instead of a nested interactive dropzone", () => {
  assert.match(source, /<label className="dropzone-label"/);
  assert.match(source, /<input[\s\S]*type="file"/);
  assert.doesNotMatch(source, /role="button"/);
  assert.doesNotMatch(source, /onKeyDown/);
});

test("file input is disabled during prediction and progress has accessible semantics", () => {
  assert.match(source, /disabled=\{loading\}/);
  assert.match(source, /role="progressbar"/);
  assert.match(source, /aria-valuenow=\{boundedValue\}/);
  assert.match(source, /aria-valuemin=\{0\}/);
  assert.match(source, /aria-valuemax=\{100\}/);
  assert.match(source, /aria-live="polite"/);
  assert.match(source, /role="alert"/);
  assert.match(source, /aria-valuetext=\{label/);
  assert.match(source, /aria-disabled=\{loading\}/);
  assert.match(source, /aria-busy=\{loading\}/);
});

test("dropzone implements drag and drop while prediction is active", () => {
  assert.match(source, /onDragEnter=\{onDragEnter\}/);
  assert.match(source, /onDragOver=\{onDragOver\}/);
  assert.match(source, /onDragLeave=\{onDragLeave\}/);
  assert.match(source, /onDrop=\{onDrop\}/);
  assert.match(source, /event\.preventDefault\(\)/);
  assert.match(source, /if \(loadingRef\.current\)/);
  assert.match(source, /dataTransfer\.files/);
});

test("prediction requests are cancellable and stale updates are ignored", () => {
  assert.match(source, /new AbortController\(\)/);
  assert.match(source, /requestId !== activeRequestIdRef\.current/);
  assert.match(source, /activeRequestIdRef\.current\+\+/);
  assert.match(source, /requestControllerRef\.current\?\.abort\(\)/);
  assert.match(source, /cancellablePredict\(/);
  assert.match(source, /requestController\.signal\.aborted/);
});

test("removing an image resets the native file input value", () => {
  assert.match(source, /fileInputRef\.current\.value = ""/);
  assert.match(source, /<label className="dropzone-label"/);
  assert.match(source, /aria-label="Remove selected image"/);
});

test("upload flow includes a truthful privacy notice", () => {
  assert.match(source, /without persistently storing/);
});
