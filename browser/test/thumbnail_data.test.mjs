/**
 * The pure parts of thumbnail attaching (#3115): the 640px fit, the size
 * cap, and the order-preserving "keep those with data, cap" selection.
 * `attachThumbnailData` itself needs `fetch`/`createImageBitmap`/
 * `OffscreenCanvas`, none of which Node has — not exercised here.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { exceedsSizeCap, fitWithin, selectAttached } from "../dist/test/thumbnail_data.mjs";

test("fitWithin downscales a landscape image to 640 on its longest side", () => {
  assert.deepEqual(fitWithin(1280, 640, 640), { width: 640, height: 320 });
});

test("fitWithin downscales a portrait image to 640 on its longest side", () => {
  assert.deepEqual(fitWithin(640, 1280, 640), { width: 320, height: 640 });
});

test("fitWithin downscales a square image to 640 on both sides", () => {
  assert.deepEqual(fitWithin(1000, 1000, 640), { width: 640, height: 640 });
});

test("fitWithin leaves an already-small image unchanged", () => {
  assert.deepEqual(fitWithin(300, 200, 640), { width: 300, height: 200 });
});

test("fitWithin never upscales an image already within the cap", () => {
  assert.deepEqual(fitWithin(640, 640, 640), { width: 640, height: 640 });
});

test("exceedsSizeCap is true over 200KB", () => {
  assert.equal(exceedsSizeCap(200 * 1024 + 1), true);
});

test("exceedsSizeCap is false at or under 200KB", () => {
  assert.equal(exceedsSizeCap(200 * 1024), false);
  assert.equal(exceedsSizeCap(1024), false);
});

const entry = (n) => ({ url: `https://example.com/${n}`, image_url: `https://example.com/${n}.jpg`, title: `Story ${n}` });

test("selectAttached keeps order and drops entries without data", () => {
  const entries = [entry(1), entry(2), entry(3)];
  const images = ["data:image/jpeg;base64,aaa", undefined, "data:image/jpeg;base64,ccc"];
  const attached = selectAttached(entries, images);
  assert.deepEqual(attached, [
    { ...entry(1), image: "data:image/jpeg;base64,aaa" },
    { ...entry(3), image: "data:image/jpeg;base64,ccc" },
  ]);
});

test("selectAttached caps at the given limit, in order", () => {
  const entries = [entry(1), entry(2), entry(3), entry(4)];
  const images = entries.map((e) => `data:image/jpeg;base64,${e.title}`);
  const attached = selectAttached(entries, images, 2);
  assert.deepEqual(attached.map((e) => e.url), [entry(1).url, entry(2).url]);
});

test("selectAttached returns nothing when no entry got data", () => {
  const entries = [entry(1), entry(2)];
  const images = [undefined, undefined];
  assert.deepEqual(selectAttached(entries, images), []);
});
