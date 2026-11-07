/**
 * Pure helpers for an incoming_message push (#3144) — the caption text and
 * the own-device check. No jsdom needed: both are data-in/data-out.
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import { deviceCaption, isOwnDevice } from "../dist/test/incoming.mjs";

test("a Signal-origin message is captioned with its device and channel", () => {
  assert.equal(deviceCaption("signal", "Kitchen Phone"), "from Kitchen Phone · Signal");
});

test("an iOS-origin message is captioned with its device and channel", () => {
  assert.equal(deviceCaption("ios", "Kitchen iPad"), "from Kitchen iPad · iOS");
});

test("a browser-origin message is captioned with its device and channel", () => {
  assert.equal(deviceCaption("browser", "Laptop"), "from Laptop · Browser");
});

test("a browser-origin message matching this sidebar's own label is its own echo", () => {
  assert.equal(isOwnDevice("browser", "Laptop", "Laptop"), true);
});

test("a browser-origin message from a different device is not its own echo", () => {
  assert.equal(isOwnDevice("browser", "Desktop", "Laptop"), false);
});

test("a Signal-origin message is never its own echo, even if the label matches", () => {
  assert.equal(isOwnDevice("signal", "Laptop", "Laptop"), false);
});

test("an iOS-origin message is never its own echo, even if the label matches", () => {
  assert.equal(isOwnDevice("ios", "Laptop", "Laptop"), false);
});
