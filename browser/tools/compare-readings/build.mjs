/**
 * Build the three artifacts of the reading-comparison harness.
 *
 *   extension/build/harness.js         the self-driving background script
 *   extension/build/comparator.js      what it injects into each page
 *   ../../dist/compare/compare-snippet.js  the paste-into-DevTools build
 *
 * Run from `browser/` as `npm run build:compare` (or via `npm run compare`,
 * which builds and then launches Firefox on a throwaway profile).
 *
 * The injected build gets the same wrapper `build-content.mjs` uses: an IIFE
 * whose RETURN value is the comparison, because that is what
 * `browser.tabs.executeScript` hands back to the caller.
 */

import { build } from "esbuild";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";

const HERE = new URL(".", import.meta.url).pathname;
const BROWSER = resolve(HERE, "../..");

/** The call `content_entry.ts` ends on — turned into the IIFE's return value. */
const ENTRY_CALL = "compareReadings(document, location.href);";

await emit("harness.ts", `${HERE}extension/build/harness.js`, iife);
await emit("content_entry.ts", `${HERE}extension/build/comparator.js`, returningIife);
await emit("snippet.ts", `${BROWSER}/dist/compare/compare-snippet.js`, iife);

/** Bundle one entry point and write it through `wrap`. */
async function emit(entry, outfile, wrap) {
  const result = await build({
    entryPoints: [`${HERE}${entry}`],
    bundle: true,
    format: "esm",
    target: "es2020",
    write: false,
  });
  const code = wrap(result.outputFiles[0].text);
  mkdirSync(dirname(outfile), { recursive: true });
  writeFileSync(outfile, code);
  console.log(`  ${relative(BROWSER, outfile)}  ${(code.length / 1024).toFixed(1)}kb`);
}

function iife(code) {
  return `(() => {\n${code}\n})();\n`;
}

/** executeScript captures the value of the last expression evaluated, so the
 *  wrapper has to RETURN the comparison rather than just compute it. */
function returningIife(code) {
  if (!code.includes(ENTRY_CALL)) {
    throw new Error(`bundle does not end on ${ENTRY_CALL} — the return wrapper would be silent`);
  }
  return iife(code.replace(ENTRY_CALL, `return ${ENTRY_CALL}`));
}
