# V1 framework-baseline validation reports

Committed eval-run artifacts from the **V1 validation pass** (epic #1691, issue #1710) —
the first live run of the instrumented eval framework (per-run artifacts #1692, graded
checks #1694, cause partition #1695, regression-aware reports #1693, run-comment assembler
#1717). Each run's durable artifacts land under its own subdir: the input `manifest.json`
(commit · model · N · required lever), `results.jsonl` (one mechanically-diffable
`CaseArtifact` per case), and the per-case `<case_id>.md` transcript. The large per-sample
`.db` files and `dirty.diff` stay local (raw artifacts per `docs/eval-report-format.md`) —
they live beside these under `data/eval-reports/v1/` in the runtime tree (gitignored), kept
for audit.

## Subdirs

- **`browse-run1/`**, **`browse-run2/`** — the V1 validation pair. Case:
  `test_chat_response.py::test_browse_answer` (`chat-browse-answer`, family `browse-answer`),
  a gated (harness default `min_pass_rate=0.75`) single-turn chat case: "what's the deepest
  lake in the world?" → `browse` a canned page → surface the fact ("Lake Baikal"). Pure
  chat-agent reasoning over injected browse pages — no collector or collection-procedure
  surface anywhere. `browse-run2` re-runs the same case with `browse-run1` as its
  `EVAL_BASELINE` to exercise the regression-diff path end-to-end.
- **`legible-run1/`** — historical record of the *first* representative case tried,
  `test_legible_prompts.py::test_edit_operations_across_turns` (`legible-edit-operations`).
  Its run-1 comment is posted on PR #1723 per the #1711 protocol. **This case was retired as
  the V1 representative**: it drives the *outmoded* collector-prompt-editing surface (post-v3,
  a collection's procedure is a render of a taught skill; modification is re-teach, not a
  direct recipe edit), so its 0.61 mean / 0-of-3 all-pass reflects surface drift, not a
  framework fault. It is the first concrete candidate for the deferred case-inventory
  deprecation audit. No run-2 was recorded for it (stopped on the case switch).

Content is synthetic fixture data only (the `board-games` and Lake-Baikal eval fixtures).
