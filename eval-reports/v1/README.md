# V1 framework-baseline validation reports

Committed eval-run artifacts from the **V1 validation pass** (epic #1691, issue #1710) —
the first live run of the instrumented eval framework (per-run artifacts #1692, graded
checks #1694, cause partition #1695, regression-aware reports #1693, run-comment assembler
#1717). Each validation run's durable artifacts land under `runN/`: the input
`manifest.json` (commit · model · N · required lever), `results.jsonl` (one
mechanically-diffable `CaseArtifact` per case), the per-case `<case_id>.md` transcript, and
the assembled PR comment. `run1/` is the first artifact-era baseline for the chosen case;
`run2/` re-runs the same case against `run1/` as its `EVAL_BASELINE` to exercise the
regression-diff path end-to-end. Every subsequent eval run diffs against a committed
reference here. Content is synthetic fixture data only (the `board-games` eval fixture).
