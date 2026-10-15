"""End-to-end job cases (#2375) — a job walked from the user's ask to its own real cycles,
in one sample: several scripted chat turns stand it up, then the job it created runs through
`end_to_end_eval`'s own real collector cycles. The declared second kind `docs/eval-case-design.md`
describes, beside the single-edge chat and collector cases the rest of the suite is built from."""
