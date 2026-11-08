# Memory-tool shape A/B (#3160)

A whole-suite comparison of two candidate replacements for Penny's seven collection/log
read-write tools (`collection_list`, `collection_get`, `collection_keys`,
`collection_write`, `log_read`, `log_get`, `log_append`) against the tools as they stood on
`main`. Neither arm beat main; both were closed without merging, and the tool surface is
unchanged.

## The question

In self-reflection, the model reported confusing collections and logs — calling a
collection tool on a store that was a log, and the reverse — and traced the cause to
needing to know a store's type before it could pick that family's tool. Asked what shape
it would want instead, it proposed a single pair of action-dispatched tools. Two arms were
built to measure that proposal against a sibling design that kept each verb its own typed
tool and let the code resolve the store's type.

## The two arms

- **Arm A — `memory_access` / `memory_admin`** (#3127, PR #3138). `memory_access(target,
  action, payload)` carries the actions `read` / `get_item` / `write` / `list_identifiers`;
  `memory_admin(target, action, params)` carries `configure` / `archive` / `restore` /
  `delete`. One pair of tools, an `action` enum, and a free-form payload per action.
- **Arm B — `memory_read` / `memory_get` / `memory_keys` / `memory_write`** (#3128, PR
  #3141). One typed tool per verb, each keeping the options its two predecessors (one
  collection-shaped, one log-shaped) had, with Python resolving which kind the named memory
  is. The old metadata tool `memory_get` was renamed `memory_info`, freeing `memory_get` for
  the new entry-get read.

Both arms kept every kind of read and write 1:1 with what it replaced — nothing lost — and
carried no prod-data migration. `collection_sandbox_run`, `log_create` and `find` were
unchanged in both. Arm A additionally folded `collection_configure`, `collection_archive`
and `collection_remove` into `memory_admin`; arm B left those three alone.

## Method

A whole-suite `make eval-remote` run per arm, on the suite's model and pinned provider,
compared against a same-day baseline run of main's tools. Measures: checks passing,
samples with a wrong-store-type refusal (the error that motivated the experiment),
malformed calls on the new tools, and input/output tokens per sample.

## Results

Whole suite on the suite's model, against main's same-day baseline
(`run-20261107T170651Z-09824`):

| | main | Arm A — `memory_access`/`memory_admin` (PR #3138) | Arm B — `memory_read`/`get`/`keys`/`write` (PR #3141) |
|---|---|---|---|
| Checks passing | 5921/6355 (93.2%) | 6122/6621 (92.5%) | 6214/6689 (92.9%) |
| Samples with a wrong-store-type refusal | 2 | 0 | 0 |
| Malformed calls on the new tools | — | 20 | 9 |
| Input tokens per sample | 19,224 | 18,772 | 19,274 |
| Output tokens per sample | 1,018 | 1,103 | 1,050 |

- The error the experiment set out to fix was already rare: 2 of 1,653 samples on main.
- **Arm A** removed that error but brought in 20 calls with the wrong shape: 11
  `memory_admin` calls written in the old flat `collection_configure` shape, and 9 that
  nested `target` inside `payload`.
- **Arm B** brought in 9: 8 teaching errors for collection-only options sent to a log, and
  1 unknown parameter.
- Neither arm saved meaningful tokens.
- Arm B's figures include three identity cases rerun after a harness fix
  (`run-20261107T221131Z-21243`) — see the lessons below.

**Verdict:** neither arm beat main's checks-passing rate, and each traded the rare
cross-store error for a larger count of new shape errors. Both PRs closed without merging.

## Lessons for the next tool-shape comparison

- **A whole-suite run with no `EVAL_LEVER` does not fail fast — it runs to completion with
  nothing to show for it.** `make eval-remote` only opens a durable report directory when
  `EVAL_LEVER` is set; left unset, the run proceeds as an ordinary ephemeral iteration run
  and writes no durable artifact, so `make eval-report` has nothing to post afterward. One
  run in this comparison cost a full 70 minutes before that was noticed. Set `EVAL_LEVER`
  to the arm's one-line hypothesis up front, before starting the run.
- **A renamed tool must be renamed everywhere the model READS it, not only in assertions.**
  Seeded conversation history, world content, and seeded job instructions that still name
  the old tool leave that arm reading a tool it no longer has, which handicaps it against
  the sibling arm rather than measuring the design itself.
- **A harness driver that calls a tool directly must assert the tool's result.** One
  identity driver in this comparison called the new tool but never checked what it
  returned, and silently scored an untouched store for a whole arm; the figures above were
  corrected by rerunning the affected cases after the harness fix.
- **Whole-suite reports can overflow GitHub's comment cap.** Fixed in PR #3152: an
  oversized case block now truncates to fit the cap and names the run artifact that holds
  the untruncated text, instead of blocking every other case's report from posting.
- **Size disk for concurrent arms.** Two concurrent whole-suite runs, on top of the usual
  backlog of recent artifacts, filled the disk during this comparison. Age-based pruning
  (`make eval-prune`, the default `EVAL_ARTIFACT_KEEP_DAYS=7`) only removes runs older than
  its keep-days window, which won't be true yet for anything from the same multi-arm
  comparison — clear space deliberately before starting the next arm.
- **Compare pass rates, not raw counts.** Completed-sample counts differ between runs (main:
  6355 checks; arm A: 6621; arm B: 6689), so the comparison above reads as a rate (93.2% vs.
  92.5% vs. 92.9%), never as one count set against another.
