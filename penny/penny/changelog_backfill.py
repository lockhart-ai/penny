"""One-off backfill for the `changelog` log (#2962).

The `changelog` log (#2845/#2891) only started recording PRs once it existed
— its own creation, #2847, merged 2026-10-30. Every Penny-scope PR merged
between this deployment's DB reset (2026-10-18) and that point never got an
entry, so a `log_read('changelog')` today reads as if nothing happened in
those twelve days.

``WINDOW_ENTRIES`` below is the fix: every `feat|fix|perf|refactor(penny)`
PR merged in that window, each as ``(pr_number, merged_at, text)`` —
``text`` is that PR's own body's first sentence
(``penny.changelog_commits.first_body_sentence``), or (no usable one) its
title, exactly the same rule a live deploy's entries are built by. Fetched
and resolved ONCE, while this module was written (``gh pr view <n> --json
body,mergedAt``, over the GitHub REST API with an App installation token —
never the squashed commit's own body, which for a PR this old is NOT the PR
body; that only became true at #2896). Baked in as static, committed data
rather than fetched at run time, so:

- the code owner and the coordinator can read the actual 76 sentences in
  this PR's diff before they ever reach prod;
- running this script needs no GitHub credential, no network call, and no
  `gh` binary inside the container — it only records data;
- a re-run is byte-for-byte deterministic.

The 76 PRs were found by walking every no-merge commit on `main` with a
commit time in `[2026-10-18T00:00:00Z, 2026-10-30T06:16:11Z)` — the second
bound is PR #2847's (the log's creation) own commit time, found by exact
epoch comparison rather than `git log --since/--until` (which resolves
through the invoking shell's local timezone and silently shifts the
boundary) — then keeping only the ones matching the same
`(feat|fix|perf|refactor)\\(penny\\): ... (#N)` scope `changelog_commits.py`
already filters commits by. Nine of those PRs (#2820, #2822, #2831, #2835,
#2839, #2844, #2847, #2848, #2849) were excluded up front: they were already
recorded, via the original `changelog.yaml` seed that predated the
commit-log mechanism (#2891).

Records through Penny's own idempotent append path
(``penny.changelog.record_new_entries`` → ``Log.append_at``) — same author
(``PennyConstants.CHANGELOG_LOG_AUTHOR``), same dedup (a PR already in the
log, including a prior run of this very script, is silently skipped), same
"no embedding written here" (the next real startup's own embedding backfill
picks these rows up exactly like a live-deployed entry, #2962's plan).
Unlike a live deploy's entries — which all share ONE "now" stamp, since they
really did all just reach her together — each backfilled entry carries its
OWN PR's merge time, so the log still reads in the order things actually
happened: a October PR never looks newer than one from November just
because it was recorded later.

Run by the owner, against prod, after reading this module's diff:

    docker compose run --rm --no-deps penny python -m penny.changelog_backfill

A dry run against a backup copy first is safer than trusting a read alone —
see this PR's body for the exact steps (stop Penny, back up `penny.db`, run,
check the printed list, restart).
"""

from __future__ import annotations

import os
from datetime import datetime

from penny.changelog import record_new_entries
from penny.changelog_commits import ChangelogEntry
from penny.constants import PennyConstants
from penny.database import Database
from penny.database.memory import Memory

DEFAULT_DB_PATH = "/penny/data/penny/penny.db"

# (pr_number, merged_at, text) — oldest first. See the module docstring for
# how this list was derived and why it's committed data, not a live fetch.
WINDOW_ENTRIES: list[tuple[int, str, str]] = [
    (
        2537,
        "2026-10-18T03:28:40Z",
        "This PR carries the whole naive approach for review: Penny's conversation "
        "machine and skill infrastructure are removed in favour of jobs that run the "
        "free-text prompt they were given, with two gemma eval runs attached — full "
        "detail on the three end-to-end cases, a summary table on the whole suite.",
    ),
    (
        2579,
        "2026-10-18T04:15:26Z",
        "Runtime config no longer hand-rolls the same session get/upsert in four "
        "different places — the iOS channel, the browser channel, the `/config` "
        "command, and `config_params.py`'s read all go through one `db.config` store.",
    ),
    (
        2580,
        "2026-10-18T04:16:10Z",
        "Replaces objects.py's bare-string tool-name comparisons in run-record "
        "rendering with named constants, and deletes the dead `update_entry` branch "
        "for a tool renamed away in 2025.",
    ),
    (
        2578,
        "2026-10-18T04:25:50Z",
        "Splits ProfileCommand.execute's ~122-line body into named private steps, so "
        "it reads as a short table of contents, with no behaviour or output change.",
    ),
    (
        2576,
        "2026-10-18T04:26:34Z",
        "Browse now validates its arguments before it resets any per-call state, as "
        "every other tool does, so a malformed call leaves the tool untouched.",
    ),
    (
        2585,
        "2026-10-18T04:35:43Z",
        "The behaviour catalog (`penny/penny/validation/conditions.py`) marked "
        "`NO_WORK_DONE` as caught live, but no live premature-`done()` check exists on "
        "main, so this moves it to post-hoc only.",
    ),
    (
        2586,
        "2026-10-18T04:36:33Z",
        "One module-level `_count()` helper in `penny/penny/database/memory/store.py` "
        "replaces four `ty: ignore[invalid-argument-type]` suppressions on "
        "`func.count(Model.id)`-style expressions with a single typed wrap, so the "
        "type checker passes at every call site with no suppression.",
    ),
    (
        2587,
        "2026-10-18T04:37:33Z",
        "Splits MessageStore's longest methods into named private steps and gives the "
        "class a table-of-contents docstring, with no change to behaviour, queries, or "
        "any public signature.",
    ),
    (
        2590,
        "2026-10-18T04:42:11Z",
        "`find`'s tool-schema docstring still named a family it no longer searches, "
        "naming `skill` alongside `collection | log | entry` after #2537/#2542 retired "
        "skills.",
    ),
    (
        2588,
        "2026-10-18T04:42:11Z",
        "Splits `LlmClient.chat` and `_chat_attempts` in `penny/penny/llm/client.py` "
        "into named, 10-20 line private steps and turns `chat()` itself into the "
        "class's summary method, right after `__init__`, with identical behaviour, "
        "request payloads, retries, logging and error paths.",
    ),
    (
        2584,
        "2026-10-18T12:53:59Z",
        "Splits `Collector._execute_cycle` — a single ~89-line method — into named "
        "private steps, so the cycle reads as a table of contents; the cycle's "
        "behaviour is unchanged.",
    ),
    (
        2608,
        "2026-10-18T13:20:37Z",
        "Conversation history now travels through `LlmClient.chat()` as typed "
        "`LlmMessage` objects instead of raw dicts, turning into the wire shape "
        "exactly once at that one entry point, with the request payload proven "
        "byte-identical by a frozen-literal test written against unmodified `main` "
        "before any",
    ),
    (
        2609,
        "2026-10-18T13:35:46Z",
        "Fixes the thinking-off retry so it actually turns thinking off on local "
        "Ollama, instead of thinking anyway and getting cut again at the same token "
        "budget.",
    ),
    (
        2620,
        "2026-10-18T14:08:18Z",
        "The agent loop (`agents/base.py`) now carries conversation history as typed "
        "`LlmMessage`s end to end, instead of raw dicts, so nothing in the loop reads "
        "or builds a message by guessing its keys.",
    ),
    (
        2622,
        "2026-10-18T15:09:57Z",
        "Chat's page-context injection now builds typed `LlmMessage` objects directly "
        "and the injected browse call sends its `arguments` as a JSON string like "
        "every other tool call, finishing the conversation-history-as-`LlmMessage` "
        "migration.",
    ),
    (
        2624,
        "2026-10-18T15:42:19Z",
        "The notify composer now writes a collector's message with thinking off, in "
        "about 5 seconds instead of 30, and in Penny's own voice with each finding's "
        "place, price and change.",
    ),
    (
        2627,
        "2026-10-18T16:44:30Z",
        "The browse extractor now reads pages with thinking off and returns what it "
        "found in plain text, so a page read takes seconds instead of minutes on local "
        "hardware and never loops on a missing part.",
    ),
    (
        2629,
        "2026-10-18T17:00:23Z",
        "Penny's startup announcement now writes its one line with thinking off, in a "
        "second or two instead of about half a minute on local hardware.",
    ),
    (
        2636,
        "2026-10-18T18:27:52Z",
        "An image now goes straight into the chat turn for the model to read, so a "
        "photo gets an answer in about 25 seconds under chat's token budget instead of "
        "a separate uncapped caption call that could hang for ten minutes.",
    ),
    (
        2634,
        "2026-10-18T20:55:47Z",
        "Fixes Penny telling the user she changed a running job without calling the "
        "tool that changes it — the eval case that reproduces the failure, plus two "
        "levers: the conversation history now shows each past turn's tool calls beside "
        "its reply, and the change-a-running-job instruction names the state",
    ),
    (
        2637,
        "2026-10-18T22:45:06Z",
        "Standing rules the user gives Penny are now kept in a seeded `rules` "
        "collection that shows in full wherever she acts — in the chat prompt's "
        "instruction part (right after her voice), in every collector cycle's system "
        "prompt, and in the notify composer's document — so a rule holds everywhere "
        "without",
    ),
    (
        2639,
        "2026-10-18T23:38:25Z",
        "Every datetime the browser channel sends to the addon now reaches it as "
        "timezone-aware UTC, so the Prompts tab displays times correctly and a "
        "still-running entry sorts in place with the finished ones instead of hours "
        "off and above them.",
    ),
    (
        2645,
        "2026-10-19T04:03:18Z",
        "Penny's runtime no longer holds any bare text-length cap: every cap is either "
        "removed (the full text renders), a runtime config param, or a named constant "
        "in `penny/penny/constants.py` with its reason — per the code owner's "
        "no-invented-truncations rule (#2642).",
    ),
    (
        2649,
        "2026-10-19T05:33:47Z",
        "Browse: a reply built on pages read through the extract micro-context now "
        "carries a source URL — the extraction value leads with its page's URL, and a "
        "reply that still composes without one gets the read page's URL appended — so "
        "a standing rule asking for a source URL can be met.",
    ),
    (
        2651,
        "2026-10-19T06:54:46Z",
        "Chat: stops the history appending a `[Tool calls: …]` line to every past "
        "reply — in one real turn the model copied the line as text and stopped, and "
        "the line went instead of a real tool call.",
    ),
    (
        2663,
        "2026-10-19T23:38:50Z",
        "Every per-call completion budget rises to 16000 tokens, so the ceiling only "
        "ever stops a runaway draw, never a healthy one like a daily digest cycle.",
    ),
    (
        2664,
        "2026-10-19T23:39:07Z",
        "Browse's narration line now names only the queries a call actually ran, so a "
        "request over the per-call cap no longer claims to have opened a page it never "
        "fetched.",
    ),
    (
        2669,
        "2026-10-19T23:47:39Z",
        "Browse now runs every requested query in one call, fetching three at a time "
        "in batches, so a job that reads four sites no longer needs a second model "
        "call for the fourth.",
    ),
    (
        2652,
        "2026-10-20T00:40:05Z",
        "Standing rules now render in the chat prompt as plain instruction sentences "
        "and outrank every other guidance in it, and search results show titles and "
        'URLs only — so a rule like "always cite a source and read the promising '
        'pages" is honoured in what reaches the user instead of being outranked,',
    ),
    (
        2692,
        "2026-10-20T03:01:11Z",
        "Jobs: a schedule change can no longer replace a job's instructions with the "
        "one-line summary Penny's current state shows for it.",
    ),
    (
        2693,
        "2026-10-20T04:41:39Z",
        "Chat lookups: the search result list Penny is shown now says outright that it "
        "is titles and URLs only and must be read to answer the user, the system "
        "prompt's browse section describes that same list, and the follow-up lookup "
        "case's world now returns search results that fit the query.",
    ),
    (
        2666,
        "2026-10-20T05:49:21Z",
        "Jobs: a job's notification is now written by Penny's chat agent from "
        "everything the job found, instead of a cut-down preview.",
    ),
    (
        2704,
        "2026-10-20T06:11:30Z",
        "Browse: a search result's title is now only the heading line above its "
        "standalone link, so a search page's knowledge-panel paragraph and infobox "
        "rows can never render as result titles.",
    ),
    (
        2700,
        "2026-10-20T06:16:35Z",
        "Prompts: the standing-rules section is left out entirely until the user has "
        "set a rule, instead of showing an empty placeholder in every job.",
    ),
    (
        2727,
        "2026-10-21T14:51:15Z",
        "Conversation history: the messagelog row for every outgoing send now keeps "
        "the reply as the model wrote it, so the channel's formatting conversion "
        "applies only on the wire.",
    ),
    (
        2728,
        "2026-10-21T16:39:41Z",
        "Memory: replaces the three-signal embedding dedup on collection writes with a "
        "ladder — deterministic rules, an embedding shortlist, a decision model, and "
        "an honest question — so two distinct entries that look alike are never merged "
        "or blocked.",
    ),
    (
        2729,
        "2026-10-21T23:00:20Z",
        "Links the model reads and writes travel as numbered handles end to end — "
        "browse results, history, memory reads and job reports all cite `[n]`, Python "
        "expands the number when a message is sent, and a citation that doesn't exist "
        "is rejected with a note naming the links that do — so a long URL is",
    ),
    (
        2725,
        "2026-10-21T23:12:13Z",
        "Jobs: changing a running job's schedule no longer rewrites its instructions — "
        "`collection_configure` now says a schedule-only change needs no prompt, and a "
        "replaced prompt shows the lines it dropped, with a new eval case proving both "
        "on a job whose multi-line prompt the self-state only shows cut.",
    ),
    (
        2730,
        "2026-10-21T23:17:22Z",
        "Collection writes to a mistyped or reworded collection name are matched by "
        "the entry-identity ladder — rules, embedding shortlist, decision model, "
        "honest did-you-mean refusal — instead of two character-ratio thresholds.",
    ),
    (
        2733,
        "2026-10-21T23:34:55Z",
        "Collection writes that auto-create a collection, and configure calls that "
        "replace a job's prompt while leaving its description alone, now say so in the "
        "reply, so the model sees when a description is a placeholder or may no longer "
        "match the job.",
    ),
    (
        2737,
        "2026-10-21T23:39:38Z",
        "collection_write no longer writes an entry whose content cites a link handle "
        "the run's table does not hold, refusing it with a note naming the handle "
        "while the rest of the batch lands.",
    ),
    (
        2740,
        "2026-10-22T00:11:31Z",
        "Browse: an opened page's own address no longer takes a link-handle number, so "
        "a page's first headline gets the number its link belongs and a digest never "
        "stores a headline with its neighbour's link.",
    ),
    (
        2738,
        "2026-10-22T02:52:03Z",
        "Browse: one call now runs a whole web lookup — the tool searches, the "
        "decision model picks which result to read, and the picked page is read and "
        "extracted, so the model gets what the pages say instead of a list of titles "
        "to choose from.",
    ),
    (
        2748,
        "2026-10-22T03:24:16Z",
        "Chat: caps the conversation history a turn sees at the last 5 user turns with "
        "their replies, as the new `CHAT_HISTORY_TURNS` runtime config option, instead "
        "of an unbounded 20-message window.",
    ),
    (
        2751,
        "2026-10-22T03:30:27Z",
        'Chat prompt: replaces the "no greetings, no sign-offs" clause with one line '
        "that agrees with Penny's friend-texting voice.",
    ),
    (
        2750,
        "2026-10-22T04:02:33Z",
        "Every chat and collector request now carries two system messages — a static "
        "one the model server reuses across turns, and a dynamic one (clock, state) "
        "behind it — so the ~5.9k-token prefix of static text plus tool schemas stops "
        "being re-read on every turn.",
    ),
    (
        2755,
        "2026-10-22T14:24:59Z",
        "Link handles: a reply citing several sources in one bracket, like [4, 6], now "
        "sends each as its link instead of the raw numbers.",
    ),
    (
        2764,
        "2026-10-23T18:27:51Z",
        "Penny's thinking switch now reaches oMLX directly, so a chat call there no "
        "longer silently runs with thinking off once the host-side proxy is gone.",
    ),
    (
        2766,
        "2026-10-23T19:39:14Z",
        "When the entry-identity ladder merges a write into an existing entry under a "
        'different key, `collection_write`\'s result still said "You saved \\"<the key '
        'the model passed>\\"", so Penny told the user she\'d created a new rule or '
        "entry when she'd actually updated an existing one under a different name.",
    ),
    (
        2760,
        "2026-10-23T19:54:31Z",
        "Browse pages reach the reader with repeated nav blocks, standalone images, "
        "consent/subscribe/footer chrome, and encoded-URL or JSON blob lines that "
        "waste model context without carrying anything the reader uses — this PR "
        "extends `clean_browser_content` with four guarded rules that trim exactly "
        "that,",
    ),
    (
        2757,
        "2026-10-23T21:00:37Z",
        "Chat prompt: static prompt, tools and append-only history come first, and "
        "per-turn state is a system message just before the user's message, so a turn "
        "reuses ~75% of the prompt from the server's cache instead of ~10%.",
    ),
    (
        2753,
        "2026-10-23T21:55:44Z",
        "Chat: Penny now sees a one-line-per-exchange map of the conversation older "
        "than her history window, kept at the end of the cached static system prompt "
        "so it changes only when the history is cut back.",
    ),
    (
        2769,
        "2026-10-23T23:06:13Z",
        "Startup registered Penny's own Signal number as the user's device instead of "
        "the actual user's number, so no incoming Signal message — or Penny's reply to "
        "one — ever linked to a device, and the iOS history scope dropped the user's "
        "Signal messages entirely.",
    ),
    (
        2772,
        "2026-10-23T23:35:28Z",
        "`collection_list` (and `collection_sample`) now cap an unbounded read at the "
        "newest/sampled 20 entries by default, and say so honestly when a result is "
        "cut.",
    ),
    (
        2782,
        "2026-10-24T01:05:25Z",
        "Collectors now wait a dedicated, longer idle threshold before starting a "
        "cycle, and a cycle cancelled by foreground activity no longer spends a retry "
        "attempt.",
    ),
    (
        2784,
        "2026-10-24T01:07:35Z",
        "`run_list`'s chat tool schema no longer changes when a job is set up or "
        "archived, so it stops invalidating the chat prompt's cached prefix on every "
        "such change.",
    ),
    (
        2785,
        "2026-10-24T01:23:57Z",
        "Images were stored and sent at full size, so page preview images make up most "
        "of the database and every outgoing attachment; this PR normalizes every image "
        "to a sane size at the two boundaries where images enter Penny.",
    ),
    (2781, "2026-10-24T01:29:41Z", "A BYHOUR-only schedule (e.g."),
    (
        2783,
        "2026-10-24T01:30:45Z",
        "A collector cycle's per-run state (holdings, run history, the clock) rode a "
        "second leading system message with an empty user turn, so prod's runner "
        "(oMLX) — which merges two leading system messages ahead of the tool schemas — "
        "never cached a cycle's prefix; a daily job measured in prod reused 0",
    ),
    (
        2776,
        "2026-10-24T01:45:02Z",
        "find's conversation-aware search now matches literal words as well as "
        "meaning, and gives the conversation its own fair share of results instead of "
        "sharing slots with stored facts.",
    ),
    (
        2770,
        "2026-10-24T04:24:54Z",
        "Several tool results told the model something the call did not do — a "
        "headline naming the wrong outcome, a count taken from the request instead of "
        "the result, or an outcome the result never mentioned — and a model narrating "
        "only the headline repeated that as fact.",
    ),
    (
        2788,
        "2026-10-24T04:27:48Z",
        "Widens chat's CALL_AS_TEXT reroll guard to catch a tool call written as JSON "
        "text anywhere in a reply, not only when the whole reply is one — the gap that "
        'let a prod reply ("On it!" followed by a `generate_image` call object) reach '
        "the user verbatim with no image drawn.",
    ),
    (
        2795,
        "2026-10-24T05:11:36Z",
        "A reply that still cites an unsourced URL or a dangling link handle after the "
        "hallucinated-URL validator's one retry no longer ships as-is: the still-bad "
        "link is removed and the message says one was left out.",
    ),
    (
        2796,
        "2026-10-24T05:13:06Z",
        "Each promptlog row only recorded one total duration_ms, so the addon couldn't "
        "show prompt-processing speed, true generation speed, or how much of the "
        "prompt the server's cache served — this adds five columns that carry it.",
    ),
    (
        2794,
        "2026-10-24T05:16:15Z",
        "Rewrites `generate_image`'s tool description and success result text so they "
        "state the order of events — the call makes the image and returns, then the "
        "reply after it goes out with the image attached and can describe it — instead "
        'of telling the model, inside the description, to compose the "your',
    ),
    (
        2797,
        "2026-10-24T05:36:40Z",
        "A model-choice reroll (chat's call-written-as-text, a call fragment, an empty "
        "draw; the collector's done-written-as-text, prose-instead-of-a-call) now "
        "shows the redraw the discarded draft plus a plain-words note naming what was "
        "wrong, instead of discarding it and re-drawing blindly on the identical",
    ),
    (
        2800,
        "2026-10-24T15:04:52Z",
        "Moves Penny's image generation off Ollama — which dropped experimental image "
        "support, so every drawing had been failing silently-retried — onto a small "
        "local mflux server running FLUX.2 [klein] outside Docker.",
    ),
    (
        2808,
        "2026-10-25T17:07:02Z",
        "This makes a reply read as a refusal only when its OPENING sentence carries a "
        "refusal phrase and the whole reply is short, so a good long answer that "
        'mentions "I can\'t" mid-text, or an honest apology for a real failure, is no '
        "longer discarded and redrawn.",
    ),
    (
        2811,
        "2026-10-25T17:20:37Z",
        "A job's finished-run report now lists only the entries the job itself kept, "
        "not every page the browse tool logged along the way.",
    ),
    (
        2809,
        "2026-10-25T17:20:37Z",
        "`browse` now opens a bare domain like `example.com` as the address it names "
        "instead of searching for it.",
    ),
    (
        2818,
        "2026-10-25T18:13:43Z",
        "Trims the `collection_configure` and `find` tool descriptions to plainer, "
        "less repetitive prose so the model reads less text for the same meaning.",
    ),
    (
        2819,
        "2026-10-25T18:30:50Z",
        "Removes two gpt-oss-era leftovers from the chat prompt's Instructions block — "
        "an imperative telling the model it MUST respond after reading pages, and a "
        "sentence forbidding CSS-selector/XPath requests that nothing in gemma's "
        "browse tool can trigger — and adds the eval case that stresses the second",
    ),
    (
        2810,
        "2026-10-25T19:19:30Z",
        "The collector's close instruction currently writes the done() call as code, "
        "which gemma sometimes copies as plain text instead of actually calling the "
        "tool.",
    ),
    (
        2823,
        "2026-10-25T19:46:00Z",
        "Jobs and the store map now ride the chat agent's cached static system-message "
        "prefix instead of being rebuilt on every turn, cutting prefill work on "
        "ordinary turns.",
    ),
    (
        2834,
        "2026-10-26T02:49:15Z",
        "When gemma's raw-template tool call fails to parse, Penny now keeps the "
        "model's actual completion text in the prompt-log row instead of discarding it "
        "as `{}`, and the agent loop's existing DEBUG line names ~200 characters of "
        "text around the exact failure point.",
    ),
    (
        2838,
        "2026-10-29T05:10:49Z",
        "The image server now releases MLX's GPU buffer cache after every generation "
        "and right after the model loads, and caps it at load, so the resident process "
        "settles near the model's own size instead of growing without bound.",
    ),
]


def build_entries(window: list[tuple[int, str, str]]) -> list[tuple[ChangelogEntry, datetime]]:
    """``WINDOW_ENTRIES`` as ``(ChangelogEntry, created_at)`` pairs — pure,
    no DB, no fetch: the parameter makes this testable against a small fake
    window instead of the real 76-row constant."""
    return [
        (ChangelogEntry(pr=pr, text=text), datetime.fromisoformat(merged_at))
        for pr, merged_at, text in window
    ]


def run_backfill(log: Memory, window: list[tuple[int, str, str]]) -> list[ChangelogEntry]:
    """Record every entry in ``window`` not already in ``log``, each
    stamped with its own ``created_at`` — one ``record_new_entries`` call
    per entry, since a single call stamps its whole batch with one shared
    time (the live-deploy semantics) and a backfill needs a distinct stamp
    per PR. Returns every entry actually recorded, in the order given."""
    recorded: list[ChangelogEntry] = []
    for entry, created_at in build_entries(window):
        recorded.extend(record_new_entries(log, [entry], created_at=created_at))
    return recorded


def main() -> None:
    db_path = os.getenv("DB_PATH", DEFAULT_DB_PATH)
    db = Database(db_path)
    changelog = db.memory(PennyConstants.MEMORY_CHANGELOG_LOG)
    if changelog is None:
        raise SystemExit(
            f"No `changelog` memory row in {db_path} — has migration 0132 run against it?"
        )

    recorded = run_backfill(changelog, WINDOW_ENTRIES)
    skipped = len(WINDOW_ENTRIES) - len(recorded)
    print(
        f"Recorded {len(recorded)} new changelog entr{'y' if len(recorded) == 1 else 'ies'} "
        f"({skipped} already present)."
    )
    for entry in recorded:
        print(f"  {entry.render()}")


if __name__ == "__main__":
    main()
