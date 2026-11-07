# penny-client
iOS client for the penny personal assistant

Build and test from the repo root with `make client-check` (requires Xcode; runs
the standalone service tests, app-hosted `PennyClientTests`, and the Testflight
build verification on a freshly booted simulator). For service-only changes,
`make client-services-check` runs `PennyServicesStandaloneTests` without
building or launching an app target. CI runs the complete gate on pull requests
touching `penny-client/`.

`make client-ui-tests` runs the `PennyClientUITests` XCUITest smoke suite (#3106) —
tapping each shell tab, pushing and returning from a memory's detail, opening
find-in-chat and stepping between matches, and sending from the composer — against
fixture data (`-PennyStory`, #2965), never a server. It's LOCAL ONLY: CI's
`client-check.yml` never builds or runs it (macOS runners are scarce), so run it
yourself before review whenever a PR adds or moves an interactive control.

While Penny works, a rainbow-edged glass live dock sits at the end of the
conversation, above the message field: a header (`Working · Step n · t s`) over
a chronological stream of her thinking and tool calls, growing with the run up
to 40% of the screen before it scrolls internally. It fades into her reply
bubble once the run ends.

Settings → Image Attachments controls automatic images for the entire iOS channel:
cited-page images, same-site images, and broader related images. All switches start
enabled. Changes save immediately to the server with confirmation and error state;
the connection settings' Save button is separate. Explicit attachments and images
generated for the current request always remain deliverable. The controls affect
new replies; existing messages and the History attachment option are unchanged.
An older server that does not expose all four switches shows disabled controls.


Navigation is one centred glass tab capsule at the top (Chat, Memory, Prompts,
Settings) with a search circle; pushed pages keep the capsule and add a glass back
circle. The Memory and Prompts tabs host the read-only data browser (previously a
 Memories/Prompts sheet off the chat's memories button). Memories are grouped into
Collections, Logs, and Archived; each has Entries,
Activity, and read-only Config panels. Prompts shows all agent runs with filters and
expandable run → prompt → turn details. The previous Insights and Memory Management
screens have been removed. Global settings live on the Settings tab.

The browser reads the current memory contract, including `schedule` (RRULE) and
notification status (`published` on the wire). Deploy the accompanying iOS server
support before this client: memory list/detail/page and prompt-log reads use optional
`request_id` values echoed on responses and `data_read_error` failures. Existing
clients without IDs remain supported. Browser requests time out after 30 seconds;
filter changes, dismissal, and reconnect discard obsolete results. No memory data is
persisted on the client, and this browser provides no mutation controls.

Received messages retain their generating `run_id` through delivery, history, and
local SQLite storage. Long-press a linked message and choose **View prompts** to
open the existing prompt browser at that exact run, expanded, for chat replies or
composed collection reports. Missing runs show an empty state; **Browse all prompts**
returns to the regular browser. Unlinked messages keep Reply and Copy only.
The server must expose `run_id` on iOS message records and support correlated
`run_detail_request` / `run_detail_response` reads. Existing cached messages gain
linkage when refreshed from server history, including already-visible message menus;
no text or timestamp matching is used.
