# Browser Tools Implementation Plan

## Progress

- [x] Step 1: Protocol additions (TS + Python)
- [x] Step 2: BrowserChannel request/response mechanism
- [x] Step 3: BrowseUrlTool (server-side)
- [x] Step 4: Dynamic tool registration
- [x] Step 5: Background script tool dispatch + hidden tab
- [x] Step 6: Content script for text extraction
- [x] Step 7: Sidebar permission dialog
- [x] Step 8: Manifest updates + integration test

## Design

### Tool execution flow
1. Agent loop calls `browse_url(url)` tool
2. `BrowseUrlTool.execute()` calls `BrowserChannel.send_tool_request()`
3. Channel sends `tool_request` via WebSocket with correlation `request_id`
4. Background script receives request, checks domain allowlist
5. If domain unknown → asks sidebar for permission via runtime message
6. User allows/denies → stored in allowlist for future calls
7. If allowed → opens hidden tab, injects content script, extracts text
8. Sends `tool_response` back via WebSocket
9. Channel resolves the asyncio Future
10. `BrowseUrlTool` summarizes via sandboxed model call
11. Returns summary to agent loop

### Permission model
- Domain allowlist stored in `browser.storage.local`
- Three states: allowed, blocked, unknown
- Unknown → prompt user → store response
- Subsequent calls to same domain auto-resolved
- Management UI deferred (dev tools for now)

### Hidden tab approach
- `browser.tabs.create({ url, active: false })` — full engine + user session
- Wait for tab load via `browser.tabs.onUpdated`
- Inject content script via `browser.tabs.executeScript`
- Content script extracts visible text (skip hidden elements, scripts, styles)
- Close tab after extraction
- Timeout handling for slow/stuck pages

### Sandboxed summarization (server-side)
- Raw page text → separate constrained model call
- System prompt: "Summarize this web page content"
- No tools, no user context, no preferences
- Agent only sees the summary, never raw untrusted content
