# Penny Browser Extension — Architecture & Design

## Vision

Penny originated as an idea 20 years ago: a personal agent that browses the web for you, finds things you'd find interesting, and builds a feed — rather than relying on external feeds. The browser extension brings this vision to life by giving Penny direct access to a browser, while keeping the existing server architecture intact.

## Architecture: Browser as a New Channel

The extension does NOT replace the current Penny architecture. It extends it.

```
                    ┌─────────────────┐
                    │   Penny Server  │
                    │   (Mac, 64GB)   │
                    │                 │
                    │  Agents         │
                    │  Scheduler      │
                    │  SQLite Store   │
                    │  Ollama (20B)   │
                    └──┬──────┬───────┘
                       │      │
              ┌────────┘      └────────┐
              │                        │
     ┌────────▼──────┐     ┌──────────▼──────────┐
     │ Signal Channel│     │ Browser Channel(s)  │
     │               │     │                      │
     │ Chat (phone,  │     │ Chat (sidebar)       │
     │  laptop, etc) │     │ Browser tools        │
     │               │     │ Active tab context   │
     └───────────────┘     │ History reading       │
                           │ Feed page            │
                           └──────────────────────┘
```

- **Penny server** stays where it is: model, store, thinking loop, scheduler, all on the Mac
- **Signal channel** stays as-is: talk to Penny from any device
- **Browser channel** is additive: a Firefox extension that connects via WebSocket, providing chat + browser capabilities
- **Multiple browsers** can connect: PC (always-on), MacBook (intermittent), etc. Each is a device in the device table with a user-chosen label
- **Shared history**: all messages appear in the same conversation history regardless of channel — ask on Signal, follow up on browser, full continuity

## Multi-Channel Architecture (implemented)

Penny uses a ChannelManager that implements the MessageChannel interface as a routing proxy. All agents, scheduler, and commands interact with the manager — they don't know which channel they're talking to.

- **Device table**: each connection point (Signal phone, browser instance) is a device with a unique identifier and label
- **Single-user model**: Penny is a personal assistant for one person. All devices map to the same user via `UserInfo`. Background agents use `get_primary_sender()` from UserInfo, not sender strings from MessageLog
- **Reply routing**: `message.sender` (device identifier) routes replies to the correct channel. `_resolve_user_sender()` maps any device to the canonical user identity for DB lookups
- **Device registration**: Signal device seeded at startup. Browser devices auto-register on first message. Sidebar prompts for a device label on first open

## What the Extension Adds

The browser gives Penny capabilities she can't have through APIs alone:

- **Direct web browsing**: open pages in hidden tabs with full rendering engine + user session, extract content with Defuddle
- **Active tab context**: automatically extract the page the user is currently viewing and inject it into the chat context so the user can ask questions about any page
- **YouTube/video discovery**: see thumbnails, durations, view counts, comments, transcripts
- **Price tracking**: revisit URLs and detect changes (price drops, back-in-stock)
- **Rich media extraction**: og:image, Open Graph data, JSON-LD structured data
- **Browsing history as preferences**: passive preference extraction from what the user actually does, not just what they say
- **Social/forum monitoring**: check Reddit threads, forums, Discord channels for interesting discussions
- **Feed page**: a browsable web page of Penny's discoveries, richer than Signal text notifications

## Extension Components (implemented)

```
browser/
  src/
    protocol.ts           — Typed WebSocket + runtime messaging protocol
    background/
      background.ts       — Owns WebSocket, tool dispatch, tab tracking, permissions
      permissions.ts      — Domain allowlist management + user prompts
      tools/
        browse_url.ts     — Hidden tab + content extraction handler
    sidebar/
      sidebar.ts          — Chat UI, page context toggle, message rendering
    feed/
      feed.ts             — Thought card grid, new/archive tabs, reactions, modal
    content/
      extract_text.ts     — The injected script: readiness, XML, og:image, PageData
      page_text.ts        — Reading a page: article body or link index (bundled via esbuild)
  sidebar/
    sidebar.html          — Chat UI markup
    sidebar.css           — Light/dark theme via CSS custom properties
  feed/
    feed.html             — Thought feed page markup
    feed.css              — Card grid, modal, tabs, reaction buttons
  icons/
    icon-{16,32,48,96}.png — Extension icons rendered from SVG
  penny.svg               — Vector logo for in-page display
  manifest.json           — Permissions: storage, tabs, <all_urls>
  tsconfig.json           — Strict TypeScript, ES2020 target
  build-content.mjs       — esbuild wrapper for content script bundling
  package.json            — TypeScript, esbuild, web-ext, concurrently, defuddle, fontawesome
```

### Background script
Owns the single WebSocket connection to the Penny server. Persists across sidebar open/close. Handles:
- Chat message relay between sidebar and server
- Tool request dispatch (receives `tool_request` from server, executes, sends `tool_response`)
- Active tab tracking (extracts page content on tab switch/load)
- Domain permission checking and user prompts
- Connection state management and reconnection

### Sidebar
Pure UI layer — no direct WebSocket connection. Communicates with background via `browser.runtime` messaging. Features:
- Chat with message persistence in `browser.storage.local` (200 message cap)
- Device registration on first open
- Page context toggle showing current page title + favicon with checkbox
- Permission dialog for domain access requests
- HTML rendering of Penny's responses (markdown → HTML conversion server-side)
- Flush image rendering with og:image page headers on contextual responses
- Connection status indicator (green dot / orange spinner)
- Smart scrolling (short messages anchor at bottom, long messages show top first, re-scroll on image load)

### Content script
Reads a page **two ways** and returns whichever carries more of the page's own words
(`page_text.ts`, #1942):
1. **The article** — Defuddle, which scores the document for the one block holding a story
   and strips nav, sidebars and boilerplate.
2. **The link index** — one markdown link per distinct page this one points at, built only
   for a page whose own text is mostly link text (a homepage, a section front). Defuddle is
   an *article* extractor, so on an index page it either returns a single card or keeps the
   grid and deletes each card's headline as a trailing heading — both of which look like a
   successful extraction downstream.

An article page is unaffected: its index reading is never built, and would lose if it were.
The script also extracts og:image metadata and serialises XML/RSS documents directly.
Bundled with esbuild as an IIFE (content scripts can't use ES module imports); the reading
logic is unit-tested against synthetic fixtures over jsdom (`browser/test/`, `npm test`).

## Browser Tools

### Implemented

- **`browse_url`** — opens a URL in a hidden tab with full web engine + user session. Content extracted via Defuddle content script. Server summarizes in a sandboxed model call before the agent sees it. Domain-gated with user permission prompts.

### Active tab context (implemented, not a tool)
The background script continuously extracts visible text from the active tab using Defuddle. When the user sends a message with the "Include page content" toggle checked, the page content is attached to the message. The server injects it as a synthetic `browse_url` tool call + result in the message history, so the model sees a pre-completed tool exchange and answers from it directly. A minimal hint (title + URL) in the system prompt disambiguates "this page" references.

### Planned

- `search_web` — use the browser's default search engine, return results
- `search_youtube` — search YouTube, return video metadata (title, duration, views, channel, thumbnail)
- `get_history` — read recent browsing history (filtered, domain + title only)
- `screenshot_page` — capture current page for vision model
- `get_current_tab` — what the user is looking at right now (partially implemented via active tab context)

The thinking agent uses browser tools autonomously when a browser is connected. All web access (search, page reading) goes through the browser extension. Tools are registered dynamically via `set_browser_tools_provider()` — available only when a browser has an active connection.

## Tool Request Protocol (implemented)

RPC over WebSocket with correlation IDs:

1. Server sends `tool_request` with `request_id`, `tool` name, and `arguments`
2. Background script receives, checks domain permissions, executes tool
3. Background sends `tool_response` with matching `request_id` and `result` or `error`
4. Server resolves the asyncio Future and returns result to the agent loop

The `BrowserChannel` maintains `_pending_requests: dict[str, asyncio.Future]` for in-flight requests. Timeout after 30 seconds. Pending futures rejected on disconnect.

## Skills

Skills are domain-specific playbooks — instructions for interacting with particular websites:

- **YouTube skill**: how to find videos, parse metadata, read transcripts, evaluate quality
- **Reddit skill**: how to find relevant threads, sort by quality, extract discussions
- **Amazon/Reverb skill**: how to find prices, reviews, availability, related products
- **Forum skill**: how to navigate threads, find expert opinions

Each skill is a document (not code) — instructions and DOM selectors that Penny reads when interacting with that domain. Skills are:

- Tied to the domain allowlist: granting access to youtube.com activates the YouTube skill
- User-extensible: write a skill document for any site
- Part of the trust boundary: defines what to extract and what to ignore

## Security Model

### Core Principles

Security is the foundational design constraint, not a feature added later.

1. **Zero permissions by default**: Penny has no access to any domain until the user explicitly grants it via an allowlist
2. **Collect, store, and expose nothing first**: only incrementally add data access as needed
3. **If Penny has no access to sensitive data, there's nothing to exfiltrate**

### Prompt Injection Defense (implemented)

The primary threat: malicious web pages embedding instructions in content that Penny reads. Attack vectors include hidden text (CSS `display:none`, white-on-white), HTML comments, meta tags, YouTube descriptions/comments, forum posts, product descriptions.

**Three-layer defense:**

1. **Pre-sanitization**: on the article reading, Defuddle extracts main content only, stripping nav/sidebar/footer/hidden elements. The **link index** reading (#1942) does not go through Defuddle — it reads the page's own anchors — so it drops `script`/`style`/`noscript`/`template` and anything the page marks `hidden` or `aria-hidden="true"` itself, and it emits link *labels* and hrefs rather than page prose. Stylesheet-hidden text (`display:none`, white-on-white) is not filtered on that path; layer 2 is what stands behind it.

2. **Sandboxed summarization**: raw page content from `browse_url` goes to a constrained model call — system prompt is "Summarize this web page content." No tools, no user profile, no preferences, no history, no browsing context. Even if injected instructions survive extraction, there's nothing to exfiltrate and no tools to act with.

3. **Summary is the only bridge**: Penny's real agent context (with tools, preferences, sensitive data) only sees the clean summary output from the sandboxed step. Untrusted content never shares a context window with sensitive data or tool access.

**Note**: Active tab context (when user includes page content with a message) bypasses the sandboxed summarization since the user explicitly chose to share it. The content still goes through the extraction above, which strips hidden/malicious elements.

### Domain Allowlist (implemented)

- No domains accessible by default
- User explicitly grants or denies access per domain via sidebar permission dialog
- Decisions stored in `browser.storage.local` for future calls
- Parent domain matching (allowing `example.com` also allows `www.example.com`)
- Three states: allowed, blocked, unknown (unknown triggers prompt)
- Management UI deferred (currently via browser dev tools)

### Data Policies

- Browsing history sent to server: domain + page title only, no full URLs with query params
- Filter out sensitive categories (health, finance, adult) before sending
- Communication over localhost WebSocket only
- Model never sees raw untrusted web content via `browse_url` — always extracted, sanitized, summarized first
- Active tab context uses Defuddle extraction (user-initiated sharing)
- Penny never relays verbatim web page text through Signal — always summarized/rephrased
- Never pass raw cookies or session tokens to the server

### Tool Constraints

- Browser tools gated by domain allowlist
- Model output cannot drive navigation directly without gating
- Tool invocations rate-limited to prevent uncontrolled spidering
- Active tab context only sent when user explicitly checks the toggle

## Multi-Browser Continuity

- Each browser instance connects as a device with a user-chosen label (e.g., "firefox macbook 16")
- Device table tracks all connected endpoints with channel type, identifier, and label
- Persistent store lives on the Penny server — all browsers share it via the same user identity
- Always-on PC browser can do background research while laptop is closed
- Signal provides fallback communication when no browser is connected
- Thinking agent uses `browse_url` autonomously on any connected browser

## Feed Page (implemented)

The feed page realizes the shift from interruption-based notifications to a browsable discovery feed:

- **Card grid**: each thought rendered as a card with image (if available), title, seed topic byline, date, and truncated content
- **New / Archive tabs**: unnotified thoughts in New, notified in Archive (paginated, 12 per page)
- **Modal viewer**: click a card to see full content with image header
- **Reactions**: thumbs up/down buttons on cards and modal. Clicking one:
  - Sets `notified_at` on the thought (moves to Archive)
  - Logs a synthetic outgoing message with the thought content + `thought_id` FK
  - Logs a reaction message (👍/👎) with `parent_id` pointing to that outgoing message
  - The preference extraction pipeline picks up the reaction identically to Signal emoji reactions
  - Fades the card out of the New tab
- **Unnotified count**: sidebar nav shows `Thoughts (N)` with the count of new thoughts, polled every 5 minutes
- **Content rendering**: thought content processed through `prepare_outgoing()` server-side (markdown → HTML), rendered directly — no client-side duplication
- **Image URLs**: stored on the `Thought` model (legacy field, no longer auto-populated)
- Both models coexist: feed for passive browsing, Signal for high-priority discoveries

## Technology

- **Extension**: TypeScript (strict), esbuild for content script bundling, web-ext for development
- **Content extraction**: Defuddle (by Obsidian creator, zero-dep core bundle) for the article reading, beside a link-index reading of the page's own anchors; whichever carries more of the page's words wins (#1942)
- **Communication**: WebSocket between background script and Penny server, `browser.runtime` messaging between sidebar and background
- **Protocol**: Typed discriminated unions for all messages (TypeScript `type + const` pattern mirrors Python Pydantic models)
- **Server**: existing Python/Docker architecture with ChannelManager routing proxy
- **Model**: Ollama on server (20B), not constrained by browser ML engine limitations
- **Storage**: server SQLite for persistent data, `browser.storage.local` for device label, chat history, domain allowlist
- **Theming**: CSS custom properties with `prefers-color-scheme` media query for automatic light/dark

## Incremental Build Path

1. ~~Minimal Firefox extension: sidebar chat panel + WebSocket to Penny server~~ **Done**
2. ~~Add `browse_url` tool: extension can fetch and sanitize a page on demand~~ **Done**
3. Add `search_web` tool: use browser's search engine
4. Browsing history reader: passive preference extraction
5. ~~Feed page: render thoughts as a browsable page~~ **Done**
6. YouTube skill: first domain-specific skill
7. Additional skills: Reddit, Amazon, forums, etc.

### Additional implementations (not in original plan)
- Multi-channel architecture (ChannelManager, device table, single-user identity)
- Active tab context injection (Defuddle extraction + synthetic tool call)
- Page context toggle with og:image headers on responses
- TypeScript with typed protocol (discriminated unions, const objects)
- Background script owns WebSocket (sidebar uses runtime messaging)
- Domain permission dialog flow
- Chat history persistence (browser.storage.local, 200 message cap)
- HTML formatting pipeline (markdown → HTML with code, links, tables-to-bullets)
- Flush image rendering with smart scrolling
- Light/dark theme support
- `/draw` command working in browser (base64 data URI rendering)
- Reconnection indicator with spinner
- `web-ext` dev setup with auto-reload
- Thoughts feed page with new/archive tabs, card grid, modal viewer
- Thought reactions (thumbs up/down) feeding preference extraction pipeline
- Image URLs on thought model (legacy field, no longer auto-populated)
- Seed topic bylines on thought cards (from preference FK)
- Unnotified thought count in sidebar nav (5-minute polling)
- Penny logo: SVG traced from PNG, rendered to crisp PNGs at all icon sizes
- Font Awesome icons (locally bundled, no CDN)
- Signal profile avatar via signal-cli-rest-api

### TODO — come back to these
- **Page header on model-initiated browse_url**: when Penny calls browse_url on her own, the sidebar response should show the browsed page's image/title/URL header (same as user-initiated page context). Requires the server to include the browsed URL metadata alongside the response
- **Deferred cleanup — drop sender column**: remove `sender` from `MessageLog`, migrate all queries to `device_id` or remove sender filtering (single-user). ~30 call sites
- **Domain allowlist management UI**: currently managed via browser dev tools, needs a proper settings page
- **Rate limiting on tool invocations**: prevent uncontrolled spidering from the thinking agent
