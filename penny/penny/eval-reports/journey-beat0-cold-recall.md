#### sample 1 — ✅ 3/3 checks

| # | Actor | Content |
|---|---|---|
| 1 | 👤 user | hey — a while back I asked you to remember what the aurora deck 2 was listed at. what was the price? |
| 2 | 🔧 Penny → tool | read_similar({"memory": "knowledge", "anchor": "Aurora Deck 2", "k": 5, "reasoning": "Searching knowledge collection for anything mention Aurora Deck 2"}) |
| 3 | 📥 tool result | You searched `knowledge` for "Aurora Deck 2": (read_similar result)<br>1 entry from `knowledge` (most relevant first):<br>1. [2026-07-16 01:09 UTC] key='aurora deck 2 price' $499 |
| 4 | 🔧 Penny → tool | collection_get({"key": "aurora deck 2 price", "memory": "knowledge", "reasoning": "Retrieve content for known key"}) |
| 5 | 📥 tool result | You looked up "aurora deck 2 price" in `knowledge`: (collection_get result)<br>1 entry from `knowledge`:<br>1. [2026-07-16 01:09 UTC] key='aurora deck 2 price' $499 |
| 6 | 🤖 Penny | I looked up “Aurora Deck 2” in the `knowledge` collection and pulled its stored entry—showing that it sold for **$499**. 🚀 |

_whole-run / missing-action checks: ✅ cold recall states $499 (storage is the only route) · ✅ answer BACKED by a storage read (find or a scoped read) · ✅ clean tool routing (no text-bail nudge fired)_

#### sample 2 — ✅ 3/3 checks

| # | Actor | Content |
|---|---|---|
| 1 | 👤 user | hey — a while back I asked you to remember what the aurora deck 2 was listed at. what was the price? |
| 2 | 🔧 Penny → tool | find({"query": "aurora deck 2", "reasoning": "Search my memory for the Aurora Deck 2 price that may have been stored in a collection or as a fact. Use find to locate any matching collection, log, skill, or stored entry."}) |
| 3 | 📥 tool result | You looked for something of your own for "aurora deck 2": (find result)<br>Found 5 things matching "aurora deck 2", best first:<br>1. entry key='aurora deck 2 price' in `knowledge` — "$499"<br>   read it: collection_get(memory='knowledge', key='aurora deck 2 price')<br>2. penny-messages — active log: Every outgoing Penny reply<br>   how to use it: read it with log_read('penny-messages')<br>3. thoughts — active collection: Penny's inner-monologue thoughts about the user's interests.<br>   how to use it: read it with collection_read_latest('thoughts'), reconfigure it with collection_update(name='thoughts', ...), archive it with collection_archive('thoughts')<br>4. user-messages — active log: Every incoming user message<br>   how to use it: read it with log_read('user-messages')<br>5. likes — active collection: Topics the user has expressed positive sentiment about<br>   how to use it: read it with collection_read_latest('likes'), reconfigure it with collection_update(name='likes', ...), archive it with collection_archive('likes')<br>Ranked by closeness — if one is what you meant, use its addressing above; otherwise narrow by its exact name. |
| 4 | 🔧 Penny → tool | collection_get({"memory": "knowledge", "key": "aurora deck 2 price", "reasoning": "Retrieve the stored entry for price of Aurora Deck 2 from knowledge"}) |
| 5 | 📥 tool result | You looked up "aurora deck 2 price" in `knowledge`: (collection_get result)<br>1 entry from `knowledge`:<br>1. [2026-07-16 01:09 UTC] key='aurora deck 2 price' $499 |
| 6 | 🔧 Penny → tool | browse({"queries": ["aurora deck 2"]}) |
| 7 | 📥 tool result | You searched for "aurora deck 2" (browse result)<br>## browse search: aurora deck 2<br>Mock search results |
| 8 | 🤖 Penny | You looked up "aurora deck 2 price" in `knowledge` and found the entry "$499" – that’s what I remembered for its price. 📦✨ |

_whole-run / missing-action checks: ✅ cold recall states $499 (storage is the only route) · ✅ answer BACKED by a storage read (find or a scoped read) · ✅ clean tool routing (no text-bail nudge fired)_

#### sample 3 — ✅ 3/3 checks

| # | Actor | Content |
|---|---|---|
| 1 | 👤 user | hey — a while back I asked you to remember what the aurora deck 2 was listed at. what was the price? |
| 2 | 🔧 Penny → tool | find({"query": "Aurora Deck 2 price", "reasoning": "Check if I have a remembered fact about the Aurora Deck 2 price in my collections."}) |
| 3 | 📥 tool result | You looked for something of your own for "Aurora Deck 2 price": (find result)<br>Found 5 things matching "Aurora Deck 2 price", best first:<br>1. entry key='aurora deck 2 price' in `knowledge` — "$499"<br>   read it: collection_get(memory='knowledge', key='aurora deck 2 price')<br>2. penny-messages — active log: Every outgoing Penny reply<br>   how to use it: read it with log_read('penny-messages')<br>3. browse-results — active log: Every browse-tool fetch result<br>   how to use it: read it with log_read('browse-results')<br>4. user-messages — active log: Every incoming user message<br>   how to use it: read it with log_read('user-messages')<br>5. thoughts — active collection: Penny's inner-monologue thoughts about the user's interests.<br>   how to use it: read it with collection_read_latest('thoughts'), reconfigure it with collection_update(name='thoughts', ...), archive it with collection_archive('thoughts')<br>Ranked by closeness — if one is what you meant, use its addressing above; otherwise narrow by its exact name. |
| 4 | 🔧 Penny → tool | collection_get({"memory": "knowledge", "key": "aurora deck 2 price", "reasoning": "Retrieve the stored fact about Aurora Deck 2 price."}) |
| 5 | 📥 tool result | You looked up "aurora deck 2 price" in `knowledge`: (collection_get result)<br>1 entry from `knowledge`:<br>1. [2026-07-16 01:09 UTC] key='aurora deck 2 price' $499 |
| 6 | 🤖 Penny | Got it—I checked the fact I had saved about the Aurora Deck 2 and it’s listed at **$499**. 🚀 |

_whole-run / missing-action checks: ✅ cold recall states $499 (storage is the only route) · ✅ answer BACKED by a storage read (find or a scoped read) · ✅ clean tool routing (no text-bail nudge fired)_

#### sample 4 — ✅ 3/3 checks

| # | Actor | Content |
|---|---|---|
| 1 | 👤 user | hey — a while back I asked you to remember what the aurora deck 2 was listed at. what was the price? |
| 2 | 🔧 Penny → tool | memory_metadata({"memory": "knowledge", "reasoning": "look up the definition of the knowledge collection to see what it contains"}) |
| 3 | 📥 tool result | You checked the details of `knowledge`: (memory_metadata result)<br>name: knowledge<br>type: collection<br>description: Summarized facts from web pages Penny has read<br><br>What it does each cycle — the recipe below is the collection's actual behaviour.  When explaining the collection, walk through THESE steps, not the operational settings.<br>extraction prompt: You extract durable knowledge from web pages Penny has read.<br><br>1. Call log_read("browse-results") to fetch new browse entries.  Each entry is one page (URL line, Title line, then page content).<br>2. For each page entry, write a single dense paragraph of 8-12 sentences capturing the key factual content.  Focus on:<br>   - What the thing IS (product, article, concept, etc.)<br>   - Specific details that would be useful to recall later (specs, names, dates, claims, findings)<br>   - What makes it notable or distinctive<br>   Do NOT include navigation/ads/site chrome, "This page describes..." meta-framing, opinions about content quality, or anything not on the page.  Plain declarative prose; no bullets, no markdown, no headers.<br>3. For each page, call collection_get("knowledge", key=<page title>) to see whether you already have a summary.  If one is returned, call update_entry("knowledge", key=<title>, content=<merged paragraph>) — integrate any new details from this fetch while preserving existing ones.  Otherwise, call collection_write("knowledge", entries=[{key: <title>, content: <new paragraph>}]).<br><br>The …[truncated] |
| 4 | 🔧 Penny → tool | collection_read_latest({"memory": "knowledge", "k": 1, "reasoning": "Get the most recent entry in knowledge, likely about a product"}) |
| 5 | 📥 tool result | You looked up your `knowledge`: (collection_read_latest result)<br>1 entry from `knowledge` (most recent first):<br>1. [2026-07-16 01:10 UTC] key='aurora deck 2 price' $499 |
| 6 | 🤖 Penny | We found an entry in the `knowledge` collection titled “aurora deck 2 price” – it says $499. No further details or source URLs were recorded for that entry, so that’s the last known listed amount. We'll give that and note it may have changed.I looked up your most recent knowledge fact about Aurora Deck 2 and found a stored entry saying its price was **$499**. That’s the latest quoted list‑price I’ve kept on record, though it might be out of date if the retailer updated it since then. 😊 |

_whole-run / missing-action checks: ✅ cold recall states $499 (storage is the only route) · ✅ answer BACKED by a storage read (find or a scoped read) · ✅ clean tool routing (no text-bail nudge fired)_

#### sample 5 — ✅ 3/3 checks

| # | Actor | Content |
|---|---|---|
| 1 | 👤 user | hey — a while back I asked you to remember what the aurora deck 2 was listed at. what was the price? |
| 2 | 🔧 Penny → tool | find({"query": "aurora deck 2 price", "reasoning": "Looking for any stored entry or collection that references the Aurora Deck 2 price."}) |
| 3 | 📥 tool result | You looked for something of your own for "aurora deck 2 price": (find result)<br>Found 5 things matching "aurora deck 2 price", best first:<br>1. entry key='aurora deck 2 price' in `knowledge` — "$499"<br>   read it: collection_get(memory='knowledge', key='aurora deck 2 price')<br>2. penny-messages — active log: Every outgoing Penny reply<br>   how to use it: read it with log_read('penny-messages')<br>3. thoughts — active collection: Penny's inner-monologue thoughts about the user's interests.<br>   how to use it: read it with collection_read_latest('thoughts'), reconfigure it with collection_update(name='thoughts', ...), archive it with collection_archive('thoughts')<br>4. browse-results — active log: Every browse-tool fetch result<br>   how to use it: read it with log_read('browse-results')<br>5. user-messages — active log: Every incoming user message<br>   how to use it: read it with log_read('user-messages')<br>Ranked by closeness — if one is what you meant, use its addressing above; otherwise narrow by its exact name. |
| 4 | 🔧 Penny → tool | collection_get({"memory": "knowledge", "key": "aurora deck 2 price", "reasoning": "Retrieve the stored price for Aurora Deck 2"}) |
| 5 | 📥 tool result | You looked up "aurora deck 2 price" in `knowledge`: (collection_get result)<br>1 entry from `knowledge`:<br>1. [2026-07-16 01:10 UTC] key='aurora deck 2 price' $499 |
| 6 | 🤖 Penny | You searched the `knowledge` collection for “aurora deck 2 price” and saw that it’s listed at **$499**. 😄 |

_whole-run / missing-action checks: ✅ cold recall states $499 (storage is the only route) · ✅ answer BACKED by a storage read (find or a scoped read) · ✅ clean tool routing (no text-bail nudge fired)_

