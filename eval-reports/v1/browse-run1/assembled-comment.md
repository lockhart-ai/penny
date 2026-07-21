### run-20260721T051017Z-abba710a

- commit: `abba710a03ae3555148fea6a86712e9af020499a`
- model: `gpt-oss:20b`
- N: 3
- lever: framework baseline (representative case switched to chat-browse — prior case outmoded)

## Run totals

mean 0.67 · all-pass 2/3
pathology-excluded mean 0.67 (3 samples) · causes — behavioral 0 · pathology 0 · harness 1

### `chat-browse-answer` — browse-answer

**RESULT:** mean 0.67 · all-pass 2/3
pathology-excluded mean 0.67 (3 samples) · causes — behavioral 0 · pathology 0 · harness 1

<details><summary>transcripts — chat-browse-answer</summary>

#### sample 1 — ✅ 2/2 checks

| # | Actor | Content |
|---|---|---|
| 1 | 👤 user | what's the deepest lake in the world? |
| 2 | 👤 user | Instruction: maximum depth in meters<br><br>Content:<br>## browse: https://en.wikipedia.org/wiki/Lake_Baikal<br>Title: The world's deepest lakes<br>A rundown of the deepest freshwater lakes on Earth.<br>[Lake Baikal — profile](https://geo.example.test/lakes/baikal)<br>Lake Baikal, in southern Siberia, is the deepest lake in the world at 1,642 metres, holding roughly a fifth of Earth's unfrozen fresh water. |
| 3 | 🔧 Penny → tool ✅ | browse({"queries": ["https://en.wikipedia.org/wiki/Lake_Baikal"], "extract": "maximum depth in meters", "reasoning": "User asked for deepest lake; we know it's Lake Baikal but let's verify with source. Extract the max depth."}) |
| 4 | 📥 tool result | You opened https://en.wikipedia.org/wiki/Lake_Baikal (browse result)<br>## browse: https://en.wikipedia.org/wiki/Lake_Baikal<br>1642 |
| 5 | 🤖 Penny ✅ | I looked up Lake Baikal on Wikipedia and it reports a maximum depth of **1,642 meters**.  <br>So the deepest lake in the world is Lake Baikal, Russia. 🌊 |

#### sample 2 — ✅ 2/2 checks

| # | Actor | Content |
|---|---|---|
| 1 | 👤 user | what's the deepest lake in the world? |
| 2 | 👤 user | Instruction: the deepest point of the lake<br><br>Content:<br>## browse: https://en.wikipedia.org/wiki/Lake_Baikal<br>Title: The world's deepest lakes<br>A rundown of the deepest freshwater lakes on Earth.<br>[Lake Baikal — profile](https://geo.example.test/lakes/baikal)<br>Lake Baikal, in southern Siberia, is the deepest lake in the world at 1,642 metres, holding roughly a fifth of Earth's unfrozen fresh water. |
| 3 | 🔧 Penny → tool ✅ | browse({"extract": "the deepest point of the lake", "queries": ["https://en.wikipedia.org/wiki/Lake_Baikal"], "reasoning": "User asked about deepest lake; fetch page for Lake Baikal to get depth info."}) |
| 4 | 📥 tool result | You opened https://en.wikipedia.org/wiki/Lake_Baikal (browse result)<br>## browse: https://en.wikipedia.org/wiki/Lake_Baikal<br>1,642 metres |
| 5 | 🤖 Penny ✅ | I pulled the page on Lake Baikal and it turns out the maximum depth there is **1,642 metres** (about 5,389 ft).  <br>Source: https://en.wikipedia.org/wiki/Lake_Baikal 🏞️ |

</details>
