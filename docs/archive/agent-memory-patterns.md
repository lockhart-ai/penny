# Agent Memory Patterns & Architecture Guide

A practical reference for structuring memory in a locally-built AI agent, with patterns, tradeoffs, and implementation sketches.

---

## The Core Problem

Without memory, an agent is stateless — it reasons well within a single prompt but forgets everything between invocations. Memory transforms a stateless LLM wrapper into something that can learn from experience, accumulate knowledge, maintain context across sessions, and improve its own workflows over time.

The challenge isn't just "store stuff and retrieve it." It's deciding *what* to remember, *how* to organize it, *when* to consolidate or forget, and *how* to surface the right memories at the right time without blowing up your context window.

---

## Memory Types

The field has converged on a taxonomy borrowed from cognitive science. Each type serves a different function and has different storage/retrieval characteristics.

### Working Memory (Short-Term)

**What it is:** The agent's scratchpad during a single task — intermediate reasoning, partial results, hypotheses being evaluated. Equivalent to what's in the LLM's context window right now.

**Lifecycle:** Created at task start, discarded at task end.

**Implementation:** This is typically just your prompt/context management. The key design decision is *what goes in and what gets evicted* when you're pushing context limits.

```python
class WorkingMemory:
    """Manages the agent's active reasoning context."""

    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.scratchpad: list[dict] = []
        self.current_plan: str | None = None
        self.evidence: list[dict] = []  # sources tracked per-item

    def add(self, item: dict, priority: int = 0):
        self.scratchpad.append({**item, "_priority": priority})
        self._evict_if_needed()

    def _evict_if_needed(self):
        """Drop lowest-priority items when approaching token limit."""
        while self._estimate_tokens() > self.max_tokens:
            self.scratchpad.sort(key=lambda x: x["_priority"])
            self.scratchpad.pop(0)

    def to_context(self) -> str:
        """Serialize for injection into the LLM prompt."""
        parts = []
        if self.current_plan:
            parts.append(f"## Current Plan\n{self.current_plan}")
        if self.evidence:
            parts.append("## Evidence Gathered\n" + "\n".join(
                f"- [{e['source']}] {e['content']}" for e in self.evidence
            ))
        if self.scratchpad:
            parts.append("## Scratchpad\n" + "\n".join(
                item["content"] for item in self.scratchpad
            ))
        return "\n\n".join(parts)
```

**Key pattern — Bounded Working Memory:** Always cap your working memory and implement an eviction strategy. The agent should track what it knows, where each piece came from, and what it still needs. This prevents context window bloat and forces the agent to be selective.

---

### Episodic Memory

**What it is:** Records of specific events — conversations, tool calls, task outcomes, errors encountered. Always timestamped and contextual. Not "users prefer dark mode" (that's semantic), but "on Feb 12, the user asked me to switch to dark mode and I updated their settings."

**Why it matters:** Lets the agent learn from specific past experiences, recognize recurring situations, and avoid repeating mistakes.

**Storage patterns:**

```python
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Episode:
    """A single recorded event in the agent's history."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = ""          # "conversation", "tool_call", "error", "task_complete"
    actor: str = ""               # "user", "agent", "system"
    content: str = ""             # what happened
    context: dict = field(default_factory=dict)   # surrounding state
    outcome: str | None = None    # result or resolution
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None  # for semantic retrieval


class EpisodicStore:
    """
    Stores and retrieves episodes. In practice, back this with
    a database — SQLite + pgvector, or a dedicated vector DB.
    """

    def __init__(self, embed_fn):
        self.episodes: list[Episode] = []
        self.embed = embed_fn

    def record(self, episode: Episode):
        episode.embedding = self.embed(episode.content)
        self.episodes.append(episode)

    def recall_recent(self, n: int = 10) -> list[Episode]:
        """Simple recency-based retrieval."""
        return sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)[:n]

    def recall_similar(self, query: str, n: int = 5) -> list[Episode]:
        """Semantic similarity retrieval."""
        query_vec = self.embed(query)
        scored = [
            (ep, cosine_sim(query_vec, ep.embedding))
            for ep in self.episodes if ep.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:n]]

    def recall_by_entity(self, entity: str, n: int = 10) -> list[Episode]:
        """Retrieve episodes involving a specific entity."""
        return [
            ep for ep in self.episodes
            if entity.lower() in ep.content.lower()
               or entity in ep.tags
        ][-n:]
```

**Key pattern — Episode Boundaries:** One of the trickiest design decisions. Too coarse (one episode = an entire session) and you lose granularity. Too fine (one episode = one message) and you lose context. A good heuristic: one episode per *task* or *intent shift*. If the user asks you to do three unrelated things in one conversation, that's three episodes.

---

### Semantic Memory

**What it is:** Abstracted facts and knowledge, divorced from when or how they were learned. User preferences, domain knowledge, entity relationships, learned rules. This is your agent's "world model."

**Why it matters:** This is what makes an agent feel like it *knows* you and *knows* its domain, rather than rediscovering everything each time.

**Storage patterns:**

```python
@dataclass
class SemanticFact:
    """A single piece of knowledge the agent has learned."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = ""         # entity this fact is about
    predicate: str = ""       # relationship or attribute
    value: str = ""           # the fact itself
    confidence: float = 1.0   # how sure are we
    source_episodes: list[str] = field(default_factory=list)  # provenance
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    valid_from: datetime | None = None   # temporal validity
    valid_until: datetime | None = None
    embedding: list[float] | None = None


class SemanticStore:
    """
    Fact store with conflict resolution.
    In production, back with pgvector or a graph DB.
    """

    def __init__(self, embed_fn, llm_fn):
        self.facts: dict[str, SemanticFact] = {}
        self.embed = embed_fn
        self.llm = llm_fn  # for conflict resolution

    def upsert(self, new_fact: SemanticFact):
        """Add or update a fact, resolving conflicts."""
        existing = self._find_conflicting(new_fact)
        if existing:
            resolved = self._resolve_conflict(existing, new_fact)
            self.facts[resolved.id] = resolved
        else:
            new_fact.embedding = self.embed(
                f"{new_fact.subject} {new_fact.predicate} {new_fact.value}"
            )
            self.facts[new_fact.id] = new_fact

    def _find_conflicting(self, fact: SemanticFact) -> SemanticFact | None:
        """Find existing facts about the same subject+predicate."""
        for f in self.facts.values():
            if f.subject == fact.subject and f.predicate == fact.predicate:
                return f
        return None

    def _resolve_conflict(self, old: SemanticFact, new: SemanticFact) -> SemanticFact:
        """Use LLM to decide: update, merge, or keep both."""
        prompt = f"""Two facts conflict. Decide how to resolve:
        Existing: {old.subject} {old.predicate} {old.value} (confidence: {old.confidence})
        New: {new.subject} {new.predicate} {new.value} (confidence: {new.confidence})
        Should we: UPDATE (replace old), MERGE (combine), or KEEP_BOTH?"""

        decision = self.llm(prompt)
        if "UPDATE" in decision:
            old.value = new.value
            old.updated_at = datetime.utcnow()
            old.source_episodes.extend(new.source_episodes)
            return old
        # ... handle MERGE and KEEP_BOTH
        return new

    def query(self, question: str, n: int = 5) -> list[SemanticFact]:
        """Semantic search over facts."""
        q_vec = self.embed(question)
        scored = [
            (f, cosine_sim(q_vec, f.embedding))
            for f in self.facts.values() if f.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored[:n]]
```

**Key pattern — Fact Conflict Resolution:** When the agent learns "user lives in Toronto" but already knows "user lives in Montreal," it needs a strategy. Common approaches: last-write-wins (simple but lossy), LLM-mediated resolution (smarter but more expensive), or temporal validity tracking (mark the old fact as expired, keep both with date ranges).

---

### Procedural Memory

**What it is:** Learned workflows, strategies, and behavioral patterns. How the agent does things, not what it knows. "When a user asks to deploy, run tests first, then build, then deploy to staging before prod" — that's procedural.

**Why it matters:** This is what separates an agent that stumbles through the same task differently every time from one that gets consistently better at recurring workflows.

```python
@dataclass
class Procedure:
    """A learned workflow or behavioral pattern."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    trigger: str = ""          # when to activate this procedure
    steps: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    last_used: datetime | None = None
    notes: str = ""            # lessons learned, edge cases

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class ProceduralStore:
    def __init__(self, llm_fn):
        self.procedures: dict[str, Procedure] = {}
        self.llm = llm_fn

    def find_applicable(self, situation: str) -> list[Procedure]:
        """Find procedures that match the current situation."""
        candidates = []
        for proc in self.procedures.values():
            # Simple keyword matching; upgrade to semantic matching in production
            if any(kw in situation.lower() for kw in proc.trigger.lower().split()):
                candidates.append(proc)
        # Rank by success rate
        candidates.sort(key=lambda p: p.success_rate, reverse=True)
        return candidates

    def record_outcome(self, proc_id: str, success: bool, notes: str = ""):
        """Update a procedure based on execution outcome."""
        proc = self.procedures.get(proc_id)
        if not proc:
            return
        if success:
            proc.success_count += 1
        else:
            proc.failure_count += 1
            proc.notes += f"\nFailure ({datetime.utcnow().isoformat()}): {notes}"
        proc.last_used = datetime.utcnow()

    def learn_from_episode(self, episode: Episode):
        """Extract and store a new procedure from a successful task."""
        if episode.outcome and "success" in episode.outcome.lower():
            prompt = f"""Extract a reusable procedure from this experience:
            {episode.content}
            Outcome: {episode.outcome}
            Return: name, trigger condition, and ordered steps."""

            result = self.llm(prompt)
            # Parse and store the new procedure
            # ...
```

**Key pattern — Procedure Evolution:** Procedures shouldn't be static. Track success/failure rates, annotate with edge cases, and periodically have the LLM review and refine procedures based on accumulated outcomes. An agent that does this is genuinely learning.

---

## Storage Backend Architectures

### Option 1: Vector Store (Simplest Starting Point)

Use a vector database (or pgvector in Postgres, which you're likely already running with Django) to store embeddings of memories. Retrieve by semantic similarity.

**Best for:** Semantic memory, fuzzy "find me something relevant" retrieval.

**Limitation:** No structured relationships, poor at temporal or multi-hop reasoning.

```
┌─────────────────────────────┐
│         Agent Core          │
├─────────────────────────────┤
│      Memory Manager         │
├─────────────────────────────┤
│   pgvector (PostgreSQL)     │
│   ┌───────────────────────┐ │
│   │ memories              │ │
│   │ - id                  │ │
│   │ - content             │ │
│   │ - memory_type (enum)  │ │
│   │ - embedding (vector)  │ │
│   │ - metadata (jsonb)    │ │
│   │ - created_at          │ │
│   │ - updated_at          │ │
│   │ - expires_at          │ │
│   └───────────────────────┘ │
└─────────────────────────────┘
```

Django model sketch:

```python
from pgvector.django import VectorField

class Memory(models.Model):
    class MemoryType(models.TextChoices):
        EPISODIC = "episodic"
        SEMANTIC = "semantic"
        PROCEDURAL = "procedural"

    content = models.TextField()
    memory_type = models.CharField(max_length=20, choices=MemoryType.choices)
    embedding = VectorField(dimensions=1536)
    metadata = models.JSONField(default=dict)
    subject = models.CharField(max_length=255, blank=True, db_index=True)
    confidence = models.FloatField(default=1.0)
    source_episode_ids = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        indexes = [
            # pgvector index for similarity search
            HnswIndex(
                name="memory_embedding_idx",
                fields=["embedding"],
                m=16, ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]
```

### Option 2: Knowledge Graph (Better Relational Reasoning)

Use a graph database (Neo4j, or lightweight options like NetworkX for local dev) to represent entities and relationships. Enables multi-hop queries like "what tools does the user prefer for projects involving streaming?"

**Best for:** Entity relationships, temporal reasoning, multi-agent shared state.

```
┌──────────────────────────────────────────┐
│              Agent Core                  │
├──────────────────────────────────────────┤
│           Memory Manager                 │
├───────────────┬──────────────────────────┤
│  Vector Store │    Knowledge Graph       │
│  (pgvector)   │    (Neo4j / NetworkX)    │
│               │                          │
│  embeddings   │  (User)──[prefers]──►    │
│  for semantic │     (DRF)                │
│  search       │  (User)──[works_on]──►   │
│               │     (StreamingAPI)        │
│               │  (StreamingAPI)──         │
│               │    [uses]──►(NDJSON)      │
└───────────────┴──────────────────────────┘
```

**Key pattern — Bi-Temporal Modeling:** Track two timelines for every relationship: *when the fact became true in reality* and *when the agent learned it*. This lets you answer both "what did I know at time X?" and "what was actually true at time X?" — critical for debugging and for handling corrections.

### Option 3: Hybrid (Production Recommendation)

Combine vector search for fuzzy retrieval with a graph or relational layer for structured queries. This is what Mem0, Zep, and most production systems do.

```
┌──────────────────────────────────────────────────┐
│                   Agent Core                     │
├──────────────────────────────────────────────────┤
│              Unified Memory Manager              │
│  ┌────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │  Working    │ │  Retrieval  │ │Consolidation│ │
│  │  Memory     │ │  Router     │ │  Engine     │ │
│  │  (in-proc)  │ │             │ │  (async)    │ │
│  └────────────┘ └──────┬──────┘ └──────┬──────┘ │
├─────────────────────────┼──────────────┼─────────┤
│         Storage Layer   │              │         │
│  ┌──────────────────────┴──────────────┴───────┐ │
│  │                                             │ │
│  │  PostgreSQL                                 │ │
│  │  ├── memories table (pgvector)              │ │
│  │  ├── episodes table                         │ │
│  │  ├── entities table                         │ │
│  │  ├── relationships table (lightweight graph)│ │
│  │  └── procedures table                       │ │
│  │                                             │ │
│  └─────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

You don't necessarily need Neo4j. A relational "graph" using a `relationships` table with `subject_id`, `predicate`, `object_id`, and temporal fields gets you 80% of the value for local development with zero additional infrastructure.

---

## Key Operational Patterns

### 1. Memory Consolidation

The most important pattern for long-running agents. Periodically distill episodic memories into semantic facts and procedural knowledge.

```python
class MemoryConsolidator:
    """
    Runs on a schedule (or after N episodes) to compress
    episodic memory into semantic facts and procedures.
    """

    def __init__(self, episodic_store, semantic_store, procedural_store, llm_fn):
        self.episodic = episodic_store
        self.semantic = semantic_store
        self.procedural = procedural_store
        self.llm = llm_fn

    def consolidate(self, since: datetime | None = None):
        recent = self.episodic.recall_recent(n=50)
        if since:
            recent = [ep for ep in recent if ep.timestamp >= since]

        if not recent:
            return

        # Step 1: Extract semantic facts
        episodes_text = "\n".join(
            f"[{ep.timestamp}] {ep.event_type}: {ep.content}" for ep in recent
        )
        fact_prompt = f"""Review these recent episodes and extract durable facts
        (user preferences, learned information, entity attributes).
        Only extract facts that are likely to remain true and useful.
        Skip transient details.

        Episodes:
        {episodes_text}

        Return as JSON list: [{{"subject": "", "predicate": "", "value": ""}}]"""

        facts = self.llm(fact_prompt)
        for fact_data in parse_json(facts):
            self.semantic.upsert(SemanticFact(**fact_data, source_episodes=[
                ep.id for ep in recent
            ]))

        # Step 2: Extract procedures from successful task sequences
        successful = [ep for ep in recent if ep.outcome and "success" in ep.outcome.lower()]
        if successful:
            proc_prompt = f"""Review these successful task completions.
            Are there any reusable workflows or patterns worth remembering?

            {chr(10).join(f"- {ep.content} -> {ep.outcome}" for ep in successful)}

            Return reusable procedures as JSON if any exist, else empty list."""

            procedures = self.llm(proc_prompt)
            for proc_data in parse_json(procedures):
                self.procedural.learn(proc_data)

        # Step 3: (Optional) Prune or compress old episodes
        # Keep summaries, drop raw content for episodes older than N days
```

**When to run:** After every N interactions, on a schedule, or triggered by a "session end" event. For a local agent, a simple "consolidate on shutdown" or "consolidate every 20 episodes" works fine.

### 2. Retrieval Router

Not every query needs every memory type. Route retrieval based on what the agent is trying to do.

```python
class RetrievalRouter:
    """
    Decides which memory stores to query and how to combine results.
    """

    def retrieve(self, query: str, task_type: str = "general") -> dict:
        results = {"working": [], "episodic": [], "semantic": [], "procedural": []}

        # Always include relevant semantic facts
        results["semantic"] = self.semantic_store.query(query, n=5)

        if task_type == "recall":
            # "What happened when..." or "Last time we..."
            results["episodic"] = self.episodic_store.recall_similar(query, n=5)

        elif task_type == "execute":
            # "Deploy the app" or "Run the migration"
            results["procedural"] = self.procedural_store.find_applicable(query)
            # Also grab recent relevant episodes for context
            results["episodic"] = self.episodic_store.recall_similar(query, n=3)

        elif task_type == "general":
            # Cast a wider net
            results["episodic"] = self.episodic_store.recall_similar(query, n=3)
            results["procedural"] = self.procedural_store.find_applicable(query)

        return results

    def format_for_context(self, results: dict) -> str:
        """Format retrieved memories for injection into the LLM prompt."""
        sections = []

        if results["semantic"]:
            facts = "\n".join(
                f"- {f.subject}: {f.predicate} = {f.value}" for f in results["semantic"]
            )
            sections.append(f"## Known Facts\n{facts}")

        if results["procedural"]:
            procs = "\n".join(
                f"- {p.name} (success rate: {p.success_rate:.0%}): {' → '.join(p.steps)}"
                for p in results["procedural"]
            )
            sections.append(f"## Relevant Procedures\n{procs}")

        if results["episodic"]:
            eps = "\n".join(
                f"- [{e.timestamp.strftime('%Y-%m-%d')}] {e.content[:200]}"
                for e in results["episodic"]
            )
            sections.append(f"## Relevant Past Experiences\n{eps}")

        return "\n\n".join(sections)
```

### 3. Memory-Aware Prompt Assembly

The final pattern — how retrieved memories actually get injected into the LLM call. This is where context engineering meets memory architecture.

```python
class AgentPromptBuilder:
    """
    Assembles the final prompt with memory context.
    Budget-aware: won't exceed the token allocation for memory.
    """

    def __init__(self, memory_token_budget: int = 4000):
        self.budget = memory_token_budget

    def build(
        self,
        system_prompt: str,
        user_message: str,
        working_memory: WorkingMemory,
        retrieved_memories: dict,
    ) -> list[dict]:

        memory_context = self._build_memory_context(retrieved_memories)
        working_context = working_memory.to_context()

        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"## Agent Memory\n{memory_context}"},
            {"role": "system", "content": f"## Current State\n{working_context}"},
            {"role": "user", "content": user_message},
        ]

    def _build_memory_context(self, memories: dict) -> str:
        """
        Build memory context within token budget.
        Priority: semantic > procedural > episodic
        """
        sections = []
        remaining = self.budget

        # Semantic facts get top priority (usually compact)
        if memories.get("semantic"):
            block = self._format_semantic(memories["semantic"])
            tokens = estimate_tokens(block)
            if tokens <= remaining:
                sections.append(block)
                remaining -= tokens

        # Procedures next (actionable)
        if memories.get("procedural") and remaining > 200:
            block = self._format_procedural(memories["procedural"])
            tokens = estimate_tokens(block)
            if tokens <= remaining:
                sections.append(block)
                remaining -= tokens

        # Episodes fill remaining budget (most verbose)
        if memories.get("episodic") and remaining > 200:
            block = self._format_episodic(memories["episodic"], max_tokens=remaining)
            sections.append(block)

        return "\n\n".join(sections) if sections else "No relevant memories."
```

---

## Anti-Patterns to Avoid

**Remember Everything** — Storing every message and interaction verbatim without consolidation. Your context window fills with noise, retrieval gets slower, and the agent can't distinguish signal from chatter. Be aggressive about consolidation and expiry.

**Flat Memory Dump** — Retrieving memories and dumping them all into the prompt without structure or prioritization. The LLM can't effectively use 50 unranked memory fragments. Curate and structure what you inject.

**No Provenance Tracking** — Storing facts without tracking where they came from. When a fact turns out to be wrong, you need to trace it back to the source episode and understand what went wrong. Always keep `source_episode_ids`.

**Static Procedures** — Defining workflows once and never updating them. If a procedure starts failing, the agent should notice and adapt. Track success/failure rates and trigger re-evaluation.

**Missing Temporal Validity** — Treating all facts as eternally true. "User's current project is the streaming API" will eventually become stale. Add `valid_from`/`valid_until` fields and build expiry logic into retrieval.

---

## Implementation Roadmap for a Local Agent

If you're building this incrementally on top of your existing Django stack:

**Phase 1 — Foundation (start here)**
Use your existing PostgreSQL. Add `pgvector` extension. Create a single `Memory` model with a `memory_type` enum. Implement basic semantic search for retrieval. This gets you working semantic memory with minimal infrastructure.

**Phase 2 — Episodic Logging**
Add an `Episode` model. Log every agent interaction (tool calls, user messages, outcomes). Implement recency + similarity retrieval. Build a simple consolidation job that extracts semantic facts from recent episodes on a schedule or trigger.

**Phase 3 — Procedural Learning**
Add a `Procedure` model. Start extracting procedures from successful multi-step tasks. Track success/failure rates. Wire procedure retrieval into the prompt assembly so the agent can reference its own playbooks.

**Phase 4 — Graph Layer (if needed)**
If you find yourself needing multi-hop relationship queries (e.g., "what tools are associated with projects that had timeout issues?"), add a lightweight graph layer. Start with a `relationships` table in Postgres before reaching for Neo4j. Evaluate Graphiti/Zep if you need full temporal knowledge graph capabilities.

**Phase 5 — Reflection & Evolution**
Add a periodic reflection loop where the agent reviews its own memory for contradictions, outdated facts, and procedure improvements. This is what makes the system genuinely self-improving rather than just accumulating data.

---

## Frameworks & Tools Worth Evaluating

| Framework | What It Does | Best For |
|-----------|-------------|----------|
| **Mem0** | Drop-in long-term memory layer with hybrid vector + graph | Fastest path to production memory |
| **Graphiti / Zep** | Temporal knowledge graph, bi-temporal model, hybrid search | When you need relationship + temporal reasoning |
| **LangGraph** | Checkpointers + cross-thread memory stores | If you're already in the LangChain ecosystem |
| **pgvector** | Vector similarity search in PostgreSQL | Staying in your existing Django/Postgres stack |
| **A-Mem** | Self-organizing memory inspired by Zettelkasten | Research-oriented, interesting for autonomous agents |

For a Django-based local agent, **pgvector** is the most natural starting point — zero new infrastructure, and you can build all four memory types on top of it with the patterns above. Layer in a graph component later if the relational queries justify it.
