# Archived Design Plans & Notes

These documents are plans and notes kept as a historical record, each describing a design as it was planned at the time and not Penny as she is today. They document earlier iterations and design thinking.

- **task-framework-plan.md** — Unified data layer and task framework design to transform Penny's behavior from hardcoded Python into user-authored tasks and memories.
- **memory-implementation-plan.md** — Twelve-stage tactical rollout plan for unified memory schema, access layer, and per-agent migration to the new memory primitive.
- **channel-manager-plan.md** — Multi-channel architecture unifying Signal, Discord, and browser connections into a single conversation history with device routing.
- **browser-tools-plan.md** — Browser extension tool execution flow with permission model, domain allowlist, and WebSocket request/response protocol.
- **tool-narration-plan.md** — Self-narrating tool design contract for tagged first-person result framing and tool-call transparency in agent replies.
- **collector-run-health-reasoning-plan.md** — Structural signals and validation discipline to give Penny observable facts about collector run outcomes and quality self-review.
- **prompt-and-recall-efficiency.md** — Optimization of Penny's prompts through static/dynamic split and context-window efficiency to reduce per-turn latency.
- **self-improvement-loop.md** — Eval isolation, intent-as-spec, and the log-to-test-to-fix loop for managed self-correction and model measurement.
- **agent-memory-patterns.md** — Practical reference guide for memory architecture patterns including working memory, episodic memory, and semantic consolidation.
- **conversation-machine.md** — Reference for the conversation state machine, round framing, skill extraction, and the skill substrate — all retired by #2537's naive-jobs baseline, which replaced a taught, state-driven routine with a collection running the free-text prompt it was configured with.
