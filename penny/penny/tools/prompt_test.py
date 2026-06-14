"""prompt_test — dry-run a candidate extraction_prompt before applying it.

The self-correction primitive: the quality collector drafts a fix for a
collection's extraction_prompt, calls this to see what a cycle with that prompt
WOULD do (messages it would send, entries it would write) without applying
anything, and only commits the fix via collection_update once the dry run is
clean.  The simulation runs on a throwaway dry-run collector — see
``Collector.dry_run`` — with captured side effects and non-consuming reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from penny.tools.base import Tool

if TYPE_CHECKING:
    from penny.agents.collector import Collector


class PromptTestArgs(BaseModel):
    collection: str
    extraction_prompt: str


class PromptTestTool(Tool):
    name = "prompt_test"
    description = (
        "Dry-run a candidate extraction_prompt for a collection WITHOUT applying it. "
        "Returns what the collector cycle would do — how many messages it would send "
        "the user and how many entries it would write/edit — so you can confirm a fix "
        "before committing it with collection_update. Always prompt_test a change "
        "before you apply it, and revise if the dry run still violates the intent."
    )
    parameters = {
        "type": "object",
        "properties": {
            "collection": {
                "type": "string",
                "description": "The collection whose cycle to simulate.",
            },
            "extraction_prompt": {
                "type": "string",
                "description": "The full candidate extraction_prompt body to test.",
            },
        },
        "required": ["collection", "extraction_prompt"],
    }

    def __init__(self, collector: Collector) -> None:
        self._collector = collector

    async def execute(self, **kwargs: Any) -> str:
        args = PromptTestArgs(**kwargs)
        return await self._collector.dry_run(args.collection, args.extraction_prompt)
