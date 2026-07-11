"""GenerateImageTool — model-driven image generation via the Ollama image model.

A thin wrapper over the same ``OllamaImageClient`` the retired ``/draw`` command
used.  Registered on the chat surface only when an image model is configured
(mirroring ``/draw``'s conditionality).

The generated image is delivered **deterministically** to its own reply, not
fuzzy-matched against the media table: the tool stores it in the ``media`` table
and stamps that row's id onto its ``ToolResult.media_id``.  The loop threads the
id onto ``ControllerResponse.generated_media_ids`` and the channel fetches
exactly that row at egress (see the image side-channel design in
``penny/CLAUDE.md``), so a just-drawn image always lands on the reply that
describes it — never a stale, embedding-nearest one.  The tool returns a text
result naming what it drew so the model's final reply honestly describes the
image the user is about to receive.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from penny.tools.base import Tool
from penny.tools.models import GenerateImageArgs, ToolResult

if TYPE_CHECKING:
    from penny.database import Database
    from penny.llm.image_client import OllamaImageClient

logger = logging.getLogger(__name__)

# Ollama's image-generation endpoint returns a base64-encoded PNG.
_GENERATED_IMAGE_MIME = "image/png"


class GenerateImageTool(Tool):
    """Generate an image from a text description and deliver it to the user."""

    name = "generate_image"
    description = (
        "Generate an image from a text description and send it to the user.  Use "
        "this when the user asks you to draw, paint, sketch, or make a "
        "picture/image of something.  Pass the full visual description as "
        "`description`; the image is delivered automatically with your reply, so "
        "your reply should tell the user their image is ready and describe what "
        "you drew."
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": (
                    "The full visual description of the image to generate — the "
                    "subject, style, and any details, drawn from what the user asked for."
                ),
            }
        },
        "required": ["description"],
    }
    args_model = GenerateImageArgs

    def __init__(self, image_client: OllamaImageClient, db: Database) -> None:
        self._image_client = image_client
        self._db = db

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Generate the image, store it for egress, and confirm what was drawn."""
        args = GenerateImageArgs(**kwargs)
        image_b64 = await self._image_client.generate_image(prompt=args.description)
        media_id = self._store_media(args.description, image_b64)
        logger.info("Generated image %d for description: %s", media_id, args.description)
        return ToolResult(
            message=(
                f"Generated an image of: {args.description}.  It will be delivered to "
                "the user with your reply — tell them their image is ready and describe "
                "what you drew."
            ),
            mutated=True,
            media_id=media_id,
        )

    def _store_media(self, description: str, image_b64: str) -> int:
        """Store the generated image and return its media id for deterministic egress.

        No embedding: the image is delivered by id to its own reply (the
        ``ToolResult.media_id`` link), not fuzzy-matched — so there is nothing
        to match on.
        """
        return self._db.media.put(
            data=base64.b64decode(image_b64),
            mime_type=_GENERATED_IMAGE_MIME,
            title=description,
        )

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Generating an image"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        """First-person narration of the image result (the #1481 per-tool override).

        The *result* twin of ``to_action_str``: the seam (``format_result``) adds
        the ``(generate_image result)`` tag; this returns only the sentence,
        branching on ``result.success`` so a failure narrates honestly.
        """
        subject = cls._drawn_subject(arguments)
        if not result.success:
            return f"You tried to draw {subject} but it didn't work:"
        return f"You drew {subject}:"

    @staticmethod
    def _drawn_subject(arguments: dict) -> str:
        """Quoted image description for narration, or a generic noun when the call
        omitted it (an arg-validation failure still narrates)."""
        description = arguments.get("description")
        return f'"{description}"' if description else "your image"
