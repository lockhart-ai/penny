"""GenerateImageTool — model-driven image generation via the Ollama image model.

A thin wrapper over the same ``OllamaImageClient`` the retired ``/draw`` command
used.  Registered on the chat surface only when an image model is configured
(mirroring ``/draw``'s conditionality).

The generated image rides the **existing media side-channel** rather than
travelling through the model: the tool stores it in the ``media`` table with an
embedding of the description, and at egress ``MediaStore.select_image`` attaches
it to the mirror-back reply — the same path browsed images use (see the image
side-channel design in ``penny/CLAUDE.md``).  The tool returns a text result
naming what it drew so the model's final reply honestly describes the image the
user is about to receive.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any

from penny.llm.embeddings import serialize_embedding
from penny.llm.similarity import embed_text
from penny.tools.base import Tool
from penny.tools.models import GenerateImageArgs, ToolResult

if TYPE_CHECKING:
    from penny.database import Database
    from penny.llm.client import LlmClient
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

    def __init__(
        self, image_client: OllamaImageClient, db: Database, embedding_client: LlmClient
    ) -> None:
        self._image_client = image_client
        self._db = db
        self._embedding_client = embedding_client

    async def execute(self, **kwargs: Any) -> ToolResult:
        """Generate the image, store it for egress, and confirm what was drawn."""
        args = GenerateImageArgs(**kwargs)
        image_b64 = await self._image_client.generate_image(prompt=args.description)
        await self._store_media(args.description, image_b64)
        logger.info("Generated image for description: %s", args.description)
        return ToolResult(
            message=(
                f"Generated an image of: {args.description}.  It will be delivered to "
                "the user with your reply — tell them their image is ready and describe "
                "what you drew."
            ),
            mutated=True,
        )

    async def _store_media(self, description: str, image_b64: str) -> None:
        """Store the generated image so egress attaches it to the mirror-back reply.

        The embedding of the description lets ``MediaStore.select_image`` match
        the image to the reply text (which describes the same subject) at egress.
        """
        vector = await embed_text(self._embedding_client, description)
        embedding = serialize_embedding(vector) if vector else None
        self._db.media.put(
            data=base64.b64decode(image_b64),
            mime_type=_GENERATED_IMAGE_MIME,
            title=description,
            embedding=embedding,
        )

    @classmethod
    def to_action_str(cls, arguments: dict) -> str:
        return "Generating an image"

    @classmethod
    def to_result_narration(cls, arguments: dict, result: ToolResult) -> str:
        description = arguments.get("description")
        phrase = f' "{description}"' if description else ""
        if not result.success:
            return f"You tried to draw{phrase} but it didn't work:"
        return f"You drew{phrase}:"
