"""Pure ``to_result_narration`` for the lifecycle tools without a dedicated tool
test home — schedule create/delete/list and image generation.

The #1481 overrides on the #1479 seam: each returns only the first-person
sentence (``format_result`` adds the ``(<tool> result)`` tag), branching on
success / failure.  These tools have no successful-no-op — a create/delete either
mutates or fails, and generate_image either draws or errors at the framework
layer — so they narrate two ways.
"""

from __future__ import annotations

from penny.tools.generate_image import GenerateImageTool
from penny.tools.models import ToolResult
from penny.tools.schedule_tools import ScheduleCreateTool, ScheduleDeleteTool, ScheduleListTool

_MUTATED = ToolResult(message="ok", mutated=True)
_OK = ToolResult(message="ok")
_FAILED = ToolResult(message="Error", success=False)


def test_schedule_narration():
    assert ScheduleCreateTool.to_result_narration({"request": "daily at 9"}, _MUTATED) == (
        "You set up a recurring task for the user:"
    )
    assert ScheduleCreateTool.to_result_narration({"request": "daily at 9"}, _FAILED) == (
        "You tried to set up a scheduled task but it didn't work:"
    )
    assert ScheduleDeleteTool.to_result_narration({"description": "the digest"}, _MUTATED) == (
        "You removed a scheduled task:"
    )
    assert ScheduleDeleteTool.to_result_narration({"description": "the digest"}, _FAILED) == (
        "You tried to remove a scheduled task but it didn't work:"
    )
    assert ScheduleListTool.to_result_narration({}, _OK) == (
        "You checked the user's scheduled tasks:"
    )


def test_generate_image_narration():
    assert GenerateImageTool.to_result_narration({"description": "a red fox"}, _MUTATED) == (
        'You drew "a red fox":'
    )
    assert GenerateImageTool.to_result_narration({"description": "a red fox"}, _FAILED) == (
        'You tried to draw "a red fox" but it didn\'t work:'
    )
    # No description (only reachable via an arg-validation failure) drops the quoted
    # phrase rather than rendering an empty "".
    assert GenerateImageTool.to_result_narration({}, _FAILED) == (
        "You tried to draw but it didn't work:"
    )
