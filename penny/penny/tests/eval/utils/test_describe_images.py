"""The describe-images tool reads prod media read-only and describes it on the
resolved roster model (#3065).

The things pinned here: the image rides the user message as a data URI carrying the
row's OWN mime type (the path ``LlmClient`` translates to ``image_url`` parts), the
instruction is the one fixed neutral wording for every image, a failed call is
recorded and the batch continues, and ``RECENT`` selects captured images only — a
generated image's row has no ``source_url``, and the measurement pools never
include one.
"""

from __future__ import annotations

import base64
import json
import sqlite3

import pytest

from penny.llm.call_link import CallLink
from penny.llm.models import LlmError, LlmMessage, LlmResponse
from penny.tests.eval.utils.describe_images import (
    DESCRIPTOR_INSTRUCTION,
    MediaImage,
    _load_by_ids,
    _load_recent,
    describe_rows,
    select_media,
)

_PNG_BYTES = b"\x89PNG fake image bytes"
_JPEG_BYTES = b"\xff\xd8 fake jpeg bytes"


def _write_media_db(tmp_path, rows: list[dict]) -> str:
    """A throwaway media DB, exactly the columns ``describe_images`` reads."""
    db_path = str(tmp_path / "media.db")
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE media (id INTEGER PRIMARY KEY, mime_type TEXT, data BLOB, "
            "source_url TEXT, title TEXT, embedding BLOB, created_at TEXT)"
        )
        for row in rows:
            connection.execute(
                "INSERT INTO media (id, mime_type, data, source_url, title, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, NULL, ?)",
                (
                    row["id"],
                    row["mime_type"],
                    row["data"],
                    row.get("source_url"),
                    row.get("title"),
                    row.get("created_at"),
                ),
            )
        connection.commit()
    finally:
        connection.close()
    return db_path


def _captured(media_id: int, created_at: str, data: bytes = _PNG_BYTES) -> dict:
    return {
        "id": media_id,
        "mime_type": "image/png",
        "data": data,
        "source_url": f"https://example.test/page-{media_id}",
        "title": f"Page {media_id}",
        "created_at": created_at,
    }


class _CapturingClient:
    """The client boundary stood in for: every ``chat`` is recorded verbatim and
    answers with one plain description; a caller-specified count of leading calls
    raise, so an errored image is distinguishable from a described one."""

    def __init__(self, descriptions: list[str], fail_first: int = 0) -> None:
        self.descriptions = descriptions
        self.fail_first = fail_first
        self.sent: list[list[LlmMessage]] = []
        self.prompt_types: list[str | None] = []

    async def chat(
        self,
        messages: list[LlmMessage],
        *,
        call_link: CallLink,
        prompt_type: str | None = None,
    ) -> LlmResponse:
        self.sent.append(messages)
        self.prompt_types.append(prompt_type)
        if self.fail_first > len(self.sent) - 1:
            raise LlmError("boom")
        return LlmResponse(
            message=LlmMessage(role="assistant", content=self.descriptions[len(self.sent) - 1])
        )


class TestSelectingMedia:
    def test_ids_load_the_rows_named_and_refuse_a_missing_one(self, tmp_path) -> None:
        db_path = _write_media_db(
            tmp_path, [_captured(1, "2026-11-01 10:00:00"), _captured(3, "2026-11-02 10:00:00")]
        )
        rows = select_media(db_path, ids=[3, 1], recent=None)
        assert [row.id for row in rows] == [1, 3]  # id order, not the given order
        assert rows[0].mime_type == "image/png"
        assert rows[0].source_url == "https://example.test/page-1"
        with pytest.raises(SystemExit, match="no media row"):
            _load_by_ids(db_path, [1, 99])

    def test_recent_selects_captured_images_only_newest_first(self, tmp_path) -> None:
        """Generated images carry no ``source_url`` — excluded, whatever their date."""
        generated = {
            "id": 2,
            "mime_type": "image/png",
            "data": _PNG_BYTES,
            "source_url": None,
            "title": "a drawn cat",
            "created_at": "2026-11-04 09:00:00",  # the newest row in the DB
        }
        db_path = _write_media_db(
            tmp_path,
            [
                generated,
                _captured(1, "2026-11-01 10:00:00"),
                _captured(3, "2026-11-03 10:00:00"),
                _captured(4, "2026-11-02 10:00:00"),
            ],
        )
        rows = _load_recent(db_path, 2)
        assert [row.id for row in rows] == [3, 4]  # newest captured first, generated skipped

    def test_asking_for_both_selectors_is_refused(self, tmp_path) -> None:
        db_path = _write_media_db(tmp_path, [_captured(1, "2026-11-01 10:00:00")])
        with pytest.raises(SystemExit, match="IDS or RECENT"):
            select_media(db_path, ids=[1], recent=2)
        with pytest.raises(SystemExit, match="IDS.*or.*RECENT"):
            select_media(db_path, ids=None, recent=None)


class TestDescribingRows:
    async def test_the_fixed_instruction_rides_the_row_s_own_data_uri(self, tmp_path) -> None:
        """Every image is sent as ONE user turn: the neutral instruction verbatim,
        the image as a data URI with the row's own mime type — the exact shape
        ``LlmClient`` translates to ``image_url`` content parts."""
        db_path = _write_media_db(
            tmp_path,
            [
                _captured(1, "2026-11-01 10:00:00"),
                {
                    **_captured(2, "2026-11-02 10:00:00"),
                    "mime_type": "image/jpeg",
                    "data": _JPEG_BYTES,
                },
            ],
        )
        rows = _load_by_ids(db_path, [1, 2])
        client = _CapturingClient(["a png thing", "a jpeg thing"])
        jsonl_path = str(tmp_path / "describe.jsonl")
        tally = await describe_rows(client, rows, jsonl_path=jsonl_path, concurrency=2)

        assert len(client.sent) == 2
        for messages in client.sent:
            assert len(messages) == 1
            assert messages[0].role == "user"
            assert messages[0].content == DESCRIPTOR_INSTRUCTION
            assert client.prompt_types == ["describe-image", "describe-image"]
        expected_uris = [
            f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode()}",
            f"data:image/jpeg;base64,{base64.b64encode(_JPEG_BYTES).decode()}",
        ]
        assert client.sent[0][0].images == [expected_uris[0]]
        assert client.sent[1][0].images == [expected_uris[1]]
        assert (tally.described, tally.errored) == (2, 0)

        with open(jsonl_path) as handle:
            lines = [json.loads(line) for line in handle]
        assert {line["media_id"] for line in lines} == {1, 2}
        assert {line["description"] for line in lines} == {"a png thing", "a jpeg thing"}
        assert all(line["error"] is None for line in lines)
        assert lines[0]["title"] == "Page 1" and lines[0]["source_url"].startswith("https://")

    async def test_an_errored_call_is_recorded_and_the_batch_continues(self, tmp_path) -> None:
        db_path = _write_media_db(
            tmp_path, [_captured(1, "2026-11-01 10:00:00"), _captured(2, "2026-11-02 10:00:00")]
        )
        rows = _load_by_ids(db_path, [1, 2])
        client = _CapturingClient(["first ok", "second ok"], fail_first=1)
        jsonl_path = str(tmp_path / "describe.jsonl")
        tally = await describe_rows(client, rows, jsonl_path=jsonl_path)

        assert (tally.described, tally.errored) == (1, 1)
        with open(jsonl_path) as handle:
            lines = [json.loads(line) for line in handle]
        errored = next(line for line in lines if line["error"] is not None)
        described = next(line for line in lines if line["error"] is None)
        assert "LlmError: boom" in errored["error"]
        assert errored["description"] is None
        assert described["description"] == "second ok"
        assert {line["media_id"] for line in lines} == {1, 2}

    async def test_no_jsonl_path_prints_only(self, tmp_path) -> None:
        db_path = _write_media_db(tmp_path, [_captured(1, "2026-11-01 10:00:00")])
        rows = _load_by_ids(db_path, [1])
        tally = await describe_rows(_CapturingClient(["a thing"]), rows)
        assert (tally.described, tally.errored) == (1, 0)


class TestMediaImageDataUri:
    def test_the_data_uri_carries_the_row_s_mime_type(self) -> None:
        row = MediaImage(
            id=7,
            mime_type="image/webp",
            data=b"bytes",
            source_url=None,
            title=None,
            created_at=None,
        )
        assert row.data_uri() == f"data:image/webp;base64,{base64.b64encode(b'bytes').decode()}"
