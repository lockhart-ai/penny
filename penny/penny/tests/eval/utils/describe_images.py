"""Describe stored media images on the suite's model (#3065).

The measurement tool behind ``make describe-images IDS=<ids>|RECENT=<n>``: it reads
``media`` rows READ-ONLY from the prod DB, sends each image (base64, its own mime
type) to the resolved roster model through the SAME ``LlmClient.chat()`` path
``replay.py`` uses — images ride the user message as data URIs, which
``LlmClient`` translates to ``image_url`` content parts and which forces the chat
transport — with one fixed neutral instruction, and writes one JSONL record per
image. It NEVER writes the DB: the connection is opened ``mode=ro`` and the only
thing it writes is the report file.

Its first use is measuring whether matching an outgoing reply against image
DESCRIPTIONS (with or without a recency bias) picks better images than today's
page-title embedding (``MediaStore.select_image``'s jittered nearest); the tool
itself is the future production describer.

Run through the make target, which mirrors ``make replay``'s recipe piece for
piece — the roster resolution from ``EVAL_MODELS`` (``resolve_from_env``), the
credential read (``from_env`` over the primary checkout's ``.env``), the provider
lock, and the container run with the primary checkout's prod DB nested bind, read
only. Never run against a model ``EVAL_MODELS`` does not define: the roster is the
one resolution and a hand-set ``LLM_PROVIDER`` is checked against it, the same as
every other run path.

    LLM_MODEL=google/gemma-4-26b-a4b-it make describe-images RECENT=50
    LLM_MODEL=google/gemma-4-26b-a4b-it make describe-images IDS=12,34,56

``RECENT`` selects the N most recent CAPTURED images — rows that carry the page
they were captured from. A generated image's row has no ``source_url``, so it is
excluded: the measurement pools are captured-by-collector images, and generated
images already carry their own description (their prompt) as their title.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

# Import order workaround, the same one ``replay.py`` carries: this module is the
# FIRST thing to import either package when run standalone, and ``penny.database``
# and ``penny.agents`` import each other.
import penny.agents  # noqa: F401
from penny.llm.call_link import CallLink
from penny.llm.client import LlmClient
from penny.llm.models import LlmError, LlmMessage, LlmResponse, ProviderPin
from penny.tests.eval.utils.roster import resolve_from_env

_DEFAULT_DB = "/penny/data/penny/penny.db"
_DEFAULT_API_URL = "http://host.docker.internal:11434"

# The fixed neutral instruction (#3065) — never varied per image, so every
# description in a batch is comparable.
DESCRIPTOR_INSTRUCTION = (
    "Describe this image in two or three plain sentences: what it shows, "
    "any visible text, and its subject."
)

# ``RECENT`` selects captured images only: a row the browse tool stored carries the
# page it came from; a generated image's row has no ``source_url``.
_CAPTURED_ONLY_SQL = "source_url IS NOT NULL"

_PROMPT_TYPE = "describe-image"


class DescribeClient(Protocol):
    """What this tool needs of the LLM client — ``LlmClient`` satisfies it.

    One seam, so the tests stand in for the client boundary without a live
    endpoint, the same shape ``replay.py``'s ``_client_for`` gives its tests."""

    async def chat(
        self,
        messages: list[LlmMessage],
        *,
        call_link: CallLink,
        prompt_type: str | None = None,
    ) -> LlmResponse: ...


@dataclass
class MediaImage:
    """One ``media`` row as this tool reads it — everything except the embedding
    (never used here) and with the blob kept raw until it is encoded for the wire."""

    id: int
    mime_type: str
    data: bytes
    source_url: str | None
    title: str | None
    created_at: str | None

    def data_uri(self) -> str:
        """The image as a data URI with its OWN mime type — the same shape the
        channels hand the chat turn (#2635), which ``LlmClient`` translates to an
        ``image_url`` content part verbatim."""
        return f"data:{self.mime_type};base64,{base64.b64encode(self.data).decode()}"


class DescribeRecord(BaseModel):
    """One image's outcome, one JSONL line: the row's own metadata for joining
    back against the pool, the description, or the error that took its place."""

    media_id: int
    mime_type: str
    title: str | None = None
    source_url: str | None = None
    created_at: str | None = None
    description: str | None = None
    error: str | None = None
    elapsed_ms: int | None = None


class DescribeTally:
    """A run's outcomes: described (a description came back) vs errored (the call
    raised) — printed at the end, and never a reason to stop the batch."""

    def __init__(self) -> None:
        self.described = 0
        self.errored = 0

    def record(self, record: DescribeRecord) -> None:
        if record.error is not None:
            self.errored += 1
        else:
            self.described += 1

    def report(self, label: str, total: int) -> None:
        print(f"\n[{label}] {self.described}/{total} described · {self.errored} errored")


def _connect(db_path: str) -> sqlite3.Connection:
    """Read-only, always: this tool reads prod media and must never write there."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _load_by_ids(db_path: str, ids: Sequence[int]) -> list[MediaImage]:
    connection = _connect(db_path)
    try:
        placeholders = ",".join("?" for _ in ids)
        rows = connection.execute(
            f"SELECT id, mime_type, data, source_url, title, created_at "
            f"FROM media WHERE id IN ({placeholders}) ORDER BY id",
            tuple(ids),
        ).fetchall()
    finally:
        connection.close()
    found = {row[0]: row for row in rows}
    missing = [str(media_id) for media_id in ids if media_id not in found]
    if missing:
        raise SystemExit(f"no media row(s) with id(s) {', '.join(missing)} in {db_path}")
    return [_media_image(row) for row in rows]


def _load_recent(db_path: str, limit: int) -> list[MediaImage]:
    """The ``limit`` most recent captured images, newest first — generated images
    (no ``source_url``) excluded (see ``_CAPTURED_ONLY_SQL``)."""
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            f"SELECT id, mime_type, data, source_url, title, created_at "
            f"FROM media WHERE {_CAPTURED_ONLY_SQL} "
            f"ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        connection.close()
    return [_media_image(row) for row in rows]


def _media_image(row: tuple) -> MediaImage:
    media_id, mime_type, data, source_url, title, created_at = row
    return MediaImage(
        id=media_id,
        mime_type=mime_type,
        data=data if isinstance(data, bytes) else bytes(data or b""),
        source_url=source_url,
        title=title,
        created_at=created_at,
    )


def select_media(db_path: str, ids: Sequence[int] | None, recent: int | None) -> list[MediaImage]:
    """Which images this run describes — IDS= or RECENT=, exactly one."""
    if ids is not None and recent is not None:
        raise SystemExit("describe-images: give IDS or RECENT, not both.")
    if ids:
        return _load_by_ids(db_path, ids)
    if recent:
        return _load_recent(db_path, recent)
    raise SystemExit("describe-images: IDS=<comma-separated media ids> or RECENT=<n> is required.")


def _record_for(row: MediaImage) -> DescribeRecord:
    return DescribeRecord(
        media_id=row.id,
        mime_type=row.mime_type,
        title=row.title,
        source_url=row.source_url,
        created_at=row.created_at,
    )


async def _write_record(
    record: DescribeRecord, handle: Any | None, lock: asyncio.Lock | None
) -> None:
    """Persist one record the moment its image finishes, never batched — a kill
    mid-batch loses only the images still in flight."""
    if handle is None:
        return
    assert lock is not None
    async with lock:
        handle.write(record.model_dump_json() + "\n")
        handle.flush()


def _row_metadata(row: MediaImage) -> dict[str, Any]:
    return {
        "media_id": row.id,
        "mime_type": row.mime_type,
        "title": row.title,
        "source_url": row.source_url,
        "created_at": row.created_at,
    }


async def _describe_one(
    client: DescribeClient,
    row: MediaImage,
    semaphore: asyncio.Semaphore,
    handle: Any | None,
    lock: asyncio.Lock | None,
) -> DescribeRecord:
    """One image's describe call, gated by the shared semaphore, its record written
    the moment it finishes."""
    async with semaphore:
        started = time.monotonic()
        try:
            response = await client.chat(
                [
                    LlmMessage(
                        role="user",
                        content=DESCRIPTOR_INSTRUCTION,
                        images=[row.data_uri()],
                    )
                ],
                call_link=CallLink.new_top(),
                prompt_type=_PROMPT_TYPE,
            )
        except LlmError as error:
            print(f"  [media {row.id}] ERROR {type(error).__name__}: {error}")
            record = DescribeRecord(
                **_row_metadata(row),
                error=f"{type(error).__name__}: {error}",
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            await _write_record(record, handle, lock)
            return record
    description = response.message.content or ""
    elapsed_ms = round((time.monotonic() - started) * 1000)
    print(f"  [media {row.id}] {description[:160]}")
    record = DescribeRecord(
        **_row_metadata(row),
        description=description,
        elapsed_ms=elapsed_ms,
    )
    await _write_record(record, handle, lock)
    return record


async def describe_rows(
    client: DescribeClient,
    rows: Sequence[MediaImage],
    *,
    jsonl_path: str | None = None,
    concurrency: int = 1,
) -> DescribeTally:
    """Describe every row through ``client`` — ``concurrency`` calls in flight,
    each record landing in ``--jsonl`` as its image finishes. Every image is
    attempted: one error never stops the batch."""
    tally = DescribeTally()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    handle = open(jsonl_path, "w") if jsonl_path else None  # noqa: SIM115
    try:
        lock = asyncio.Lock() if handle is not None else None
        records = await asyncio.gather(
            *(_describe_one(client, row, semaphore, handle, lock) for row in rows)
        )
    finally:
        if handle is not None:
            handle.close()
    for record in records:
        tally.record(record)
    return tally


def _client_for(entry: Any) -> LlmClient:
    """The SAME client construction ``replay.py``'s plain mode uses: the resolved
    roster entry's model and its own pin, the shell's ``LLM_API_URL``/``LLM_API_KEY``.
    No metrics db — this tool reads no per-call timing back, and the promptlog rows
    a throwaway db would collect serve nothing here."""
    return LlmClient(
        api_url=os.environ.get("LLM_API_URL", _DEFAULT_API_URL),
        model=entry.model,
        api_key=os.environ.get("LLM_API_KEY") or "not-needed",
        provider_pin=ProviderPin.pin(entry.provider),
        max_retries=3,
        retry_delay=1.0,
        timeout=120.0,
    )


async def _main(args: argparse.Namespace) -> None:
    # The ONE resolution for the whole invocation, the same as every run path: the
    # model and its pin come from `EVAL_MODELS` through the roster — never from a
    # hand-set ``LLM_MODEL``/``LLM_PROVIDER``.
    entry = resolve_from_env()
    ids = _ids_from_arg(args.ids)
    rows = select_media(args.db, ids=ids, recent=args.recent)
    print(
        f"describing {len(rows)} image(s) with {entry.model}"
        f" at {os.environ.get('LLM_API_URL', _DEFAULT_API_URL)}"
    )
    tally = await describe_rows(
        _client_for(entry),
        rows,
        jsonl_path=args.jsonl,
        concurrency=args.concurrency,
    )
    tally.report("describe-images", len(rows))


def _ids_from_arg(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    try:
        return [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit(
            f"describe-images: IDS must be comma-separated integers: {error}"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default=_DEFAULT_DB, help="SQLite DB with the media table (read-only)"
    )
    parser.add_argument(
        "--ids",
        help="comma-separated media row ids to describe; give this or --recent, not both",
    )
    parser.add_argument(
        "--recent",
        type=int,
        help="describe the N most recent CAPTURED images (generated images excluded)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="how many describe calls to run in flight at once (default 1 — serial); "
        "size this to what the resolved provider sustains",
    )
    parser.add_argument(
        "--jsonl",
        help="write one JSON object per image (its metadata, the description, or the "
        "error) to this path, one line per image as each finishes",
    )
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
