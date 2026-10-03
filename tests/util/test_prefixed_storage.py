"""The S3 storage client prefixes a key once, not on every round trip."""

import asyncio
from pathlib import PurePosixPath
from typing import Any

import pytest
from chainlit.data.storage_clients.s3 import S3StorageClient

from util.chainlit_helpers import PrefixedS3StorageClient


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []

    async def upload(_self: Any, key: str, *_a: Any, **_k: Any) -> dict[str, Any]:
        calls.append(("upload", key))
        return {"object_key": key, "url": f"https://bucket/{key}"}

    async def delete(_self: Any, key: str) -> bool:
        calls.append(("delete", key))
        return True

    async def read(_self: Any, key: str) -> str:
        calls.append(("read", key))
        return f"https://bucket/{key}"

    monkeypatch.setattr(S3StorageClient, "upload_file", upload)
    monkeypatch.setattr(S3StorageClient, "delete_file", delete)
    monkeypatch.setattr(S3StorageClient, "get_read_url", read)
    return calls


def _client() -> PrefixedS3StorageClient:
    client = object.__new__(PrefixedS3StorageClient)  # no boto3, no bucket
    client._prefix = PurePosixPath("P")
    return client


def test_the_stored_key_comes_back_to_the_same_object(
    seen: list[tuple[str, str]],
) -> None:
    # upload returns the key it stored, prefix included; the data layer saves
    # it and later reads and deletes with it. Prefixing again gave 'P/P/...'
    # (review, area 2).
    client = _client()
    stored = asyncio.run(client.upload_file("user/el/counts.tsv", b"x"))["object_key"]
    asyncio.run(client.get_read_url(stored))
    asyncio.run(client.delete_file(stored))
    assert seen == [
        ("upload", "P/user/el/counts.tsv"),
        ("read", "P/user/el/counts.tsv"),
        ("delete", "P/user/el/counts.tsv"),
    ]


def test_an_unprefixed_key_is_still_prefixed(seen: list[tuple[str, str]]) -> None:
    asyncio.run(_client().delete_file("user/el/counts.tsv"))
    assert seen == [("delete", "P/user/el/counts.tsv")]
