"""A matrix waiting for labels, across messages rather than a blocking ask."""

from pathlib import Path

import pytest

from gsa import chat
from gsa.pending import PendingMatrices
from gsa.upload import Matrix


def matrix(tmp_path: Path, name: str = "m.tsv") -> Matrix:
    path = tmp_path / name
    path.write_text("\tA\tB\nTP53\t1\t2\n")
    return Matrix(path=path, size_bytes=10, samples=["A", "B"], gene_count=1)


def test_a_matrix_waits_and_is_taken_once(tmp_path: Path) -> None:
    store = PendingMatrices()
    m = matrix(tmp_path)
    store.put("s", m)
    assert store.peek("s") is m
    assert store.take("s") is m
    assert store.take("s") is None


def test_putting_the_same_matrix_back_keeps_its_file(tmp_path: Path) -> None:
    # After a reply that was not usable labels, the matrix goes back; that
    # must not delete the file it is waiting with.
    store = PendingMatrices()
    m = matrix(tmp_path)
    store.put("s", m)
    store.put("s", m)
    assert m.path.exists()


def test_a_new_upload_replaces_and_deletes_the_old(tmp_path: Path) -> None:
    store = PendingMatrices()
    old, new = matrix(tmp_path, "old.tsv"), matrix(tmp_path, "new.tsv")
    store.put("s", old)
    store.put("s", new)
    assert not old.path.exists()
    assert store.peek("s") is new


def test_an_expired_matrix_is_gone_with_its_file(tmp_path: Path) -> None:
    store = PendingMatrices(wait_seconds=60)
    m = matrix(tmp_path)
    store.put("s", m, now=0)
    assert store.peek("s", now=61) is None
    assert not m.path.exists()


def test_dropping_a_session_deletes_its_file(tmp_path: Path) -> None:
    store = PendingMatrices()
    m = matrix(tmp_path)
    store.put("s", m)
    store.drop("s")
    assert not m.path.exists()


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("control, control, treated, treated", True),  # usable
        ("control, treated", True),  # wrong count: a mistake to explain
        ("control; control; treated", True),
        ("What does this matrix show?", False),  # a question for the model
        ("tell me about apoptosis", False),
    ],
)
def test_what_counts_as_an_attempt_at_labels(reply: str, expected: bool) -> None:
    assert chat.looks_like_labels(reply, 4) is expected
