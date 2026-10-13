"""Structural differences between retrieval baseline captures must fail comparison."""

import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("nltk", reason="retrieval stack not installed")

pytestmark = pytest.mark.requires_retrieval_stack

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "bin" / "retrieval_baseline"
_loader = importlib.machinery.SourceFileLoader("retrieval_baseline", str(_SCRIPT))
_spec = importlib.util.spec_from_loader("retrieval_baseline", _loader)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_loader.exec_module(_module)
compare = _module.compare

BASE_RESULTS: dict[str, Any] = {
    "What is TP53?": {"reactions": {"bm25": ["R-HSA-1", "R-HSA-2"]}}
}


def _write_capture(path: Path, results: dict[str, Any]) -> Path:
    path.write_text(json.dumps({"results": results}))
    return path


@pytest.mark.parametrize("results", [BASE_RESULTS, {}])
def test_identical_captures_succeed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], results: dict[str, Any]
) -> None:
    before = _write_capture(tmp_path / "before.json", results)
    after = _write_capture(tmp_path / "after.json", results)
    compare(argparse.Namespace(before=before, after=after))
    total = 1 if results else 0
    assert f"0/{total} retriever results changed." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("after_results", "expected_errors"),
    [
        (
            {},
            ("! question missing from the newer capture: What is TP53?",),
        ),
        (
            {"What is TP53?": {}},
            (
                "! collection missing from the newer capture: [reactions]  What is TP53?",
            ),
        ),
        (
            {"What is TP53?": {"reactions": {}}},
            (
                "! retriever missing from the newer capture: [reactions/bm25]  What is TP53?",
            ),
        ),
        (
            {**BASE_RESULTS, "Extra?": {"reactions": {"bm25": ["R-HSA-1"]}}},
            ("! unexpected question in the newer capture: Extra?",),
        ),
        (
            {
                "What is TP53?": {
                    **BASE_RESULTS["What is TP53?"],
                    "extra": {"bm25": ["R-HSA-1"]},
                }
            },
            ("! unexpected collection in the newer capture: [extra]  What is TP53?",),
        ),
        (
            {
                "What is TP53?": {
                    "reactions": {
                        **BASE_RESULTS["What is TP53?"]["reactions"],
                        "extra": ["R-HSA-1"],
                    }
                }
            },
            (
                "! unexpected retriever in the newer capture: [reactions/extra]  What is TP53?",
            ),
        ),
        (
            {
                "What is TP53?": {
                    "reactions": {
                        **BASE_RESULTS["What is TP53?"]["reactions"],
                        "extra_ret": ["R-HSA-1"],
                    },
                    "extra_coll": {"bm25": ["R-HSA-1"]},
                },
                "Extra?": {"reactions": {"bm25": ["R-HSA-1"]}},
            },
            (
                "! unexpected question in the newer capture: Extra?",
                "! unexpected collection in the newer capture: [extra_coll]  What is TP53?",
                "! unexpected retriever in the newer capture: [reactions/extra_ret]  What is TP53?",
            ),
        ),
    ],
)
def test_structural_mismatch_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    after_results: dict[str, Any],
    expected_errors: tuple[str, ...],
) -> None:
    before = _write_capture(tmp_path / "before.json", BASE_RESULTS)
    after = _write_capture(tmp_path / "after.json", after_results)
    with pytest.raises(SystemExit) as exc_info:
        compare(argparse.Namespace(before=before, after=after))
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    for err in expected_errors:
        assert err in out


def test_reordered_documents_fail_with_reordered_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _write_capture(tmp_path / "before.json", BASE_RESULTS)
    after = _write_capture(
        tmp_path / "after.json",
        {"What is TP53?": {"reactions": {"bm25": ["R-HSA-2", "R-HSA-1"]}}},
    )
    with pytest.raises(SystemExit) as exc_info:
        compare(argparse.Namespace(before=before, after=after))
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "REORDERED  [reactions/bm25]  What is TP53?" in out
    assert "1/1 retriever results changed." in out


def test_different_documents_fail_with_dropped_and_added(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _write_capture(tmp_path / "before.json", BASE_RESULTS)
    after = _write_capture(
        tmp_path / "after.json",
        {"What is TP53?": {"reactions": {"bm25": ["R-HSA-1", "R-HSA-3"]}}},
    )
    with pytest.raises(SystemExit) as exc_info:
        compare(argparse.Namespace(before=before, after=after))
    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "DIFFERENT DOCS  [reactions/bm25]  What is TP53?" in out
    assert "dropped: ['R-HSA-2']" in out
    assert "added:   ['R-HSA-3']" in out
    assert "1/1 retriever results changed." in out


def test_inputs_are_not_mutated(tmp_path: Path) -> None:
    before = _write_capture(tmp_path / "before.json", BASE_RESULTS)
    after = _write_capture(
        tmp_path / "after.json",
        {"What is TP53?": {"reactions": {"bm25": ["R-HSA-1", "R-HSA-3"]}}},
    )
    before_raw = before.read_text()
    after_raw = after.read_text()
    with pytest.raises(SystemExit):
        compare(argparse.Namespace(before=before, after=after))
    assert before.read_text() == before_raw
    assert after.read_text() == after_raw
