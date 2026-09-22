"""What the user is asked, what they are told, and what the model is not.

These are the decisions in the chat flow. The Chainlit handler around them
is wiring; a browser tests wiring, and nothing tests a decision that only
exists inside a UI callback.
"""

import json
import tempfile
from pathlib import Path

import pytest

from gsa import chat
from gsa.client import AnalysisStatus
from gsa.job import Finished
from gsa.upload import Matrix


def matrix(samples: list[str], genes: int | None = 12) -> Matrix:
    return Matrix(
        path=Path(tempfile.gettempdir()) / "x.tsv",
        size_bytes=1_200_000,
        samples=samples,
        gene_count=genes,
    )


class TestParsingTheReply:
    def test_accepts_a_plain_comma_list(self) -> None:
        grouping = chat.parse_grouping("control, control, treated, treated", 4)
        assert grouping.labels == ["control", "control", "treated", "treated"]
        assert (grouping.group1, grouping.group2) == ("control", "treated")

    @pytest.mark.parametrize("reply", ["a;a;b;b", "a\ta\tb\tb", "a , a ,b,  b"])
    def test_is_forgiving_about_separators_and_spacing(self, reply: str) -> None:
        assert chat.parse_grouping(reply, 4).labels == ["a", "a", "b", "b"]

    def test_groups_case_insensitively_but_keeps_the_spelling(self) -> None:
        # The labels reach the service and come back in its output, so
        # silently lower-casing someone's "Treated" makes the result harder
        # to read against their own notes.
        grouping = chat.parse_grouping("Treated, treated, Control, control", 4)
        assert grouping.labels == ["Treated", "Treated", "Control", "Control"]
        assert {grouping.group1, grouping.group2} == {"Treated", "Control"}

    def test_refuses_a_count_that_does_not_match(self) -> None:
        # The dangerous one. Three labels for four columns still analyses
        # *something*, and returns a plausible answer to the wrong question.
        with pytest.raises(chat.ReplyUnusableError, match="one per sample"):
            chat.parse_grouping("a, b, c", 4)

    def test_refuses_one_group(self) -> None:
        with pytest.raises(chat.ReplyUnusableError, match="nothing to compare"):
            chat.parse_grouping("treated, treated", 2)

    def test_refuses_three_groups_and_names_them(self) -> None:
        with pytest.raises(chat.ReplyUnusableError, match="a, b, c"):
            chat.parse_grouping("a, b, c", 3)

    def test_refuses_an_empty_reply(self) -> None:
        with pytest.raises(chat.ReplyUnusableError, match="could not find"):
            chat.parse_grouping("   ,, ", 4)


class TestWhatTheUserIsTold:
    def test_the_sample_names_are_echoed_back(self) -> None:
        # They have to see what was read in order to label it, and it is
        # their own data being shown to them.
        text = chat.describe_matrix(matrix(["Ctrl_1", "Ctrl_2", "Tr_1"]))
        assert "Ctrl_1" in text
        assert "3 samples" in text

    def test_a_long_sample_list_is_truncated(self) -> None:
        text = chat.describe_matrix(matrix([f"S{i}" for i in range(50)]))
        assert "and 26 more" in text
        assert "50 samples" in text

    def test_an_uncounted_gene_total_is_said_in_words(self) -> None:
        # `gene_count` is None for a file too long to finish counting. It
        # must not render as "None genes" or "-1 genes".
        text = chat.describe_matrix(matrix(["A", "B"], genes=None))
        assert "unknown number" in text
        assert "None" not in text

    def test_progress_is_a_single_clamped_line(self) -> None:
        line = chat.describe_progress(
            AnalysisStatus("running", "Permutation 900 / 1000", 0.9)
        )
        assert "90%" in line
        assert "\n" not in line
        # The service has reported completion values outside 0..1.
        assert "100%" in chat.describe_progress(AnalysisStatus("running", "x", 4.2))
        assert "0%" in chat.describe_progress(AnalysisStatus("running", "x", -1.0))


def finished(tmp_path: Path) -> Finished:
    table = tmp_path / "t.tsv"
    table.write_text("Pathway\tName\n")
    return Finished(
        analysis_id="an-1",
        for_model={
            "no_result": False,
            "pathway_count": 2679,
            "significant_count": 412,
            "top_pathways": [
                {
                    "stId": "R-HSA-1",
                    "name": "Hemostasis",
                    "direction": "Up",
                    "fdr": 1e-5,
                    "genes": 6,
                }
            ],
        },
        links=[
            (
                "Gene Set Analysis Summary",
                "https://reactome.org/PathwayBrowser/#/ANALYSIS=TOKEN",
            )
        ],
        table_path=table,
    )


class TestWhatTheUserReads:
    def test_it_reports_exact_counts_and_the_top_pathways(self, tmp_path: Path) -> None:
        text = chat.describe_result(finished(tmp_path))
        assert "412" in text
        assert "2,679" in text
        assert "Hemostasis" in text

    def test_it_includes_the_pathway_browser_link(self, tmp_path: Path) -> None:
        # The person gets the link. This is the half that must be present.
        assert "PathwayBrowser" in chat.describe_result(finished(tmp_path))

    def test_the_link_is_still_absent_from_what_the_model_sees(
        self, tmp_path: Path
    ) -> None:
        # And this is the half that must not. The URL carries the analysis
        # token, and whoever holds it can fetch the unredacted result --
        # including the user's own gene identifiers.
        assert "PathwayBrowser" not in json.dumps(finished(tmp_path).for_model)
