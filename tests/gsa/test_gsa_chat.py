"""What the user is asked, what they are told, and what the model is not.

These are the decisions in the chat flow. The Chainlit handler around them
is wiring; a browser tests wiring, and nothing tests a decision that only
exists inside a UI callback.
"""

import json
import re
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

    def test_progress_is_one_line_carrying_the_services_own_account(self) -> None:
        line = chat.describe_progress(
            AnalysisStatus("running", "Permutation 860 / 1000", 0.6)
        )
        assert "Permutation 860 / 1000" in line
        assert "\n" not in line

    def test_progress_does_not_show_a_percentage_the_service_does_not_update(
        self,
    ) -> None:
        # Measured against the real service: `completed` stays at 0.6 for the
        # whole permutation phase. Showing it produced
        # "60% · Permutation 1000 / 1000".
        line = chat.describe_progress(
            AnalysisStatus("running", "Permutation 1000 / 1000", 0.6)
        )
        assert "%" not in line

    def test_a_blank_description_still_says_something(self) -> None:
        assert "working" in chat.describe_progress(AnalysisStatus("running", "  ", 0.1))


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


class TestPathwayNamesSurviveMarkdown:
    """Real Reactome names contain markdown syntax.

    Measured over a real 2,679-pathway result: `NOTCH1:M1580_K2555` and
    `H139Hfs13* PPM1K ...`. Unescaped, paired `_` or `*` become emphasis and
    a variant identifier renders with characters missing -- silently, in a
    results table someone may copy into a paper.
    """

    def result_with(self, tmp_path: Path, name: str) -> Finished:
        table = tmp_path / "t.tsv"
        table.write_text("Pathway\\tName\\n")
        return Finished(
            analysis_id="an-1",
            for_model={
                "no_result": False,
                "pathway_count": 1,
                "significant_count": 1,
                "top_pathways": [
                    {
                        "stId": "R-HSA-1",
                        "name": name,
                        "direction": "Up",
                        "fdr": 1e-5,
                        "genes": 6,
                    }
                ],
            },
            links=[],
            table_path=table,
        )

    @pytest.mark.parametrize(
        "name",
        [
            "Signaling by NOTCH1 t(7;9)(NOTCH1:M1580_K2555) Translocation Mutant",
            "H139Hfs13* PPM1K causes a mild variant of  MSUD",
            "a_b_c and x*y*z",
            "left | right",
        ],
    )
    def test_special_characters_are_escaped(self, tmp_path: Path, name: str) -> None:
        text = chat.describe_result(self.result_with(tmp_path, name))
        row = next(
            line for line in text.splitlines() if line.startswith("| ") and "Up" in line
        )
        for ch in "_*|":
            if ch in name:
                assert "\\" + ch in row, f"{ch!r} not escaped in {row!r}"

    def test_a_pipe_does_not_add_a_column(self, tmp_path: Path) -> None:
        # An unescaped `|` splits the cell and shifts Direction and FDR one
        # column right, so the table reports the wrong value in each.
        text = chat.describe_result(self.result_with(tmp_path, "left | right"))
        row = next(line for line in text.splitlines() if "Up" in line)
        assert row.replace("\\|", "").count("|") == 4


class TestRecognisingARequestToRunGsa:
    """Asked in words, the chat said it could not run a GSA. These pin which
    messages get the how-to instead of going to the model.

    Reported with "can we run gsa in this chat please", answered "no"."""

    @pytest.mark.parametrize(
        "text",
        [
            "can we run gsa in this chat please",
            "Can I run a gene set analysis in this chat?",
            "Run a GSEA on my RNA-seq data",
            "I have an expression matrix, how do I analyse it here?",
            "is it possible to do a GSEA here",
            "could you perform a gene set enrichment analysis for me",
            "I'd like to run ReactomeGSA on my data",
            "help me analyse my microarray data",
            "how can I upload my count matrix",
            "I want to do gene-set analysis on my proteomics data",
            "please run GSA",
            "can you analyse my RNA-seq expression data",
            "Kannst du eine GSEA machen? can you run gsea",
            "start a gene set analysis",
            "how do I submit expression data for analysis",
        ],
    )
    def test_a_request_to_run_one_is_recognised(self, text: str) -> None:
        assert chat.asks_to_run_gsa(text), text

    @pytest.mark.parametrize(
        "text",
        [
            # Questions *about* the method: explained by the model, not
            # answered with upload instructions.
            "what is GSEA?",
            "what's the difference between GSA and over-representation analysis",
            "explain gene set enrichment analysis",
            "what does RNA-seq measure",
            # Ordinary questions.
            "which pathways involve TP53?",
            "what does CDK5 phosphorylate?",
            "how many species are in Reactome",
            "analyse these genes: TP53, MDM2, CDKN1A",
            "can you tell me about apoptosis",
            "how do I cite Reactome",
            "run through the steps of glycolysis",
            "",
        ],
    )
    def test_anything_else_goes_to_the_model(self, text: str) -> None:
        assert not chat.asks_to_run_gsa(text), text

    def test_the_reply_says_yes_and_how(self) -> None:
        # The failure was an answer starting "no".
        assert chat.HOW_TO_RUN_GSA.startswith("Yes")
        assert "Attach" in chat.HOW_TO_RUN_GSA
        assert "20 MB" in chat.HOW_TO_RUN_GSA


def test_the_gene_list_example_it_gives_is_one_the_chat_runs() -> None:
    # The reply promises the chat analyses a pasted gene list. The example it
    # gives must be one the recogniser accepts, or the promise is false.
    from analysis.gene_list import gene_list_request

    example = re.search(r"\*(run a pathway analysis on [^*]+)\*", chat.HOW_TO_RUN_GSA)
    assert example is not None
    assert gene_list_request(example.group(1)) == ["TP53", "ERBB2", "RUNX2"]


def test_the_matrix_only_reply_offers_no_gene_list() -> None:
    # Sent after the reader declined a gene-list analysis; it must not offer
    # that analysis again.
    assert chat.HOW_TO_RUN_GSA.startswith(chat.HOW_TO_RUN_GSA_WITH_A_MATRIX)
    assert "Attach" in chat.HOW_TO_RUN_GSA_WITH_A_MATRIX
    assert "list of genes" not in chat.HOW_TO_RUN_GSA_WITH_A_MATRIX
    assert "list of genes" in chat.HOW_TO_RUN_GSA
