"""The flow, driven without a browser.

`run_analysis` takes its four chat operations as arguments, so the paths
that matter -- a refused file, an unusable reply, a submission that fails,
an analysis that fails after being accepted -- can be exercised here. What
is left for a browser is whether Chainlit's own callbacks are wired to the
right arguments.
"""

import asyncio
import functools
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gsa import chainlit_flow
from gsa.client import AnalysisStatus
from gsa.job import AnalysisFailedError

FIXTURE = Path(__file__).parent / "result_fixture.json"


def asyncio_test(fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class Chat:
    """Records what a user would have seen."""

    def __init__(self, reply: str | None = "a, a, b, b") -> None:
        self.reply = reply
        self.said: list[str] = []
        self.progress: list[str] = []
        self.files: list[Path] = []

    async def ask(self) -> str | None:
        return self.reply

    async def send(self, text: str) -> None:
        self.said.append(text)

    async def update(self, text: str) -> None:
        self.progress.append(text)

    async def send_file(self, path: Path) -> None:
        self.files.append(path)

    @property
    def transcript(self) -> str:
        return "\n".join(self.said)


class StubClient:
    def __init__(self, *, fail_submit: bool = False, fail_run: bool = False) -> None:
        self.fail_submit = fail_submit
        self.fail_run = fail_run
        self.submitted: dict[str, Any] = {}

    async def submit(self, **kwargs: Any) -> str:
        if self.fail_submit:
            raise AnalysisFailedError("the service refused it")
        self.submitted = kwargs
        return "an-1"

    async def analysis_status(self, analysis_id: str) -> AnalysisStatus:
        if self.fail_run:
            return AnalysisStatus("failed", "CONNECTION_FORCED - broker closed", 1.0)
        return AnalysisStatus("complete", "Analysis done", 1.0)

    async def result(self, analysis_id: str) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(FIXTURE.read_text())
        return loaded


class Attached:
    def __init__(self, path: Path) -> None:
        self.name = path.name
        self.path = str(path)


def a_matrix(tmp_path: Path, name: str = "counts.tsv") -> Attached:
    path = tmp_path / name
    path.write_text("\tS1\tS2\tS3\tS4\nENSG1\t1\t2\t3\t4\nENSG2\t5\t6\t7\t8\n")
    return Attached(path)


async def run(attachment: Attached, chat_: Chat, client: Any, out: Path) -> None:
    os.environ[chainlit_flow.RESULTS_DIR_ENV] = str(out)
    await chainlit_flow.run_analysis(
        attachment,
        ask_for_grouping=chat_.ask,
        send=chat_.send,
        update_progress=chat_.update,
        send_file=chat_.send_file,
        client=client,
    )


class TestRouting:
    def test_an_ordinary_message_is_not_an_analysis(self) -> None:
        # Every normal message goes through here, so it must say no to the
        # common case without knowing anything about Chainlit.
        assert chainlit_flow.matrix_attachment(None) is None
        assert chainlit_flow.matrix_attachment([]) is None

    def test_a_pdf_is_not_a_matrix(self, tmp_path: Path) -> None:
        pdf = tmp_path / "paper.pdf"
        pdf.write_text("x")
        assert chainlit_flow.matrix_attachment([Attached(pdf)]) is None

    def test_a_tsv_is(self, tmp_path: Path) -> None:
        assert chainlit_flow.matrix_attachment([a_matrix(tmp_path)]) is not None


class TestTheUnhappyPaths:
    @asyncio_test
    async def test_a_rejected_file_is_explained_and_deleted(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "notes.txt"
        bad.write_text("just some prose\n")
        chat_ = Chat()

        await run(Attached(bad), chat_, StubClient(), tmp_path / "out")

        assert "expression matrix" in chat_.transcript
        assert not bad.exists()
        assert not chat_.files

    @asyncio_test
    async def test_an_unusable_reply_deletes_the_file_too(self, tmp_path: Path) -> None:
        # The count mismatch is the likeliest mistake, and the file must
        # not survive it -- this host has 5 GB free.
        attachment = a_matrix(tmp_path)
        chat_ = Chat(reply="a, b")

        await run(attachment, chat_, StubClient(), tmp_path / "out")

        assert "one per sample" in chat_.transcript
        assert not Path(attachment.path).exists()

    @asyncio_test
    async def test_no_reply_at_all_deletes_the_file(self, tmp_path: Path) -> None:
        attachment = a_matrix(tmp_path)
        chat_ = Chat(reply=None)

        await run(attachment, chat_, StubClient(), tmp_path / "out")

        assert "not run anything" in chat_.transcript
        assert not Path(attachment.path).exists()

    @asyncio_test
    async def test_a_failed_submission_is_reported(self, tmp_path: Path) -> None:
        chat_ = Chat()
        await run(
            a_matrix(tmp_path), chat_, StubClient(fail_submit=True), tmp_path / "out"
        )

        assert "could not start" in chat_.transcript
        assert not chat_.files

    @asyncio_test
    async def test_an_analysis_that_fails_after_starting_is_reported(
        self, tmp_path: Path
    ) -> None:
        # The measured failure: accepted with a 200, then dead. A flow that
        # only checked the submission would sit waiting forever, or claim
        # success.
        chat_ = Chat()
        await run(
            a_matrix(tmp_path), chat_, StubClient(fail_run=True), tmp_path / "out"
        )

        assert "did not finish" in chat_.transcript
        assert "CONNECTION_FORCED" in chat_.transcript
        assert not chat_.files


class TestTheHappyPath:
    @asyncio_test
    async def test_it_delivers_a_summary_and_a_file(self, tmp_path: Path) -> None:
        chat_ = Chat()
        client = StubClient()

        await run(a_matrix(tmp_path), chat_, client, tmp_path / "out")

        assert "significant" in chat_.transcript
        assert chat_.files, "the user must get their table"
        assert chat_.files[0].exists()
        assert "MeanWeightT0" in chat_.files[0].read_text()

    @asyncio_test
    async def test_the_filename_never_reaches_the_service(self, tmp_path: Path) -> None:
        chat_ = Chat()
        client = StubClient()

        await run(
            a_matrix(tmp_path, "smith_lab_unpublished_2026.tsv"),
            chat_,
            client,
            tmp_path / "out",
        )

        assert "smith_lab" not in json.dumps(client.submitted.get("dataset_name"))

    @asyncio_test
    async def test_the_uploaded_matrix_is_gone_afterwards(self, tmp_path: Path) -> None:
        attachment = a_matrix(tmp_path)
        await run(attachment, Chat(), StubClient(), tmp_path / "out")
        assert not Path(attachment.path).exists()


class TestTheUploadNeverSurvives:
    """The file is gone when this function returns, whatever happened.

    Each branch used to delete it for itself, which covered the cases I had
    thought of and not the ones I had not: a file Chainlit has not finished
    writing, or an `AskUserMessage` that times out. An unhandled error is
    precisely when nobody is around to tidy up, and this host has 5 GB free.
    """

    @asyncio_test
    async def test_a_file_that_is_not_there_is_reported_not_raised(
        self, tmp_path: Path
    ) -> None:
        missing = Attached(tmp_path / "never-written.tsv")
        chat_ = Chat()

        await run(missing, chat_, StubClient(), tmp_path / "out")

        assert "could not read that file" in chat_.transcript
        # And it does not describe it as the user's mistake, because it is
        # not one they can act on.
        assert "expression matrix" not in chat_.transcript

    @asyncio_test
    async def test_an_exception_from_the_question_still_deletes_it(
        self, tmp_path: Path
    ) -> None:
        attachment = a_matrix(tmp_path)

        class Exploding(Chat):
            async def ask(self) -> str | None:
                raise TimeoutError("the user never answered")

        with pytest.raises(TimeoutError):
            await run(attachment, Exploding(), StubClient(), tmp_path / "out")

        assert not Path(attachment.path).exists()

    @asyncio_test
    async def test_a_binary_file_is_refused_rather_than_crashing(
        self, tmp_path: Path
    ) -> None:
        # Somebody will attach a spreadsheet or an image with a .tsv name.
        path = tmp_path / "image.tsv"
        path.write_bytes(bytes(range(256)) * 50)
        chat_ = Chat()

        await run(Attached(path), chat_, StubClient(), tmp_path / "out")

        assert chat_.said, "it must say something rather than fail silently"
        assert not path.exists()


class TestTheResultFile:
    def test_carries_an_explicit_mime_type(self, tmp_path: Path) -> None:
        """Without it, the whole chat UI dies at the moment of success.

        Chainlit infers a path-based element's type from magic bytes, and a
        TSV has none, so `mime` was null; the browser then called
        `mime.startsWith(...)` on it and replaced the chat with a
        JavaScript error. Found by a headless browser against the deployed
        image, after every server-side check had passed.
        """
        kwargs = chainlit_flow.result_file_kwargs(tmp_path / "reactome-gsa-an-1.tsv")
        assert kwargs["mime"].startswith("text/")

    def test_names_the_file_after_the_table(self, tmp_path: Path) -> None:
        path = tmp_path / "reactome-gsa-an-1.tsv"
        kwargs = chainlit_flow.result_file_kwargs(path)
        assert kwargs["name"] == "reactome-gsa-an-1.tsv"
        assert kwargs["path"] == str(path)
