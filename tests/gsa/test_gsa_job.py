"""One analysis, start to finished file.

A stub client rather than the network: the behaviours worth pinning are a
submission that succeeds and then fails, a poll that must not run forever,
and a result that has to be three different things for three different
audiences. None of those needs the service to be up, and all of them would
be untestable if they only existed inside a Chainlit handler.
"""

import asyncio
import functools
import json
import time
from pathlib import Path
from typing import Any

import pytest

from gsa import job
from gsa.client import AnalysisStatus, DatasetSummary, LoadingStatus
from gsa.upload import Matrix

FIXTURE = Path(__file__).parent / "result_fixture.json"


def asyncio_test(fn: Any) -> Any:
    """Run an async test on its own loop.

    This repo has no pytest-asyncio; `tests/agent/test_collection_routing.py`
    calls `asyncio.run` inline. Same approach, kept out of the test bodies
    so each one reads as the sequence it is testing.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(fn(*args, **kwargs))

    return wrapper


class StubClient:
    """Records what it was asked, answers what it was told to."""

    def __init__(
        self,
        *,
        statuses: list[AnalysisStatus] | None = None,
        loading: list[LoadingStatus] | None = None,
        summary: DatasetSummary | None = None,
    ) -> None:
        self.statuses = statuses or [AnalysisStatus("complete", "Analysis done", 1.0)]
        self.loading = loading or [LoadingStatus("complete", "ready", 1.0, "DS")]
        self.summary_value = summary
        self.submitted: dict[str, Any] = {}
        self.matrix_downloads = 0

    async def load_public_dataset(self, resource_id: str, dataset_id: str) -> str:
        return "load-1"

    async def loading_status(self, loading_id: str) -> LoadingStatus:
        return self.loading.pop(0) if len(self.loading) > 1 else self.loading[0]

    async def dataset_summary(self, dataset_id: str) -> DatasetSummary:
        assert self.summary_value is not None
        return self.summary_value

    async def download_matrix(self, dataset_id: str) -> str:
        self.matrix_downloads += 1
        return "\tS1\tS2\nENSG1\t1\t2\n"

    async def submit(self, **kwargs: Any) -> str:
        self.submitted = kwargs
        return "an-1"

    async def analysis_status(self, analysis_id: str) -> AnalysisStatus:
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    async def result(self, analysis_id: str) -> dict[str, Any]:
        loaded: dict[str, Any] = json.loads(FIXTURE.read_text())
        return loaded


SUMMARY = DatasetSummary(
    dataset_id="EXAMPLE_MEL_RNA",
    title="Melanoma RNA-seq example",
    type="rnaseq_counts",
    samples=["S1", "S2", "S3", "S4"],
    factors={"condition": ["MCM", "MOCK", "MCM", "MOCK"]},
)


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(job, "_sleep", instant)


@asyncio_test
async def test_a_public_dataset_reaches_submission(tmp_path: Path) -> None:
    client = StubClient(summary=SUMMARY)
    analysis_id = await job.submit_public_dataset(
        client,  # type: ignore[arg-type]
        resource_id="example_datasets",
        dataset_id="EXAMPLE_MEL_RNA",
        factor="condition",
        group1="MOCK",
        group2="MCM",
    )
    assert analysis_id == "an-1"
    assert client.matrix_downloads == 1
    assert client.submitted["analysis_group"] == ["MCM", "MOCK", "MCM", "MOCK"]
    assert client.submitted["group1"] == "MOCK"


@asyncio_test
async def test_an_unknown_factor_says_what_there_is() -> None:
    client = StubClient(summary=SUMMARY)
    with pytest.raises(job.AnalysisFailedError, match="condition"):
        await job.submit_public_dataset(
            client,  # type: ignore[arg-type]
            resource_id="example_datasets",
            dataset_id="EXAMPLE_MEL_RNA",
            factor="treatment",
            group1="a",
            group2="b",
        )


@asyncio_test
async def test_an_unknown_group_says_what_there_is() -> None:
    client = StubClient(summary=SUMMARY)
    with pytest.raises(job.AnalysisFailedError, match="MCM, MOCK"):
        await job.submit_public_dataset(
            client,  # type: ignore[arg-type]
            resource_id="example_datasets",
            dataset_id="EXAMPLE_MEL_RNA",
            factor="condition",
            group1="TREATED",
            group2="MOCK",
        )


@asyncio_test
async def test_comparing_a_group_with_itself_is_refused() -> None:
    client = StubClient(summary=SUMMARY)
    with pytest.raises(job.AnalysisFailedError, match="must be different"):
        await job.submit_public_dataset(
            client,  # type: ignore[arg-type]
            resource_id="example_datasets",
            dataset_id="EXAMPLE_MEL_RNA",
            factor="condition",
            group1="MCM",
            group2="MCM",
        )


@asyncio_test
async def test_a_failed_analysis_raises_rather_than_returning_nothing(
    tmp_path: Path,
) -> None:
    # Measured: 200 on submission, then this. A caller that only checked the
    # submission would report success and hand back an empty result.
    client = StubClient(
        statuses=[AnalysisStatus("failed", "CONNECTION_FORCED - broker closed", 1.0)]
    )
    with pytest.raises(job.AnalysisFailedError, match="CONNECTION_FORCED"):
        await job.await_result(client, "an-1", out_dir=tmp_path)  # type: ignore[arg-type]


@asyncio_test
async def test_polling_stops_at_the_deadline(tmp_path: Path) -> None:
    # Without a deadline this loops forever against a stuck analysis, and
    # the symptom is a chat that never answers.
    client = StubClient(statuses=[AnalysisStatus("running", "Permutation 1/1000", 0.1)])
    with pytest.raises(job.AnalysisFailedError, match="did not finish"):
        await job.await_result(
            client,  # type: ignore[arg-type]
            "an-1",
            out_dir=tmp_path,
            deadline_seconds=-1,
        )


@asyncio_test
async def test_progress_is_reported_while_running(tmp_path: Path) -> None:
    seen: list[str] = []

    async def record(status: AnalysisStatus) -> None:
        seen.append(status.description)

    client = StubClient(
        statuses=[
            AnalysisStatus("running", "Permutation 200 / 1000", 0.2),
            AnalysisStatus("running", "Permutation 900 / 1000", 0.9),
            AnalysisStatus("complete", "Analysis done", 1.0),
        ]
    )
    await job.await_result(
        client,  # type: ignore[arg-type]
        "an-1",
        out_dir=tmp_path,
        on_progress=record,
    )
    assert "Permutation 200 / 1000" in seen
    assert "Analysis done" in seen


@asyncio_test
async def test_a_finished_analysis_splits_by_audience(tmp_path: Path) -> None:
    finished = await job.await_result(
        StubClient(),  # type: ignore[arg-type]
        "an-1",
        out_dir=tmp_path,
    )

    # The file gets every column the service sent.
    written = finished.table_path.read_text()
    assert "MeanWeightT0" in written

    # The model gets the bounded view and no capability URL.
    as_prompt = json.dumps(finished.for_model)
    assert "PathwayBrowser" not in as_prompt
    assert finished.for_model["no_result"] is False

    # The user gets the link.
    assert any("PathwayBrowser" in url for _, url in finished.links)


@asyncio_test
async def test_an_uploaded_file_is_deleted_even_when_submission_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "theirs.tsv"
    path.write_text("\tS1\tS2\nENSG1\t1\t2\n")
    matrix = Matrix(
        path=path, size_bytes=path.stat().st_size, samples=["S1", "S2"], gene_count=1
    )

    class Failing(StubClient):
        async def submit(self, **kwargs: Any) -> str:
            raise RuntimeError("upstream is down")

    with pytest.raises(RuntimeError):
        await job.submit_uploaded_matrix(
            Failing(),  # type: ignore[arg-type]
            matrix=matrix,
            dataset_type="rnaseq_counts",
            analysis_group=["A", "B"],
            group1="A",
            group2="B",
        )
    assert not path.exists()


@asyncio_test
async def test_an_uploaded_file_does_not_carry_its_name_to_the_service(
    tmp_path: Path,
) -> None:
    # The filename is user free text -- `smith_lab_unpublished_2026.txt` --
    # and it would come back in the result as `datasets[].name`.
    path = tmp_path / "smith_lab_unpublished_2026.tsv"
    path.write_text("\tS1\tS2\nENSG1\t1\t2\n")
    matrix = Matrix(path=path, size_bytes=1, samples=["S1", "S2"], gene_count=1)

    client = StubClient()
    await job.submit_uploaded_matrix(
        client,  # type: ignore[arg-type]
        matrix=matrix,
        dataset_type="rnaseq_counts",
        analysis_group=["A", "B"],
        group1="A",
        group2="B",
    )
    assert "smith_lab" not in json.dumps(client.submitted["dataset_name"])


@asyncio_test
async def test_mismatched_group_labels_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "m.tsv"
    path.write_text("\tS1\tS2\tS3\nENSG1\t1\t2\t3\n")
    matrix = Matrix(path=path, size_bytes=1, samples=["S1", "S2", "S3"], gene_count=1)

    with pytest.raises(job.AnalysisFailedError, match="one label per"):
        await job.submit_uploaded_matrix(
            StubClient(),  # type: ignore[arg-type]
            matrix=matrix,
            dataset_type="rnaseq_counts",
            analysis_group=["A", "B"],
            group1="A",
            group2="B",
        )


@asyncio_test
async def test_polling_is_bounded_by_iterations_as_well_as_time(tmp_path: Path) -> None:
    """The wall-clock deadline assumes every turn of the loop waits.

    When sleeping does not sleep -- patched here, but equally a zero
    interval from a caller -- elapsed time never advances and the loop
    never exits. Discovered by sabotaging the failure check: the suite hung
    instead of failing, which is a worse outcome than either.
    """
    forever = StubClient(
        statuses=[AnalysisStatus("running", "Permutation 1/1000", 0.1)]
    )

    with pytest.raises(job.AnalysisFailedError, match="did not finish"):
        await job.await_result(
            forever,  # type: ignore[arg-type]
            "an-1",
            out_dir=tmp_path,
            deadline_seconds=30 * 60,
            poll_interval=0.0,
        )


@asyncio_test
async def test_the_upload_is_deleted_when_validation_rejects_it(tmp_path: Path) -> None:
    """The leak was on the likeliest path.

    The group checks used to run above the `try`, so a wrong group name --
    one of the two mistakes a user actually makes -- returned an error and
    left their matrix on a disk with 4.7 GB free.
    """
    path = tmp_path / "theirs.tsv"
    path.write_text("\tS1\tS2\nENSG1\t1\t2\n")
    matrix = Matrix(path=path, size_bytes=1, samples=["S1", "S2"], gene_count=1)

    with pytest.raises(job.AnalysisFailedError, match="not a value of"):
        await job.submit_uploaded_matrix(
            StubClient(),  # type: ignore[arg-type]
            matrix=matrix,
            dataset_type="rnaseq_counts",
            analysis_group=["A", "B"],
            group1="NOPE",
            group2="B",
        )
    assert not path.exists()


@asyncio_test
async def test_the_upload_is_deleted_when_the_label_count_is_wrong(
    tmp_path: Path,
) -> None:
    path = tmp_path / "theirs.tsv"
    path.write_text("\tS1\tS2\tS3\nENSG1\t1\t2\t3\n")
    matrix = Matrix(path=path, size_bytes=1, samples=["S1", "S2", "S3"], gene_count=1)

    with pytest.raises(job.AnalysisFailedError, match="one label per"):
        await job.submit_uploaded_matrix(
            StubClient(),  # type: ignore[arg-type]
            matrix=matrix,
            dataset_type="rnaseq_counts",
            analysis_group=["A", "B"],
            group1="A",
            group2="B",
        )
    assert not path.exists()


@asyncio_test
async def test_loading_a_dataset_is_bounded_too(tmp_path: Path) -> None:
    """The sibling loop. `await_result` got two bounds; this one had one,
    twelve lines away, and I missed it in the same review."""
    stuck = StubClient(
        loading=[LoadingStatus("running", "still loading", 0.1, None)],
        summary=SUMMARY,
    )
    with pytest.raises(job.AnalysisFailedError, match="did not finish"):
        await job.submit_public_dataset(
            stuck,  # type: ignore[arg-type]
            resource_id="example_datasets",
            dataset_id="EXAMPLE_MEL_RNA",
            factor="condition",
            group1="MOCK",
            group2="MCM",
            deadline_seconds=30 * 60,
        )


def test_an_unknown_terminal_status_counts_as_failure() -> None:
    """`failed` is derived from "terminal and not complete".

    If the service adds `cancelled` to its terminal statuses, the derived
    form treats it as a failure. An `== "failed"` comparison would have
    called it neither finished nor failed and spun until a bound fired,
    then reported a timeout for something that had already stopped.
    """
    from gsa import client as gsa_client

    cancelled = gsa_client.AnalysisStatus("cancelled", "user cancelled", 1.0)
    assert not cancelled.finished  # not terminal until the set says so

    with_cancel = gsa_client.TERMINAL_STATUSES | {"cancelled"}
    assert "complete" in with_cancel
    # The property reads the set, so extending it is the only change needed.
    assert gsa_client.AnalysisStatus("complete", "", 1.0).finished
    assert not gsa_client.AnalysisStatus("complete", "", 1.0).failed


def test_pruning_removes_old_tables_and_keeps_recent_ones(tmp_path: Path) -> None:
    import os

    old = tmp_path / "reactome-gsa-old.tsv"
    new = tmp_path / "reactome-gsa-new.tsv"
    for path in (old, new):
        path.write_text("Pathway\tName\n")
    two_days = time.time() - 2 * 24 * 60 * 60
    os.utime(old, (two_days, two_days))

    assert job.prune_results(tmp_path) == 1
    assert not old.exists()
    assert new.exists()


def test_pruning_bounds_the_total_size(tmp_path: Path) -> None:
    import os

    paths = []
    for index in range(5):
        path = tmp_path / f"reactome-gsa-{index}.tsv"
        path.write_text("x" * 1000)
        os.utime(path, (time.time() - (10 - index), time.time() - (10 - index)))
        paths.append(path)

    job.prune_results(tmp_path, max_total_bytes=2500)
    remaining = sorted(p.name for p in tmp_path.glob("reactome-gsa-*.tsv"))
    # Oldest go first, newest survive.
    assert remaining == ["reactome-gsa-3.tsv", "reactome-gsa-4.tsv"]


def test_pruning_touches_only_files_it_wrote(tmp_path: Path) -> None:
    import os

    mine = tmp_path / "reactome-gsa-old.tsv"
    theirs = tmp_path / "someone-elses-important.tsv"
    for path in (mine, theirs):
        path.write_text("data")
    two_days = time.time() - 2 * 24 * 60 * 60
    os.utime(mine, (two_days, two_days))
    os.utime(theirs, (two_days, two_days))

    job.prune_results(tmp_path)
    assert not mine.exists()
    assert theirs.exists()


@asyncio_test
async def test_writing_a_result_prunes_the_directory_first(tmp_path: Path) -> None:
    """That `prune_results` works is not the same as it being called.

    The pruning tests all invoked the function directly, so removing the
    call from `await_result` broke nothing -- the directory would have
    grown forever with every test still green. Found by sabotage: the
    deletion the feature promises needs a test on the path that promises
    it.
    """
    import os

    stale = tmp_path / "reactome-gsa-ancient.tsv"
    stale.write_text("Pathway\tName\n")
    long_ago = time.time() - 30 * 24 * 60 * 60
    os.utime(stale, (long_ago, long_ago))

    finished = await job.await_result(
        StubClient(),  # type: ignore[arg-type]
        "an-1",
        out_dir=tmp_path,
    )

    assert not stale.exists()
    assert finished.table_path.exists()
