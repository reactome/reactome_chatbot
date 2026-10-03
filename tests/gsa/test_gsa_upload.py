"""Accepting a matrix, and refusing one in terms the user can act on."""

from pathlib import Path

import pytest

from gsa import upload


def write(tmp_path: Path, text: str, name: str = "matrix.tsv") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


GOOD = "\tS1\tS2\tS3\tS4\nENSG1\t10\t20\t30\t40\nENSG2\t5\t6\t7\t8\n"


def test_accepts_a_real_matrix_shape(tmp_path: Path) -> None:
    # The header's first cell is empty, which is what the service's own
    # export produces -- the measured example began with a tab.
    matrix = upload.validate(write(tmp_path, GOOD))
    assert matrix.samples == ["S1", "S2", "S3", "S4"]
    assert matrix.gene_count == 2


def test_accepts_comma_separated(tmp_path: Path) -> None:
    matrix = upload.validate(write(tmp_path, GOOD.replace("\t", ","), "m.csv"))
    assert matrix.samples == ["S1", "S2", "S3", "S4"]


def test_a_comma_inside_a_tsv_does_not_split_the_header(tmp_path: Path) -> None:
    # Tab wins when both are present. Splitting on the comma would turn
    # "Tumour, left" into two samples and mangle the analysis rather than
    # fail it.
    text = "\tTumour, left\tTumour, right\nENSG1\t1\t2\n"
    assert upload.validate(write(tmp_path, text)).samples == [
        "Tumour, left",
        "Tumour, right",
    ]


def test_refuses_a_file_over_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(upload.MAX_UPLOAD_BYTES_ENV, "100")
    with pytest.raises(upload.UploadRejectedError, match="limit"):
        upload.validate(write(tmp_path, GOOD * 100))


def test_the_cap_defaults_well_below_chainlit(tmp_path: Path) -> None:
    # Chainlit ships max_size_mb = 500 and the host has 4.7 GB free.
    assert upload.max_upload_bytes() == 20 * 1024 * 1024


@pytest.mark.parametrize("bad", ["", "0", "-1", "not-a-number"])
def test_an_unusable_cap_falls_back_rather_than_disabling_the_limit(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dangerous direction: a typo must not mean "no limit".
    monkeypatch.setenv(upload.MAX_UPLOAD_BYTES_ENV, bad)
    assert upload.max_upload_bytes() == upload.DEFAULT_MAX_UPLOAD_BYTES


def test_refuses_an_empty_file(tmp_path: Path) -> None:
    with pytest.raises(upload.UploadRejectedError, match="empty"):
        upload.validate(write(tmp_path, ""))


def test_refuses_something_that_is_not_a_matrix(tmp_path: Path) -> None:
    with pytest.raises(upload.UploadRejectedError, match="expression matrix"):
        upload.validate(write(tmp_path, "just some prose\nand another line\n"))


def test_refuses_a_header_with_no_data(tmp_path: Path) -> None:
    with pytest.raises(upload.UploadRejectedError, match="no data rows"):
        upload.validate(write(tmp_path, "\tS1\tS2\tS3\n"))


def test_refuses_a_single_sample(tmp_path: Path) -> None:
    # An analysis compares two groups. One column cannot be compared with
    # anything, and the service would say so minutes later in R's words.
    text = "gene\tS1\tS2\nENSG1\t1\t2\n"
    matrix = upload.validate(write(tmp_path, text))
    assert len(matrix.samples) == 2

    with pytest.raises(upload.UploadRejectedError, match="only one sample"):
        upload.validate(write(tmp_path, "gene\tS1\nENSG1\t1\nENSG2\t2\n", "one.tsv"))


def test_discard_removes_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, GOOD)
    upload.discard(path)
    assert not path.exists()


def test_discard_is_safe_to_repeat(tmp_path: Path) -> None:
    # Called from a `finally`, so it runs on paths that may already be gone.
    path = write(tmp_path, GOOD)
    upload.discard(path)
    upload.discard(path)


def test_a_large_file_reports_an_unknown_gene_count_not_a_sentinel(
    tmp_path: Path,
) -> None:
    """Counting every row of a 20,000-gene file answers a question nobody
    asked, so it stops early -- but "stopped early" must not be encoded as
    a number. A `-1` here is the kind of value that reaches a user as
    "your matrix has -1 genes".
    """
    rows = "".join(f"G{i}\t1\t2\n" for i in range(6000))
    matrix = upload.validate(write(tmp_path, "\tS1\tS2\n" + rows, "big.tsv"))

    assert matrix.gene_count is None
    assert matrix.samples == ["S1", "S2"]


def test_a_small_file_still_reports_a_real_count(tmp_path: Path) -> None:
    # The control: if every file reported None, the test above would pass
    # and the count would be useless.
    assert upload.validate(write(tmp_path, GOOD)).gene_count == 2


# --- review, area 2 -----------------------------------------------------------


def test_an_r_style_header_keeps_its_first_sample(tmp_path: Path) -> None:
    # write.table writes no cell for the gene column; header[1:] dropped S1.
    text = "S1\tS2\tS3\tS4\nTP53\t10\t12\t30\t33\nMDM2\t1\t2\t3\t4\nBAX\t5\t6\t7\t8\n"
    assert upload.validate(write(tmp_path, text)).samples == ["S1", "S2", "S3", "S4"]


def test_a_csv_reaches_the_service_tab_separated(tmp_path: Path) -> None:
    text = 'gene,ctrl_1,"treated, day 1"\nTP53,1,2\nMDM2,3,4\nBAX,5,6\n'
    matrix = upload.validate(write(tmp_path, text, "m.csv"))
    assert matrix.text.splitlines()[0] == "gene\tctrl_1\ttreated, day 1"
    assert "," not in matrix.text.splitlines()[1]


@pytest.mark.parametrize("encoding", ["cp1252", "utf-16"])
def test_windows_encodings_read_the_same_at_validation_and_at_submit(
    tmp_path: Path, encoding: str
) -> None:
    # Validation decoded with replacement, submission strictly: such a file
    # passed, was labelled, then failed every retry.
    text = "\tcontrôle_1\tcontrôle_2\tbehandelt\nTP53\t1\t2\t3\nMDM2\t4\t5\t6\nBAX\t7\t8\t9\n"
    path = tmp_path / "m.tsv"
    path.write_bytes(text.encode(encoding))
    matrix = upload.validate(path)
    assert matrix.samples == ["contrôle_1", "contrôle_2", "behandelt"]
    assert "contrôle_1" in matrix.text


def test_too_many_columns_is_refused(tmp_path: Path) -> None:
    header = "\t" + "\t".join(f"s{i}" for i in range(upload.MAX_SAMPLES + 5))
    with pytest.raises(upload.UploadRejectedError, match="columns"):
        upload.validate(
            write(tmp_path, header + "\nTP53" + "\t1" * (upload.MAX_SAMPLES + 5) + "\n")
        )


def test_blank_lines_cannot_make_validation_read_the_whole_file(tmp_path: Path) -> None:
    import time

    text = "\tS1\tS2\n" + "\n" * 3_000_000 + "TP53\t1\t2\n"
    started = time.perf_counter()
    with pytest.raises(upload.UploadRejectedError):
        upload.validate(write(tmp_path, text))
    assert time.perf_counter() - started < 1.0
