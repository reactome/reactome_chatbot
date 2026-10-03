"""Accept an expression matrix from a user, or refuse it clearly.

**Why a cap well below Chainlit's.** `.chainlit/config.toml` ships
`max_size_mb = 500`. The host this runs on has 4.7 GB free of 88 GB, and
`~/update-beta-chat.sh` refuses to deploy under 6 GB, so a handful of
default-sized uploads would take the chat down *and* block the fix. A real
matrix is far smaller: the 16-sample melanoma example is 1.2 MB, and 20 MB
covers a large study comfortably.

**Why shape is checked before the service sees it.** ReactomeGSA answers a
malformed matrix minutes later, through a status field, with a message
written for someone reading R output. A user who pasted the wrong file
deserves to hear so immediately and in their own terms.
"""

import contextlib
import csv
import io
import os
from dataclasses import dataclass
from pathlib import Path

MAX_UPLOAD_BYTES_ENV = "GSA_MAX_UPLOAD_BYTES"
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: Two columns -- a gene identifier and one sample -- is still a table, and
#: saying "this is not an expression matrix" to someone who uploaded one
#: with a single sample is both wrong and unhelpful. So the structural floor
#: is two columns, and "not enough samples to compare" is a separate,
#: specific refusal below.
MIN_COLUMNS = 2
MIN_DATA_ROWS = 2


class UploadRejectedError(ValueError):
    """The file cannot be analysed, with a reason meant for the user."""


@dataclass(frozen=True)
class Matrix:
    """A validated matrix, and what could be learned about it cheaply."""

    path: Path
    size_bytes: int
    samples: list[str]
    #: `None` when the file was too long to finish counting. Not `-1`: a
    #: sentinel of the same type as a real count leaves the function, and
    #: the first thing anyone does with a gene count is show it to someone.
    #: "unknown" is a fact; "-1 genes" is a bug wearing a number.
    gene_count: int | None
    #: How the upload separates cells; `text` always hands the service tabs.
    delimiter: str = "\t"

    @property
    def text(self) -> str:
        """The matrix, tab-separated. Never logged, never shown, never prompted.

        Decoded the way `validate` decoded it -- a strict read here let a
        Windows-1252 file pass validation and then fail at submit, every
        retry (review, area 2) -- and tab-separated whatever was uploaded:
        the service reads only tabs, and a CSV went through unconverted.
        A blocking read: call it with `asyncio.to_thread`.
        """
        text = decode(self.path.read_bytes())
        if self.delimiter == "\t":
            return text
        rows = csv.reader(io.StringIO(text))
        return "\n".join("\t".join(cell.strip() for cell in row) for row in rows if row)


def max_upload_bytes() -> int:
    raw = os.environ.get(MAX_UPLOAD_BYTES_ENV)
    if not raw:
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_UPLOAD_BYTES
    return value if value > 0 else DEFAULT_MAX_UPLOAD_BYTES


#: More samples than any expression study here; a bound on what one upload
#: can make the server hold (a 20 MB one-line header was 7 million names).
MAX_SAMPLES = 1000
#: Lines read before deciding it is a matrix, blank ones included: blank
#: lines used to skip the early exit, and 21 million of them stalled every
#: session for two seconds.
MAX_LINES_READ = 5000


def decode(raw: bytes) -> str:
    """The file's text: UTF-8 (with or without BOM), UTF-16, or Windows-1252.

    Excel on Windows writes 1252 or UTF-16; reading those as UTF-8 with
    replacement showed the reader garbled sample names to label.
    """
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _split(line: str) -> list[str]:
    # Tab first: it is what the service wants and what every export
    # produces. Comma only if there is no tab at all, because a TSV cell can
    # legitimately contain a comma and splitting on it would silently
    # mangle the header rather than fail.
    return line.split("\t") if "\t" in line else line.split(",")


def validate(path: Path) -> Matrix:
    """Check an uploaded file and describe it, or raise `UploadRejectedError`.

    Reads the header and counts lines; does not hold the whole matrix.
    """
    size = path.stat().st_size
    limit = max_upload_bytes()
    if size > limit:
        raise UploadRejectedError(
            f"That file is {size / 1e6:.1f} MB and the limit is "
            f"{limit / 1e6:.0f} MB. An expression matrix this large is "
            f"unusual -- if it is right, it needs to go through "
            f"reactome.org/PathwayBrowser rather than the chat."
        )
    if size == 0:
        raise UploadRejectedError("That file is empty.")

    header: list[str] = []
    first_row: list[str] = []
    rows = 0
    counted = True
    delimiter = "\t"
    # Iterated, not split: 21 million blank lines as a list is the problem
    # this loop's bound exists to avoid.
    for index, raw_line in enumerate(io.StringIO(decode(path.read_bytes()))):
        line = raw_line.rstrip("\r\n")
        if index > MAX_LINES_READ:
            # Enough to know whether it is a matrix. Counting every gene of
            # a 20,000-row file is work nobody asked for -- and a file of
            # blank lines must not make us read all of it.
            if rows <= MIN_DATA_ROWS:
                raise UploadRejectedError(
                    "That file is mostly empty lines. It needs a header row "
                    "naming the samples, then one row per gene."
                )
            counted = False
            break
        if not line.strip():
            continue
        if not header:
            delimiter = "\t" if "\t" in line else ","
            header = _split(line)
            if len(header) > MAX_SAMPLES + 1:
                raise UploadRejectedError(
                    f"That file has {len(header) - 1:,} columns. This handles up "
                    f"to {MAX_SAMPLES:,} samples."
                )
            continue
        if not first_row:
            first_row = _split(line)
        rows += 1

    if len(header) < MIN_COLUMNS:
        raise UploadRejectedError(
            "That does not look like an expression matrix. It needs a header "
            "row naming the samples, then one row per gene -- tab- or "
            "comma-separated, with at least two samples to compare."
        )
    if rows == 0:
        raise UploadRejectedError("That file has a header but no data rows.")

    # The first header cell labels the gene column and is often blank --
    # the measured example's header starts with a tab.
    # R's write.table writes no cell for the gene column, so the header is
    # one short of the rows; taking header[1:] then dropped the first
    # sample, and its label would have gone to the wrong column (review, 2).
    names = header if len(first_row) == len(header) + 1 else header[1:]
    samples = [name.strip() for name in names if name.strip()]
    if len(samples) < 2:
        raise UploadRejectedError(
            "There is only one sample in that file. A gene set analysis "
            "compares two groups of samples, so it needs at least two."
        )

    return Matrix(
        path=path,
        size_bytes=size,
        samples=samples,
        gene_count=rows if counted else None,
        delimiter=delimiter,
    )


def discard(path: Path) -> None:
    """Delete an uploaded file. Safe to call twice, and on a missing file.

    Called once the matrix has been submitted, whether or not the analysis
    then succeeds: the service has its own copy, and this host has 4.7 GB.
    """
    # A file that cannot be deleted must not fail an analysis that has
    # already been submitted; the disk check on the next deploy will notice.
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
