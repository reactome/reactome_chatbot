"""A three-collection Reactome bundle: its CSVs, its vector stores, its manifest.

`generate_reactome_bundle`, which `make` calls for Reactome, does it all, in an
order chosen so that a mistake is caught before it costs anything:

1. Fetch every source from Neo4j, then write one CSV per collection, holding
   every source that feeds it. Every row is labelled with its record type and
   every list becomes one readable field. Nothing is written until everything
   has been fetched.
2. Check the release and every CSV against the collection definitions.
3. Embed each collection into its vector store.
4. Write the manifest, last.

Existing CSVs are reused, so rebuilding with another embedding model does not
query the graph again. The CSVs can also be written on their own, into
<bundle-dir>/csv_files, without embedding anything, so the documents can be
read before any embedding is paid for and `make` can then reuse them:

    NEO4J_URI=bolt://localhost:7687 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=... \\
        PYTHONPATH=src python -m data_generation.reactome.bundle_generator <bundle-dir>

The manifest is what lets the chatbot tell this bundle from an older one and
read it correctly: the keyword index has to render each document exactly as
it was embedded, and only the manifest says which columns that was.
"""

import csv
import json
import os
import re
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma

from data_generation.embeddings import build_embeddings
from data_generation.metadata_csv_loader import MetaDataCSVLoader
from data_generation.reactome.collection_definitions import (
    COLLECTIONS,
    RECORD_TYPE,
    Collection,
)
from data_generation.reactome.neo4j_connector import (
    Neo4jConnector,
    fetch_release,
    fetch_source,
)
from util.bundle_manifest import INCOMPLETE, MANIFEST, SCHEMA_VERSION
from util.embedding_environment import release_from_name

CSV_DIR = "csv_files"

#: Written beside the CSVs: the release they came from. A build that reuses
#: the CSVs without contacting Neo4j still has to know it, to check it against
#: the release the bundle is named for.
RELEASE_FILE = "release.json"

#: Joins a list into one field. Not a comma: chemical names contain commas, as
#: in "PI(4,5)P2", and the model has to see where one name ends.
LIST_SEPARATOR = "; "

#: Columns holding sentences rather than names: several summations, or the
#: sections of a UniProt comment. Joined as prose, so they do not read
#: "Plasma.; Interacts with...".
PROSE_COLUMNS = frozenset({"summation"})
PROSE_SEPARATOR = " "

# Whitespace that includes a line break. The loader reads the file line by line
# and drops any line starting with "#", so a field spanning lines could lose
# part of itself without a word. One line per record also keeps the CSV
# readable by eye.
_LINE_BREAK = re.compile(r"\s*[\r\n]\s*")


#: A collection mixes record types that lack each other's columns. Skipping
#: empty fields keeps a protein from carrying a bare "st_id: " line. Recorded in
#: the manifest, because the keyword index must load documents the same way.
SKIP_EMPTY_FIELDS = True


# --- the CSVs ---------------------------------------------------------------


def serialise(value: Any, separator: str = LIST_SEPARATOR) -> str:
    """One field as text: lists joined, missing values and empty lists blank."""
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        parts = (serialise(item, separator) for item in value)
        return separator.join(part for part in parts if part)
    return _LINE_BREAK.sub(" ", str(value)).strip()


def collection_rows(
    connector: Neo4jConnector, collection: Collection
) -> list[dict[str, str]]:
    """Every source of one collection, labelled, as rows of text.

    Raises if a query returns a column the definitions do not place, if a
    placed column is returned by no query, or if a record appears twice.
    """
    placed = set(collection.columns)
    returned: set[str] = set()
    origin: dict[str, str] = {}
    rows: list[dict[str, str]] = []

    for source in collection.sources:
        fetched = fetch_source(connector, source.name)
        columns = set(fetched[0])
        if RECORD_TYPE in columns:
            raise ValueError(
                f"The {source.name} query returns {RECORD_TYPE}, which is added "
                "from the collection definitions, not from a query."
            )
        unplaced = sorted(columns - placed)
        if unplaced:
            raise ValueError(
                f"The {source.name} query returns {', '.join(unplaced)}, which the "
                f"{collection.name} collection does not place. Add each to its "
                "embedded or metadata columns in collection_definitions.py."
            )
        returned |= columns

        for row in fetched:
            merged = {
                column: serialise(
                    row.get(column),
                    PROSE_SEPARATOR if column in PROSE_COLUMNS else LIST_SEPARATOR,
                )
                for column in collection.columns
            }
            merged[RECORD_TYPE] = source.record_type
            # Every record has its own page, so a repeated link is a repeated
            # record: a normal and disease pair that overlap, or one record
            # returned twice. Either would become two documents for one thing.
            link = merged["url"]
            if link in origin:
                where = (
                    f"twice by the {source.name} query"
                    if origin[link] == source.name
                    else f"by both {origin[link]} and {source.name}"
                )
                raise ValueError(
                    f"{link} is returned {where}. Each record must appear once in "
                    f"the {collection.name} collection."
                )
            origin[link] = source.name
            # The fetch checks should make this impossible. Checked again on
            # the text actually written, because a nameless or linkless
            # document is one that can be neither read nor cited.
            if not merged["display_name"] or not link:
                raise ValueError(
                    f"A {source.name} row has no display_name or url once "
                    f"written as text: {row!r}"
                )
            if merged["display_name"].startswith("#"):
                raise ValueError(
                    f"{link} has a display name starting with '#', and the loader "
                    "drops every line that starts with one."
                )
            rows.append(merged)

    never_returned = sorted(placed - returned - {RECORD_TYPE})
    if never_returned:
        raise ValueError(
            f"The {collection.name} collection places {', '.join(never_returned)}, "
            "which none of its queries returns."
        )
    return rows


def write_collection_csv(
    rows: list[dict[str, str]], collection: Collection, csv_dir: Path
) -> Path:
    """Write one collection's CSV, replacing any earlier one."""
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"{collection.name}.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(collection.columns), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_collection_csvs(connector: Neo4jConnector, csv_dir: Path) -> dict[str, Path]:
    """Every collection's CSV and the release they came from.

    Fetches everything before writing anything. A query that fails halfway
    would otherwise leave some collections new and some old in the same
    directory, and nothing would say which.
    """
    release = fetch_release(connector)
    merged = {c.name: (c, collection_rows(connector, c)) for c in COLLECTIONS}
    paths = {
        name: write_collection_csv(rows, collection, csv_dir)
        for name, (collection, rows) in merged.items()
    }
    (csv_dir / RELEASE_FILE).write_text(json.dumps({"release": release}) + "\n")
    return paths


def read_csv_release(csv_dir: Path) -> int | None:
    """The release the CSVs in `csv_dir` came from, or None if unrecorded."""
    try:
        recorded = json.loads((csv_dir / RELEASE_FILE).read_text())
    except (OSError, ValueError):
        return None
    release = recorded.get("release") if isinstance(recorded, dict) else None
    return release if isinstance(release, int) else None


# --- the vector stores and the manifest -------------------------------------


def bundle_release(bundle_dir: Path) -> int | None:
    """The release a bundle directory is named for: `Release98` gives 98."""
    return release_from_name(bundle_dir.name)


def check_release(bundle_dir: Path) -> int:
    """The bundle's release, if its CSVs came from the same one. Raises if not.

    A bundle named for one release and built from another is wrong in a way
    nothing downstream reports: answers would quietly describe the wrong
    release, and cached answers would be kept or dropped against the wrong one.
    """
    expected = bundle_release(bundle_dir)
    if expected is None:
        raise ValueError(
            f"{bundle_dir} is not named for a release, like Release98, so there is "
            "nothing to check its contents against."
        )
    found = read_csv_release(bundle_dir / CSV_DIR)
    if found is None:
        raise ValueError(
            f"The CSVs in {bundle_dir / CSV_DIR} do not record which release they "
            "came from. Write them again, with make --force, from a graph that has a "
            "DBInfo version."
        )
    if found != expected:
        raise ValueError(
            f"{bundle_dir} is named for release {expected}, but its CSVs came from "
            f"release {found}. Rebuild them from the right graph, or build into a "
            f"directory named Release{found}."
        )
    return expected


def _csv_path(bundle_dir: Path, collection: Collection) -> Path:
    return bundle_dir / CSV_DIR / f"{collection.name}.csv"


def check_csv(bundle_dir: Path, collection: Collection) -> None:
    """Raise unless the collection's CSV has exactly the columns now defined.

    CSVs are reused between builds. One written before the definitions changed
    would be embedded under the old column choices while the manifest claimed
    the new ones, and the keyword index would then disagree with the vectors.
    """
    path = _csv_path(bundle_dir, collection)
    with open(path, newline="", encoding="utf-8") as handle:
        header = tuple(next(csv.reader(handle), []))
    if header != collection.columns:
        raise ValueError(
            f"{path} has columns {', '.join(header)}, but the {collection.name} "
            f"collection now defines {', '.join(collection.columns)}. Write the CSVs "
            "again with make --force."
        )


def build_collection_store(
    bundle_dir: Path,
    collection: Collection,
    hf_model: str | None,
    device: str | None = None,
) -> Counter[str]:
    """Embed one collection, replacing any earlier store. Returns rows per label."""
    check_csv(bundle_dir, collection)
    documents = MetaDataCSVLoader(
        file_path=str(_csv_path(bundle_dir, collection)),
        content_columns=list(collection.embedded),
        metadata_columns=list(collection.metadata),
        skip_empty_fields=SKIP_EMPTY_FIELDS,
        encoding="utf-8",
    ).load()
    if not documents:
        raise ValueError(f"{_csv_path(bundle_dir, collection)} has no rows.")

    # Chroma.from_documents adds to whatever is already there, so a second
    # build would store every document twice without a word.
    persist = bundle_dir / collection.name
    if persist.exists():
        shutil.rmtree(persist)
        # chromadb caches one client per path. Without this, the next write
        # fails with "attempt to write a readonly database".
        SharedSystemClient.clear_system_cache()

    store = Chroma.from_documents(
        documents=documents,
        embedding=build_embeddings(hf_model, device, chunk_size=400),
        persist_directory=str(persist),
    )
    stored = store._collection.count()
    # chromadb allows one set of client settings per path in a process, and
    # the retriever opens these stores with its own. Released here, so a
    # process that builds a bundle and then reads it -- a test, or a check
    # run straight after `make` -- is not refused.
    SharedSystemClient.clear_system_cache()
    if stored != len(documents):
        raise ValueError(
            f"The {collection.name} store holds {stored} documents, but "
            f"{len(documents)} were loaded."
        )
    return Counter(document.metadata[RECORD_TYPE] for document in documents)


def _describe(collection: Collection, record_types: Counter[str]) -> dict[str, Any]:
    return {
        "csv": f"{CSV_DIR}/{collection.name}.csv",
        "documents": sum(record_types.values()),
        "embedded": list(collection.embedded),
        "metadata": list(collection.metadata),
        "shown_to_model": list(collection.shown_to_model),
        "skip_empty_fields": SKIP_EMPTY_FIELDS,
        "sources": {
            source.name: {
                "record_type": source.record_type,
                "rows": record_types.get(source.record_type, 0),
            }
            for source in collection.sources
        },
    }


def build_bundle(
    bundle_dir: Path, hf_model: str | None, device: str | None = None
) -> Path:
    """Every collection's store, then the manifest. Returns the manifest's path.

    Everything that can be checked is checked before anything is embedded, so
    a stale CSV or a wrong release costs nothing.
    """
    release = check_release(bundle_dir)
    for collection in COLLECTIONS:
        check_csv(bundle_dir, collection)

    # Marked unfinished until the manifest is written, so a build that fails
    # halfway is refused rather than read as an old bundle.
    marker = bundle_dir / INCOMPLETE
    marker.touch()
    manifest_path = bundle_dir / MANIFEST
    manifest_path.unlink(missing_ok=True)

    collections = {
        collection.name: _describe(
            collection, build_collection_store(bundle_dir, collection, hf_model, device)
        )
        for collection in COLLECTIONS
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release": release,
        "embedding_model": hf_model,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "collections": collections,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    marker.unlink()
    return manifest_path


# --- make, and the CSVs on their own -----------------------------------------


def generate_reactome_bundle(
    embeddings_dir: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_username: str | None = None,
    neo4j_password: str | None = None,
    force: bool = False,
    hf_model: str | None = None,
    device: str | None = None,
    disease_variant_tsv: str | None = None,
) -> Path:
    """`make` for Reactome: the CSVs, then the stores and the manifest.

    Existing CSVs are reused unless `force`, as before, so rebuilding with
    another embedding model does not query the graph again. The release they
    came from was recorded when they were written, so it is still checked.
    """
    if disease_variant_tsv:
        raise ValueError(
            "The disease_variants collection is retired: the ewas query now "
            "carries each protein's disease variants. Drop --disease-variant-tsv."
        )
    bundle_dir = Path(embeddings_dir)
    csv_dir = bundle_dir / CSV_DIR
    reusable = all(_csv_path(bundle_dir, c).is_file() for c in COLLECTIONS)
    if force or not reusable:
        connector = Neo4jConnector(
            uri=neo4j_uri, user=neo4j_username, password=neo4j_password
        )
        try:
            write_collection_csvs(connector, csv_dir)
        finally:
            connector.close()
    else:
        print(f"Using the existing CSVs in {csv_dir}; pass --force to query Neo4j.")
    return build_bundle(bundle_dir, hf_model, device)


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit(
            "usage: python -m data_generation.reactome.bundle_generator <bundle-dir>"
        )
    connector = Neo4jConnector(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USERNAME"),
        password=os.environ.get("NEO4J_PASSWORD"),
    )
    csv_dir = Path(argv[0]) / CSV_DIR
    try:
        paths = write_collection_csvs(connector, csv_dir)
    finally:
        connector.close()
    release = read_csv_release(csv_dir)
    print(
        f"release {release}"
        if release is not None
        else "release unknown: the graph has no DBInfo version. The CSVs can be "
        "read, but a bundle will not be built from them."
    )
    for collection in COLLECTIONS:
        path = paths[collection.name]
        with open(path, newline="", encoding="utf-8") as handle:
            counts: dict[str, int] = {}
            for row in csv.DictReader(handle):
                counts[row[RECORD_TYPE]] = counts.get(row[RECORD_TYPE], 0) + 1
        detail = ", ".join(f"{label}: {n}" for label, n in counts.items())
        print(f"{path}  ({detail})")


if __name__ == "__main__":
    main(sys.argv[1:])
