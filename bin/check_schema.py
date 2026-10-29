#!/usr/bin/env python
"""Fail when the Chainlit schema has drifted from the installed Chainlit.

initdb/init.sql is the only thing standing between a Chainlit upgrade and
`psycopg.errors.UndefinedColumn` on every read and write. It drifted for 22
months once and nothing complained, because the data layer builds its INSERT
column list from every non-None key of the dict it is handed -- there is no
whitelist -- so a key added upstream becomes a column this repository does not
have, and the first anyone hears of it is a user-visible failure.

So the invariant is one-directional: **each table must be a superset of the
shape written into it.** Extra columns are harmless. Missing ones are the bug.

Three modes, because two different gaps need covering:

    PYTHONPATH=./src .venv/bin/python bin/check_schema.py --sql initdb/init.sql
        Parse the file. Needs no database, so it runs on a developer's machine
        and from checks.sh. Catches init.sql missing a column.

    PYTHONPATH=./src .venv/bin/python bin/check_schema.py
        Query a live database. Uses the app's own path -- util.secrets.get_db_uri
        -- so the unix socket and any Vault credential are honoured. That socket
        exists only inside the container, so on the host this exits 2.

    PYTHONPATH=./src .venv/bin/python bin/check_schema.py \
        --dsn postgresql://user:pw@localhost:5432/chatbio_chainlit
        The above against an explicit DSN. This is what CI runs, against a
        database actually built from initdb/init.sql, which is the only way to
        also catch a live database that was never migrated.

The venv is not optional: psycopg lives in it, and the shebang resolves `python`
through PATH. Inside the container that is already the venv, which is why the
short form works there and nowhere else.

Exit codes: 0 matches, 1 drift, 2 could not look. Keep them distinct -- a
connection problem must not read as a pass, which is the same reason checks.sh
exists at all.
"""

import argparse
import os
import re
import sys
from dataclasses import fields, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import psycopg

from util.secrets import get_db_uri


def _shape_keys(shape: Any) -> set[str]:
    """Every field name of a TypedDict, pydantic model or dataclass.

    All three appear here: StepDict and ElementDict are TypedDicts, Feedback is
    a pydantic model. Resolving them from the installed package rather than from
    a list in this file is the point -- a literal list would drift exactly the
    way init.sql did.
    """
    model_fields = getattr(shape, "model_fields", None)
    if model_fields:
        return set(model_fields)
    if is_dataclass(shape):
        return {f.name for f in fields(shape)}
    keys: set[str] = set()
    for base in reversed(shape.__mro__):
        keys.update(getattr(base, "__annotations__", {}))
    return keys


def _expected() -> dict[str, tuple[set[str], dict[str, str]]]:
    """table -> (fields written into it, fields never written, and why)."""
    from chainlit.element import ElementDict
    from chainlit.step import StepDict
    from chainlit.types import Feedback, ThreadDict

    return {
        "steps": (
            _shape_keys(StepDict),
            {"feedback": "read path only: Step.to_dict() never emits it"},
        ),
        "elements": (_shape_keys(ElementDict), {}),
        "threads": (
            _shape_keys(ThreadDict),
            {
                "steps": "nested read model, not a column",
                "elements": "nested read model, not a column",
            },
        ),
        # upsert_feedback serialises asdict(Feedback). FeedbackDict is what the
        # read path constructs by hand and is a different shape -- checking that
        # one instead would let the real write shape drift unchecked.
        "feedbacks": (_shape_keys(Feedback), {}),
        # create_user is a literal INSERT, so the shape is that literal. UserDict
        # carries display_name, which is never written; checking it here would
        # report a missing column that must never exist.
        "users": ({"id", "identifier", "createdAt", "metadata"}, {}),
    }


def _columns_from_sql(text: str) -> dict[str, set[str]]:
    """Column names per table, read out of initdb/init.sql.

    A text parse rather than a live query, so the check runs where the database
    cannot be reached -- which is every developer's machine, since Postgres is
    unix-socket-only and /sockets/postgres/ exists only inside the container.
    It is a regex over a file this repository writes, in one fixed layout.

    It does NOT catch a live database that was never migrated from an older
    init.sql. The CI job, which builds a real Postgres from the same file, is
    what catches that.
    """
    tables: dict[str, set[str]] = {}
    pattern = r'CREATE TABLE (?:IF NOT EXISTS )?"?(\w+)"?\s*\((.*?)\n\)'
    for match in re.finditer(pattern, text, re.DOTALL):
        body = match.group(2)
        # A column line opens with a quoted name. FOREIGN KEY and CONSTRAINT
        # lines open with a keyword, so the anchor excludes them without needing
        # a list of keywords to skip.
        tables[match.group(1)] = set(
            re.findall(r'^\s*"([A-Za-z_]\w*)"', body, re.MULTILINE)
        )
    return tables


def _columns(conn: psycopg.Connection, table: str) -> set[str]:
    sql = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = current_schema()
    """
    with conn.cursor() as cur:
        cur.execute(sql, (table,))
        return {row[0] for row in cur.fetchall()}


def _compare(have: dict[str, set[str]]) -> int:
    """The one verdict both modes reach through."""
    drift: list[str] = []
    for table, (keys, excludes) in _expected().items():
        columns = have.get(table, set())
        if not columns:
            drift.append(f"{table}: table does not exist")
            continue
        wanted = keys - set(excludes)
        missing = wanted - columns
        if missing:
            drift.append(f'{table}: missing {", ".join(sorted(missing))}')
        unused = columns - wanted
        if unused:
            print(f'  note: {table} has unused {", ".join(sorted(unused))}')

    if drift:
        print("Chainlit schema drift:", file=sys.stderr)
        for line in drift:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nThe data layer sends every non-None key, so a column missing "
            "here fails the INSERT that uses it.",
            file=sys.stderr,
        )
        return 1

    print("schema matches the installed Chainlit")
    return 0


def _print_expected() -> int:
    """What each table must hold, for writing initdb/init.sql.

    Names only. The SQL type is not inferred: showInput is TEXT in the schema
    although it is typed bool | str, and a guessed type that is wrong is worse
    than no type.
    """
    for table, (keys, excludes) in _expected().items():
        wanted = keys - set(excludes)
        print(f"-- {table}: {len(wanted)} columns")
        for name in sorted(wanted):
            print(f'    "{name}" ...')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the Chainlit schema against the installed Chainlit."
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help="Defaults to get_db_uri(POSTGRES_CHAINLIT_DB).",
    )
    parser.add_argument(
        "--sql",
        default=None,
        help="check the text of initdb/init.sql; needs no database",
    )
    parser.add_argument(
        "--print-expected",
        action="store_true",
        help="print the required columns and exit; needs no database",
    )
    args = parser.parse_args(argv)

    print(f"chainlit {version('chainlit')}")

    if args.print_expected:
        return _print_expected()

    if args.sql:
        text = Path(args.sql).read_text(encoding="utf-8")
        return _compare(_columns_from_sql(text))

    dsn = args.dsn or get_db_uri(os.getenv("POSTGRES_CHAINLIT_DB"))
    if not dsn:
        print(
            "no DSN: pass --dsn or --sql, or set POSTGRES_CHAINLIT_DB and run "
            "where the Postgres socket exists",
            file=sys.stderr,
        )
        return 2

    live: dict[str, set[str]] = {}
    conn: psycopg.Connection | None = None
    try:
        conn = psycopg.connect(dsn)
        for table in _expected():
            live[table] = _columns(conn, table)
    except psycopg.Error as exc:
        print(f"cannot read the schema: {exc}", file=sys.stderr)
        return 2
    finally:
        if conn is not None:
            conn.close()

    return _compare(live)


if __name__ == "__main__":
    sys.exit(main())
