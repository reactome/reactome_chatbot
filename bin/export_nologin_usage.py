import csv
import os
from argparse import ArgumentParser
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from util.secrets import get_db_uri

load_dotenv()

_db_name = os.getenv("POSTGRES_LANGGRAPH_DB")
_uri = get_db_uri(f"{_db_name}_no_login" if _db_name else None)
if _uri is None:
    raise SystemExit(
        "POSTGRES_LANGGRAPH_DB is not set, or no Postgres password is available. "
        "This script exports from the database; it cannot run without one."
    )
# Rebound as str: the check above does not narrow a module global for
# code inside a function.
LANGGRAPH_NOLOGIN_DB_URI: str = _uri


def build_query() -> str:
    return """
        SELECT
            thread_id,
            checkpoint_id,
            checkpoint->'ts' AS checkpoint_ts
        FROM
            checkpoints
        WHERE
            metadata->>'source' = 'input' AND
            NOT (metadata ? 'langgraph_node')
        ORDER BY
            checkpoint->'ts';
    """


def main(records_dir: Path) -> None:
    records_dir.mkdir(exist_ok=True)

    query: str = build_query()

    with psycopg.connect(LANGGRAPH_NOLOGIN_DB_URI) as conn, conn.cursor() as cur:
        cur.execute(query)
        header = [col.name for col in cur.description] if cur.description else None
        records = cur.fetchall()

    if len(records) == 0:
        print("No new records found.")
        return

    record_file = records_dir / "nologin_usage.csv"

    with open(record_file, mode="w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        if header:
            writer.writerow(header)
        writer.writerows(records)

    print("Wrote", record_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("records_dir", type=Path, nargs="?", default=Path("records"))
    args = parser.parse_args()
    main(**vars(args))
