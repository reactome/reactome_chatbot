import csv
import os
from argparse import ArgumentParser
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

LANGGRAPH_NOLOGIN_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_LANGGRAPH_DB')}_no_login?sslmode=disable"


def build_query() -> str:
    query = """
        SELECT
            thread_id,
            checkpoint_id,
            checkpoint->'ts' AS checkpoint_ts
        FROM
            checkpoints
        WHERE
            checkpoint_ns = '' AND
            parent_checkpoint_id IS NULL
        ORDER BY
            checkpoint->'ts';
    """
    return query


def main(records_dir: Path):
    records_dir.mkdir(exist_ok=True)

    query: str = build_query()

    with psycopg.connect(LANGGRAPH_NOLOGIN_DB_URI) as conn:
        with conn.cursor() as cur:
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
