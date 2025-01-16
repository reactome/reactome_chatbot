import csv
import os
from argparse import ArgumentParser
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

CHAINLIT_DB_URI = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@postgres:5432/{os.getenv('POSTGRES_CHAINLIT_DB')}?sslmode=disable"


def build_query(since_timestamp: str | None) -> str:
    if since_timestamp is None:
        since_timestamp = ""
    query = f"""
        SELECT
            steps."threadId",
            steps."createdAt",
            steps.name,
            steps.type,
            steps.output,
            feedbacks.value,
            feedbacks.comment
        FROM steps
        LEFT JOIN
            feedbacks ON steps."parentId" = feedbacks."forId"
        WHERE
            steps.type IN ('user_message', 'assistant_message') AND
            steps."createdAt" > '{since_timestamp}'
        ORDER BY
            (
                SELECT MIN(s."createdAt")
                FROM steps s
                WHERE s."threadId" = steps."threadId"
            ),
            steps."createdAt";
    """
    return query


def last_record_timestamp(records_dir: Path) -> str | None:
    record_names: list[str] = list(f.stem for f in records_dir.glob("records_*.csv"))
    if len(record_names) > 0:
        last_record: str = max(record_names)
        return last_record[len("records_") :]
    else:
        return None


def main(records_dir: Path):
    records_dir.mkdir(exist_ok=True)

    since_timestamp: str | None = last_record_timestamp(records_dir)
    query: str = build_query(since_timestamp)

    with psycopg.connect(CHAINLIT_DB_URI) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            header = [col.name for col in cur.description] if cur.description else None
            records = cur.fetchall()

    if len(records) == 0:
        print("No new records found.")
        return

    latest_timestamp: str = max(row[1] for row in records)

    record_file = records_dir / f"records_{latest_timestamp}.csv"

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
