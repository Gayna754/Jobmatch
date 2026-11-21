"""Copy the existing local SQLite profile, job, and chat data into DATABASE_URL."""
import argparse
import json
import sqlite3

from app.core import add_turn, update_preferences, upsert_jobs


def migrate(source: str) -> None:
    with sqlite3.connect(source) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "jobs" in tables:
            jobs = [dict(row) for row in db.execute("SELECT * FROM jobs")]
            print(f"Jobs copied: {upsert_jobs(jobs)}")
        if "profiles" in tables:
            profiles = db.execute("SELECT user_id,data FROM profiles").fetchall()
            for row in profiles:
                update_preferences(row["user_id"], json.loads(row["data"]))
            print(f"Profiles copied: {len(profiles)}")
        if "turns" in tables:
            turns = db.execute("SELECT user_id,role,content FROM turns ORDER BY id").fetchall()
            for row in turns:
                add_turn(row["user_id"], row["role"], row["content"])
            print(f"Conversation turns copied: {len(turns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/jobmatch.sqlite3")
    parser.add_argument("--confirm", action="store_true", help="Required to write data to DATABASE_URL")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("Pass --confirm after setting DATABASE_URL to the PostgreSQL destination")
    migrate(args.source)
