"""Build a shuffled, source-blind candidate pool for manual relevance grading."""
from __future__ import annotations

import csv
import json
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import constraints_as_exclusions, embedding_backend, get_jobs, retrieve, rerank, violates

ROOT = Path(__file__).parent
PROFILES_PATH = ROOT / "benchmark_profiles.json"
LABELS_PATH = ROOT / "blind_pool.csv"
MANIFEST_PATH = ROOT / "blind_pool_manifest.json"
SYSTEMS = ("ngram", "embeddings", "hybrid", "hybrid_filtered")


def main() -> None:
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    jobs = get_jobs()
    if not jobs:
        raise SystemExit("No postings found. Import the job corpus first.")
    manifest: dict = {"created": date.today().isoformat(), "embedding_backend": embedding_backend(),
                      "corpus_size": len(jobs), "systems": {}, "profiles": {}}
    rows_by_key: dict[tuple[str, str], dict] = {}

    for entry in profiles:
        profile_id, profile = entry["id"], entry["profile"]
        manifest["profiles"][profile_id] = {"split": entry["split"], "statement": entry["statement"]}
        exclusions = constraints_as_exclusions(profile)
        ngram = retrieve(profile, jobs, 10, backend="hashed-ngram")
        embeddings = retrieve(profile, jobs, 10, backend="auto")
        hybrid = rerank(profile, retrieve(profile, jobs, 50, backend="auto"))[:10]
        eligible = [job for job in retrieve(profile, jobs, 50, backend="auto")
                    if not violates(job, exclusions)]
        hybrid_filtered = rerank(profile, eligible)[:10]
        rankings = {"ngram": ngram, "embeddings": embeddings,
                    "hybrid": hybrid, "hybrid_filtered": hybrid_filtered}
        manifest["systems"][profile_id] = {
            name: [job["id"] for job in ranked] for name, ranked in rankings.items()
        }
        for ranked in rankings.values():
            for job in ranked:
                key = (profile_id, str(job["id"]))
                rows_by_key.setdefault(key, {
                    "profile_id": profile_id,
                    "profile_statement": entry["statement"],
                    "posting_id": job["id"],
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                    "location": job.get("location", ""),
                    "url": job.get("url", ""),
                    "description": job.get("description", ""),
                    "grade": "",
                    "violates_dealbreaker": "",
                })

    rows = list(rows_by_key.values())
    random.Random(731904).shuffle(rows)
    fields = ("profile_id", "profile_statement", "posting_id", "title", "company", "location",
              "url", "description", "grade", "violates_dealbreaker")
    with LABELS_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Profiles: {len(profiles)} (6 dev, 4 held-out test); corpus: {len(jobs)} postings.")
    print(f"Blind pool: {len(rows)} unique profile/posting pairs in {LABELS_PATH}.")
    print(f"System/source mapping saved separately to {MANIFEST_PATH}; do not open it while grading.")
    print("Rubric: grade 0 = poor/no match; 1 = plausible/partial match; 2 = strong role, skill, and preference fit.")
    print("For violates_dealbreaker enter yes/no based on the profile's explicit hard constraints.")


if __name__ == "__main__":
    main()
