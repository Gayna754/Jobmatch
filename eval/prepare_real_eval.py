"""Create a human-label template from the local real-posting corpus."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import constraints_as_exclusions, get_jobs, retrieve, rerank, violates


ROOT = Path(__file__).parent
profiles = json.loads((ROOT / "real_profiles.json").read_text(encoding="utf-8"))


def main() -> None:
    jobs = get_jobs()
    if not jobs:
        raise SystemExit("No postings found. Import jobs first, then rerun this command.")
    pairs = []
    for profile in profiles:
        exclusions = constraints_as_exclusions(profile)
        retrieved = retrieve(profile, jobs, k=min(50, len(jobs)))
        filtered = [job for job in retrieved if not violates(job, exclusions)]
        hybrid = rerank(profile, filtered)
        pool: dict[str, dict] = {}
        seen_roles: set[tuple[str, str]] = set()
        for job in [*retrieved[:5], *hybrid[:5], *retrieved]:
            role_key = (str(job.get("company") or "").strip().casefold(),
                        str(job.get("title") or "").strip().casefold())
            if job["id"] in pool or role_key in seen_roles:
                continue
            pool[job["id"]] = job
            seen_roles.add(role_key)
            if len(pool) == 10:
                break
        for job in pool.values():
            pairs.append({
                "profile_id": profile["id"],
                "job_id": job["id"],
                "relevance": None,
                "title": job.get("title", ""),
                "company": job.get("company", ""),
                "location": job.get("location", ""),
                "url": job.get("url", ""),
                "description_excerpt": (job.get("description") or "")[:900],
            })
    out = ROOT / "real_judgments_to_label.json"
    out.write_text(json.dumps(pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(pairs)} real-posting/profile pairs to {out}.")
    print("Review each posting and replace relevance=null with 0 (not relevant), 1 (somewhat), or 2 (strong).")
    print("The pool includes the top 5 from semantic retrieval and filtered hybrid ranking for each profile.")


if __name__ == "__main__":
    main()
