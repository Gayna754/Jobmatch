"""Compare semantic retrieval, hard-filtered retrieval, and the full hybrid ranker."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import constraints_as_exclusions, grounded_reason, retrieve, rerank, violates

ROOT = Path(__file__).parent
profiles = json.loads((ROOT / "profiles.json").read_text(encoding="utf-8"))
jobs = json.loads((ROOT / "jobs.json").read_text(encoding="utf-8"))
labels = json.loads((ROOT / "labels.json").read_text(encoding="utf-8"))
judgments = {(item["profile_id"], item["job_id"]): item["relevance"] for item in labels}


def evaluate(stage: str, k: int = 5) -> dict:
    recalls, ndcgs, violations, faithfulness = [], [], [], []
    for profile in profiles:
        exclusions = constraints_as_exclusions(profile)
        candidates = retrieve(profile, jobs, 50)
        if stage != "embeddings_only":
            candidates = [job for job in candidates if not violates(job, exclusions)]
        ranked = rerank(profile, candidates) if stage == "hybrid_ranker" else sorted(
            candidates, key=lambda job: -job["semantic_score"]
        )
        top = ranked[:k]
        relevant = {
            job["id"] for job in jobs
            if judgments[(profile["id"], job["id"])] > 0 and not violates(job, exclusions)
        }
        hits = sum(job["id"] in relevant for job in top)
        recalls.append(hits / len(relevant) if relevant else 1.0)
        gains = [judgments[(profile["id"], job["id"])] for job in top]
        dcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(gains))
        ideal = sorted((judgments[(profile["id"], job_id)] for job_id in relevant), reverse=True)[:k]
        idcg = sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        violations.extend(violates(job, exclusions) for job in top)
        for job in top:
            reason = grounded_reason(profile, job)
            quote = reason.partition("“")[2].partition("”")[0]
            faithfulness.append(bool(quote) and quote in job.get("description", ""))
    return {
        "recall@5": sum(recalls) / len(recalls),
        "ndcg@5": sum(ndcgs) / len(ndcgs),
        "constraint_violation_rate": sum(violations) / len(violations) if violations else 0.0,
        "reason_evidence_faithfulness": sum(faithfulness) / len(faithfulness) if faithfulness else 1.0,
    }


if __name__ == "__main__":
    print(f"Evaluation on {len(labels)} graded profile/job pairs; {len(profiles)} profiles, {len(jobs)} postings.")
    print("Fixture source: illustrative demo postings; not live listing performance.")
    print(json.dumps({
        "embeddings_only": evaluate("embeddings_only"),
        "embeddings_plus_filters": evaluate("embeddings_plus_filters"),
        "embeddings_filters_plus_hybrid_ranker": evaluate("hybrid_ranker"),
    }, indent=2))
