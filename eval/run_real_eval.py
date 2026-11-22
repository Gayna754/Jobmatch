"""Score locally collected human judgments against the real ingested corpus."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core import constraints_as_exclusions, embedding_backend, get_jobs, grounded_reason, retrieve, rerank, violates


ROOT = Path(__file__).parent
profiles = json.loads((ROOT / "real_profiles.json").read_text(encoding="utf-8"))
label_path = ROOT / "real_judgments_to_label.json"


def ndcg(grades: list[int], ideal_grades: list[int], k: int) -> float:
    def dcg(values: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(values[:k]))
    ideal = dcg(sorted(ideal_grades, reverse=True))
    return dcg(grades) / ideal if ideal else 0.0


def main() -> None:
    if not label_path.exists():
        raise SystemExit("Run `python -m eval.prepare_real_eval` and label the generated pairs first.")
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    pending = [row for row in labels if row.get("relevance") not in (0, 1, 2)]
    if pending:
        raise SystemExit(f"{len(pending)} pairs still need relevance grades 0, 1, or 2.")
    grades = {(row["profile_id"], row["job_id"]): row["relevance"] for row in labels}
    jobs = get_jobs()
    by_id = {job["id"]: job for job in jobs}
    if any(job_id not in by_id for _, job_id in grades):
        raise SystemExit("Some judged job IDs are missing from the current local corpus; regenerate the label template.")

    metrics: dict[str, list[float]] = defaultdict(list)
    for profile in profiles:
        profile_id = profile["id"]
        exclusions = constraints_as_exclusions(profile)
        candidates = retrieve(profile, jobs, k=min(50, len(jobs)))
        systems = {
            "embeddings_only": candidates,
            "embeddings_plus_filters_rerank": rerank(
                profile, [job for job in candidates if not violates(job, exclusions)]
            ),
        }
        judged_ids = {job_id for pid, job_id in grades if pid == profile_id}
        relevant_ids = {job_id for job_id in judged_ids if grades[(profile_id, job_id)] > 0}
        for system, ranked in systems.items():
            # Only compute metrics for judged postings; pooled judgments cover both systems' top five.
            top = [job for job in ranked if job["id"] in judged_ids][:5]
            hits = sum(job["id"] in relevant_ids for job in top)
            metrics[system + ":recall@5"].append(hits / len(relevant_ids) if relevant_ids else 1.0)
            top_grades = [grades[(profile_id, job["id"])] for job in top]
            pool_grades = [grades[(profile_id, job_id)] for job_id in judged_ids]
            metrics[system + ":ndcg@5"].append(ndcg(top_grades, pool_grades, 5))
            metrics[system + ":constraint_violation_rate"].append(
                sum(violates(job, exclusions) for job in top) / len(top) if top else 0.0
            )
            faithfulness = []
            for job in top:
                reason = grounded_reason(profile, job)
                quote = reason.partition("“")[2].partition("”")[0]
                faithfulness.append(bool(quote) and quote in job.get("description", ""))
            metrics[system + ":reason_evidence_faithfulness"].append(
                sum(faithfulness) / len(faithfulness) if faithfulness else 1.0
            )

    report = {}
    for system in ("embeddings_only", "embeddings_plus_filters_rerank"):
        report[system] = {
            metric: round(sum(metrics[f"{system}:{metric}"]) / len(metrics[f"{system}:{metric}"]), 4)
            for metric in ("recall@5", "ndcg@5", "constraint_violation_rate", "reason_evidence_faithfulness")
        }
    print(f"Corpus: {len(jobs)} real postings; judged pairs: {len(labels)}; backend: {embedding_backend()}.")
    print("Recall/NDCG are measured over the manually judged candidate pool, not every corpus posting.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
