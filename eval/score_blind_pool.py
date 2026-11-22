"""Compute dev and held-out metrics from completed blind-pool labels."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
LABELS = ROOT / "blind_pool.csv"
MANIFEST = ROOT / "blind_pool_manifest.json"
OUTPUT = ROOT / "blind_pool_results.json"
SYSTEMS = ("ngram", "embeddings", "hybrid", "hybrid_filtered")


def ndcg(grades: list[int], ideal: list[int], k: int) -> float:
    def dcg(values: list[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(values[:k]))
    denominator = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / denominator if denominator else 0.0


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def main() -> None:
    if not LABELS.exists() or not MANIFEST.exists():
        raise SystemExit("Run `python -m eval.build_blind_pool` first.")
    with LABELS.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row.get("grade") not in {"0", "1", "2"} or row.get("violates_dealbreaker", "").lower() not in {"yes", "no"}:
            raise SystemExit("Every row needs grade 0/1/2 and violates_dealbreaker yes/no before scoring.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    grade_map = {(row["profile_id"], row["posting_id"]): int(row["grade"]) for row in rows}
    violation_map = {(row["profile_id"], row["posting_id"]): row["violates_dealbreaker"].lower() == "yes" for row in rows}
    per_split: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for profile_id, profile_meta in manifest["profiles"].items():
        split = profile_meta["split"]
        pool_ids = [posting_id for pid, posting_id in grade_map if pid == profile_id]
        relevant_ids = {posting_id for posting_id in pool_ids if grade_map[(profile_id, posting_id)] > 0}
        ideal = [grade_map[(profile_id, posting_id)] for posting_id in pool_ids]
        for system in SYSTEMS:
            ranked = manifest["systems"][profile_id][system]
            top10 = ranked[:10]
            top_grades = [grade_map[(profile_id, posting_id)] for posting_id in top10]
            per_split[split][system]["ndcg@5"].append(ndcg(top_grades, ideal, 5))
            per_split[split][system]["ndcg@10"].append(ndcg(top_grades, ideal, 10))
            per_split[split][system]["recall@10"].append(
                sum(posting_id in relevant_ids for posting_id in top10) / len(relevant_ids)
                if relevant_ids else 0.0
            )
            per_split[split][system]["violation_rate@10"].append(
                sum(violation_map[(profile_id, posting_id)] for posting_id in top10) / len(top10)
                if top10 else 0.0
            )

    report = {split: {system: {metric: mean(values) for metric, values in metrics.items()}
                      for system, metrics in systems.items()}
              for split, systems in per_split.items()}
    result = {"pool_only": True, "profiles": {pid: info["split"] for pid, info in manifest["profiles"].items()},
              "judged_pairs": len(rows), "systems": report}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Scored {len(rows)} pooled judgments; metrics cover pooled postings only.")
    print(json.dumps(report, indent=2))
    print(f"Saved results to {OUTPUT}.")


if __name__ == "__main__":
    main()
