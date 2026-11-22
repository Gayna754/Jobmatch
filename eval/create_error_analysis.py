"""Export up to three worst held-out hybrid-with-filters cases for review."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).parent


def main() -> None:
    label_path = ROOT / "blind_pool.csv"
    manifest_path = ROOT / "blind_pool_manifest.json"
    if not label_path.exists() or not manifest_path.exists():
        raise SystemExit("Build and label the blind pool before running error analysis.")
    with label_path.open(newline="", encoding="utf-8-sig") as stream:
        labels = list(csv.DictReader(stream))
    if any(row.get("grade") not in {"0", "1", "2"} for row in labels):
        raise SystemExit("Complete relevance grades before running error analysis.")
    by_pair = {(row["profile_id"], row["posting_id"]): row for row in labels}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = []
    for profile_id, info in manifest["profiles"].items():
        if info["split"] != "test":
            continue
        ranking = manifest["systems"][profile_id]["hybrid_filtered"]
        ranked_ids = set(ranking[:5])
        false_positives = [by_pair[(profile_id, posting_id)] for posting_id in ranking[:5]
                           if int(by_pair[(profile_id, posting_id)]["grade"]) == 0]
        missed_strong = [row for (pid, _), row in by_pair.items()
                         if pid == profile_id and int(row["grade"]) == 2 and row["posting_id"] not in ranked_ids]
        if not false_positives and not missed_strong:
            continue
        bad = false_positives[0] if false_positives else None
        missed = missed_strong[0] if missed_strong else None
        cases.append({
            "profile_id": profile_id,
            "profile_statement": info["statement"],
            "top5_false_positive_id": bad["posting_id"] if bad else "",
            "top5_false_positive_title": bad["title"] if bad else "",
            "false_positive_grade": bad["grade"] if bad else "",
            "missed_strong_id": missed["posting_id"] if missed else "",
            "missed_strong_title": missed["title"] if missed else "",
            "failure_analysis": "",
            "proposed_fix_for_dev": "",
        })
    # Prioritize a hard false positive; retain profile order to keep the selection reproducible.
    cases.sort(key=lambda item: (not bool(item["top5_false_positive_id"]), item["profile_id"]))
    cases = cases[:3]
    path = ROOT / "error_analysis.csv"
    fields = ("profile_id", "profile_statement", "top5_false_positive_id", "top5_false_positive_title",
              "false_positive_grade", "missed_strong_id", "missed_strong_title", "failure_analysis",
              "proposed_fix_for_dev")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)
    print(f"Wrote {len(cases)} held-out error cases to {path}; fill the final two columns after review.")
    print("Use these to describe errors. Do not tune on these test cases; use dev examples, then collect a fresh test set for a new headline score.")


if __name__ == "__main__":
    main()
