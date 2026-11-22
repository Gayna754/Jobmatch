"""Score a completed yes/no faithfulness audit of generated explanations."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
PROFILES = json.loads((ROOT / "benchmark_profiles.json").read_text(encoding="utf-8"))


def main() -> None:
    path = ROOT / "explanation_audit.csv"
    if not path.exists():
        raise SystemExit("Run `python -m eval.build_explanation_audit` first.")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 30 or any(row.get("faithful", "").lower() not in {"yes", "no"} for row in rows):
        raise SystemExit("Enter yes/no for all 30 rows in explanation_audit.csv before scoring.")
    split_by_profile = {profile["id"]: profile["split"] for profile in PROFILES}
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[split_by_profile[row["profile_id"]]].append(row["faithful"].lower() == "yes")
    report = {split: {"audited": len(values), "faithfulness": round(sum(values) / len(values), 4)}
              for split, values in grouped.items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
