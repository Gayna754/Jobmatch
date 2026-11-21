import json, math
from pathlib import Path
ROOT=Path(__file__).parent
profiles=json.loads((ROOT/"profiles.json").read_text(encoding="utf-8")); jobs=json.loads((ROOT/"jobs.json").read_text(encoding="utf-8"))
# Hand-assigned graded relevance judgments. 0=irrelevant, 1=some fit, 2=strong fit.
strong={"ml":{"demo-1":2,"demo-2":1,"demo-5":1,"demo-6":1},"data":{"demo-5":2,"demo-1":1,"demo-2":1},"platform":{"demo-6":2,"demo-2":1,"demo-1":1}}
labels=[{"profile_id":p["id"],"job_id":j["id"],"relevance":strong.get(p["id"],{}).get(j["id"],0)} for p in profiles for j in jobs]
(ROOT/"labels.json").write_text(json.dumps(labels,indent=2),encoding="utf-8")
