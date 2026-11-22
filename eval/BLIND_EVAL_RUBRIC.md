# Blind real-posting evaluation

## Relevance grades

- **0 — poor fit:** the posting does not match the profile's target role family or essential skills.
- **1 — partial fit:** it is plausibly useful, but has a material mismatch in role, skills, location, work mode, or stated seniority.
- **2 — strong fit:** the role and important skills align, and there is no material mismatch with stated preferences.

Grade against the profile statement and the full posting, not the order in which the row appears. A preferred location or mode is not a deal-breaker unless the profile explicitly says it is mandatory. Conflicting or vague preferences should not be silently simplified; use grade 1 when fit is genuinely ambiguous.

## Deal-breaker label

Set `violates_dealbreaker` to **yes** only when the posting contradicts a hard constraint stated in the profile (for example, unpaid work when the profile says no unpaid work, or hybrid/on-site when it says remote only). Otherwise use **no**. Unknown pay is not automatically unpaid except where the profile explicitly seeks an internship and its rule treats undisclosed compensation as a violation.

## Blinding and split

`benchmark_profiles.json` fixes the six dev and four test profile IDs before this new benchmark is run. `build_blind_pool.py` shuffles the label CSV and omits system names. `blind_pool_manifest.json` contains the system-to-posting mapping; keep it closed while grading. Use dev profiles for any scoring changes. Report test results once, after decisions are finalized. If a test error drives a code change, preserve the original score and use a new untouched test set for a new headline result.

The pool is the union of each system's top 10. Recall and ranking metrics therefore apply to judged pooled postings only, not all postings in the corpus.
