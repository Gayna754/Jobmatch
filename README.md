# JobMatch — conversational job discovery

JobMatch is a Python service that turns a candidate's chat into a persistent structured profile, searches real job posts, enforces deal-breakers with deterministic rules, reranks the remaining candidates, and returns explanations grounded in the posting text.

## End-to-end flow

1. `/chat` receives a user message. An OpenAI-compatible tool-calling agent can use OpenAI or Gemini when its provider key is configured; a local parser supports a basic demo without a key.
2. The agent updates a profile with `preferred_roles`, `preferred_skills`, `preferred_locations`, `preferred_work_modes`, `seniority`, and `hard_constraints`.
3. Profiles, conversation turns, and jobs are stored in PostgreSQL when `DATABASE_URL` is set. SQLite is the zero-configuration local fallback.
4. Importers collect and clean postings from Remotive and Hacker News monthly “Who is hiring?” threads.
5. Sentence Transformers creates normalized posting embeddings. Vectors are cached in process memory and refreshed when the corpus changes. Without the optional model package, a deterministic hashed n-gram fallback is used.
6. Retrieval returns up to 50 candidates.
7. A deterministic hard-filter stage removes candidates that violate saved constraints. For example, `{"unpaid": false}` excludes unpaid listings.
8. The hybrid ranker computes an interpretable score from semantic similarity, skills, role, and location:

   `score = 100 × (0.4S + 0.3K + 0.2R + 0.1L)`

   These are starting weights, not tuned weights. Per-result component values are returned so they can be inspected and tuned against labeled data.
9. When configured, the LLM writes concise explanations and cites exact posting sentences. The service checks each quote against the source and falls back to a copied source sentence if it cannot validate the citation.
10. `/search` returns personalized matches, ranking scores, score components, matched skills, and explanation evidence.

## Setup on Windows

Run from the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Use PostgreSQL

Install Docker Desktop, then start the development database:

```powershell
docker compose up -d postgres
python -m pip install -r requirements-postgres.txt
$env:DATABASE_URL = "postgresql+psycopg://jobmatch:jobmatch_dev_only@localhost:5432/jobmatch"
```

The Compose credentials are for local development only. Use a private secret and managed database for deployment. To carry your existing SQLite data into PostgreSQL, run this once after setting `DATABASE_URL`:

```powershell
python migrate_sqlite_to_postgres.py --confirm
```

The migration copies profiles, jobs, and conversation turns from `data/jobmatch.sqlite3`. It is intended to be run once against an empty PostgreSQL database.

### Enable optional LLM chat and explanations

Install the optional OpenAI-compatible SDK, then select one provider and set its key in the same PowerShell window used to launch the API. For Gemini:

```powershell
python -m pip install -r requirements-llm.txt
$env:LLM_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "your-Gemini-AI-Studio-key"
$env:LLM_MODEL = "gemini-3.8-flash"
```

Google currently lists free input and output tokens for select Gemini models, subject to provider limits and terms. On the free tier, prompts may be used to improve Google products; do not send sensitive personal information. Check the current [Gemini API pricing and tier details](https://ai.google.dev/gemini-api/docs/pricing) before configuring a key. This service makes no provider calls unless the matching provider key is set.

For OpenAI, use `$env:LLM_PROVIDER = "openai"`, `$env:OPENAI_API_KEY = "your-key"`, and optionally `$env:OPENAI_MODEL = "gpt-4o-mini"`. Without a key, the local parser, deterministic filters, retrieval, ranker, and source-sentence explanations still work. `GET /health` reports the selected provider and whether a key is configured.

### Enable semantic embeddings

```powershell
python -m pip install -r requirements-ml.txt
```

The configured Sentence Transformers model downloads the first time it is used. The hashed n-gram fallback remains available for offline development, but is not semantic retrieval.

### Import postings and start the API

Keep `DATABASE_URL` set in the same PowerShell session when using PostgreSQL:

```powershell
python ingest_remotive.py --limit 1000
python ingest_hn.py --months 6
uvicorn app.main:app --reload
```

The public data sources are [Remotive](https://remotive.com/api/remote-jobs) and the [Hacker News API](https://github.com/HackerNews/API). Counts vary over time; Remotive returned only 16 posts in one initial fetch. HN posts tend to provide a larger monthly corpus, but formatting and metadata are user-submitted and inconsistent. Source URLs and IDs are retained where available. When a user rejects unpaid work, internship listings with no compensation disclosure are filtered conservatively as well as postings explicitly marked unpaid.

## Try it

Open `http://127.0.0.1:8000/docs`. Submit `POST /chat`:

```json
{
  "user_id": "demo",
  "message": "I am looking for a backend internship. I prefer Python and distributed systems. No unpaid internships, no frontend-heavy roles."
}
```

The resulting profile should include `Backend Engineer Intern`, the stated skills, and hard constraints with `unpaid` and `frontend_heavy` set to `false`. Locations remain empty unless the candidate states them; the service does not guess a location.

Then send `POST /search`:

```json
{
  "user_id": "demo",
  "k": 10,
  "mode": "hybrid"
}
```

Use `"mode": "filtered"` to compare hard-filtered semantic retrieval with the hybrid ranker. The production API never offers an unfiltered mode. `GET /profile/demo` reads saved preferences; `GET /health` shows database type, job count, and whether LLM support is enabled.

## Evaluation

Run:

```powershell
python -m eval.run_eval
```

The 30 graded pairs (3 profiles × 10 illustrative postings) compare:

| Pipeline stage | Recall@5 | NDCG@5 | Constraint violations | Evidence faithfulness |
|---|---:|---:|---:|---:|
| Hashed unigram/bigram cosine retrieval | 0.806 | 0.894 | 0.400 | 1.000 |
| Retrieval + hard filters | 1.000 | 0.991 | 0.000 | 1.000 |
| Retrieval + filters + hybrid ranker | 1.000 | 1.000 | 0.000 | 1.000 |

These results use **10 synthetic demo postings** and 30 hand-assigned relevance judgments (3 profiles × 10 postings), with the offline hashed unigram/bigram cosine-similarity fallback. The hybrid ranker reaches NDCG@5 1.000 versus 0.894 for retrieval alone on this small sanity-check fixture; the set has no held-out split and is not a live-job performance claim. The labels are in `eval/labels.json`.

## Project layout

```text
app/                 API, persistence, filtering, retrieval, ranking
eval/                profiles, postings, relevance labels, metrics
ingest_remotive.py   Remotive collector and cleaner
ingest_hn.py         Hacker News hiring-thread collector and cleaner
migrate_sqlite_to_postgres.py  Local-data migration helper
docker-compose.yml   Local PostgreSQL for development
```

## Limitations and next steps

- PostgreSQL stores durable data; embeddings are currently cached in the API process. For larger or multi-instance deployments, persist vectors in pgvector or use a FAISS index with an explicit refresh job.
- The current reranker is an interpretable weighted scorer, not a trained learning-to-rank model. Tune weights only on a held-out labeled dataset.
- Exclusion rules are deterministic but depend on the source fields and phrase patterns. Expand the rule set with labeled false-positive/false-negative examples.
- A public deployment needs authentication, privacy controls, rate limiting, monitoring, and a defined job-data retention policy.

## Suggested CV wording

**JobMatch | Python, FastAPI, PostgreSQL, Docker Compose**

Built a chat-based job-matching service with persistent structured preferences, hybrid ranking, deterministic deal-breaker filters, and source-grounded explanations. Added Remotive and Hacker News ingestion plus an evaluation harness comparing hashed n-gram retrieval, hard filters, and hybrid reranking.
