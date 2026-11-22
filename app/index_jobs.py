"""Precompute and persist job vectors before starting the API."""
from __future__ import annotations

import time

from .core import embedding_backend, get_jobs, retrieve


def main() -> None:
    jobs = get_jobs()
    if not jobs:
        raise SystemExit("No postings found. Run an ingestion script first.")
    started = time.perf_counter()
    retrieve({"preferred_roles": ["job matching index warmup"]}, jobs, k=1)
    elapsed = time.perf_counter() - started
    print(f"Indexed {len(jobs)} postings with {embedding_backend()} in {elapsed:.1f}s.")
    print("Vectors are cached in data/job_embeddings.npz for later searches.")


if __name__ == "__main__":
    main()
