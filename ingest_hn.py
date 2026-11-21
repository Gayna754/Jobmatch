"""Import jobs from recent Hacker News "Who is hiring?" monthly threads."""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from app.core import clean_description, replace_hn_jobs

ALGOLIA = "https://hn.algolia.com/api/v1/search_by_date"
FIREBASE = "https://hacker-news.firebaseio.com/v0"
HEADERS = {"User-Agent": "JobMatch portfolio project (public HN API client)"}


def clean_markup(value: str) -> str:
    return clean_description(value)


def get_hiring_threads(client: httpx.Client, months: int) -> list[dict]:
    response = client.get(ALGOLIA, params={
        "query": "Ask HN Who is hiring", "tags": "story", "hitsPerPage": 100,
    })
    response.raise_for_status()
    hits = response.json().get("hits", [])
    return [hit for hit in hits if re.search(r"who is hiring", hit.get("title", ""), re.I)][:months]


def fetch_comments(client: httpx.Client, thread: dict) -> list[dict]:
    response = client.get(f"{FIREBASE}/item/{thread['objectID']}.json")
    response.raise_for_status()
    story = response.json() or {}
    comment_ids = story.get("kids", [])
    items = []
    for offset in range(0, len(comment_ids), 20):
        batch = comment_ids[offset:offset + 20]
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(client.get, f"{FIREBASE}/item/{comment_id}.json") for comment_id in batch]
            for future in as_completed(futures):
                result = future.result()
                if result.is_success:
                    item = result.json() or {}
                    if item.get("text") and not item.get("deleted") and not item.get("dead"):
                        items.append(item)
        time.sleep(0.05)  # Keep the public API request rate modest.
    return items


def parse_job(comment: dict, thread: dict) -> dict | None:
    description = clean_markup(comment.get("text", ""))
    if len(description) < 80:
        return None
    first_line = next((line.strip(" -*\t") for line in description.splitlines() if line.strip()), "Hacker News role")
    parts = [part.strip() for part in first_line.split("|") if part.strip()]
    company = parts[0][:120] if parts else "Unknown"
    title = parts[1][:180] if len(parts) > 1 else first_line[:180]
    location = parts[2][:160] if len(parts) > 2 else "Unspecified"
    location = re.split(r"\b(?:i am|i'm|we are|we're|about us|description:)\b", location, maxsplit=1, flags=re.I)[0].strip(" ,-/") or "Unspecified"

    # Monthly HN threads occasionally contain candidate self-advertisements, not job ads.
    personal_name = bool(re.fullmatch(r"[A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+){1,2}", company))
    first_person_bio = bool(re.search(r"\b(i am|i'm|my skills|my experience|i enjoy|i specialize)\b", description, re.I))
    employer_intent = bool(re.search(r"\b(we are hiring|we're hiring|we are looking for|we're looking for|join our team|apply now|open position)\b", description, re.I))
    if personal_name and first_person_bio and not employer_intent:
        return None

    header = " | ".join(parts[2:5])
    employment_type = "internship" if re.search(r"\bintern(ship)?\b", header, re.I) else ""
    if not employment_type:
        employment_type = next((value for value in ("full-time", "part-time", "contract", "temporary")
                                if re.search(r"\b" + value.replace("-", "[- ]?") + r"\b", header, re.I)), "")
    salary_match = re.search(r"(?:\$|USD\s?|₹|£|€)\s?[\d,]+(?:\s?[-–]\s?(?:\$|USD\s?|₹|£|€)?\s?[\d,]+)?\s?(?:k|K)?", first_line)
    salary = salary_match.group(0).strip() if salary_match else ("paid" if re.search(r"\bpaid\b", first_line, re.I) else "")
    urls = re.findall(r"https?://[^\s<>\]\[()]+", description)
    url = next(
        (
            candidate.rstrip(".,;)")
            for candidate in urls
            if "news.ycombinator.com/item" not in candidate
        ),
        "",
    )
    if not url and comment.get("id"):
        url = f"https://news.ycombinator.com/item?id={comment['id']}"
    return {
        "id": f"hn:{comment.get('id') or hashlib.sha1(description.encode()).hexdigest()}",
        "title": title, "company": company, "location": location, "url": url,
        "description": description, "employment_type": employment_type, "salary": salary,
        "source": f"Hacker News: {thread.get('title', 'Who is hiring?')}",
    }


def fetch_jobs(months: int = 6) -> list[dict]:
    with httpx.Client(timeout=30, headers=HEADERS, follow_redirects=True) as client:
        threads = get_hiring_threads(client, months)
        jobs = []
        for thread in threads:
            for comment in fetch_comments(client, thread):
                job = parse_job(comment, thread)
                if job:
                    jobs.append(job)
        return jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=6, help="Number of recent monthly threads to import")
    args = parser.parse_args()
    imported = replace_hn_jobs(fetch_jobs(max(1, args.months)))
    print(f"Imported or refreshed {imported} job posts from recent HN hiring threads")
