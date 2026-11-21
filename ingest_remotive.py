"""Import recent remote jobs from Remotive's public API."""
import argparse, hashlib, httpx
import html, re
from app.core import upsert_jobs

def clean_description(value):
    value = re.sub(r"(?i)<br\s*/?>|</p\s*>", "\n", value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in value.splitlines() if line.strip())

def fetch(limit=1000):
    response=httpx.get("https://remotive.com/api/remote-jobs",params={"limit":limit},timeout=60)
    response.raise_for_status()
    payload=response.json()
    jobs=[]
    for x in payload.get("jobs",[]):
        jid=str(x.get("id") or hashlib.sha1(x.get("url","").encode()).hexdigest())
        jobs.append({"id":"remotive:"+jid,"title":x.get("title",""),"company":x.get("company_name",""),"location":x.get("candidate_required_location",""),"url":x.get("url",""),"description":clean_description(x.get("description","")),"employment_type":x.get("job_type",""),"salary":x.get("salary",""),"source":"Remotive"})
    return jobs
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--limit",type=int,default=1000); args=p.parse_args()
    jobs=fetch(args.limit); upsert_jobs(jobs); print(f"Imported {len(jobs)} Remotive postings")
