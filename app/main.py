"""FastAPI routes and preference-aware chat agent."""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import add_turn, get_jobs, get_profile, history, search_jobs, update_preferences

app = FastAPI(
    title="JobMatch API",
    description="Conversational job matching with durable PostgreSQL profiles, hard filters, hybrid ranking, and cited explanations.",
    version="1.1.0",
)
logger = logging.getLogger("jobmatch.agent")


class ChatRequest(BaseModel):
    user_id: str = Field(default="demo", min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=4000)


class SearchRequest(BaseModel):
    user_id: str = Field(default="demo", min_length=1, max_length=120)
    k: int = Field(default=10, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(hybrid|filtered)$")


def extract_preferences(text: str) -> dict[str, Any]:
    """Local fallback for common language. The LLM tool handles richer phrasing."""
    patch: dict[str, Any] = {}
    low = text.lower()
    negative = re.search(r"\b(no|never|avoid|don't want|do not want|can't take|cannot take)\b", low)
    if negative:
        constraints = {}
        for key, patterns in {
            "unpaid": (r"\bunpaid\b", r"\bvolunteer\b", r"\buncompensated\b"),
            "sales": (r"\bsales\b", r"\bselling\b", r"\baccount executive\b"),
            "commission-only": (r"\bcommission[- ]only\b",),
            "on-site": (r"\bon[- ]site\b", r"\bin office\b"),
            "frontend_heavy": (r"\bfrontend[- ]heavy\b", r"\bfront[- ]end[- ]heavy\b"),
        }.items():
            if any(re.search(pattern, low) for pattern in patterns):
                constraints[key] = False
        if constraints:
            patch["hard_constraints"] = constraints

    role_match = re.search(
        r"(?:looking for|seeking)\s+(?:a|an)?\s*(backend|back-end|data science|machine learning|software|frontend|front-end)\s+(?:engineering\s+)?(internship|intern|role|job)",
        text, re.I,
    )
    if role_match:
        role = {
            "backend": "Backend Engineer", "back-end": "Backend Engineer",
            "data science": "Data Science", "machine learning": "Machine Learning",
            "software": "Software Engineer", "frontend": "Frontend Engineer",
            "front-end": "Frontend Engineer",
        }[role_match.group(1).lower()]
        if "intern" in role_match.group(2).lower():
            role += " Intern"
            patch["seniority"] = "Intern"
        patch["preferred_roles"] = [role]
    else:
        role_interest = re.search(r"(?:interested in|want to work on)\s+([^.!?]+)", text, re.I)
        if role_interest:
            patch["preferred_roles"] = [part.strip(" ,") for part in re.split(r",|\band\b", role_interest.group(1), flags=re.I) if part.strip(" ,")]

    skills_match = re.search(
        r"(?:i know|i prefer|prefer|my skills (?:include|are)|skills include|experienced in|work with)\s+([^.!?]+)",
        text, re.I,
    )
    if skills_match:
        preference_text = skills_match.group(1).strip(" ,.!?").lower()
        work_mode_only = bool(re.fullmatch(
            r"(?:remote|hybrid|on[- ]?site)(?:\s+(?:work|roles?|jobs?))?", preference_text
        ))
        if not work_mode_only:
            patch["preferred_skills"] = [
                item.strip(" ,") for item in re.split(r",|\band\b|\balong with\b", skills_match.group(1), flags=re.I)
                if item.strip(" ,")
            ]

    location_match = re.search(r"(?:located in|based in|locations?\s*(?:are|:))\s+([^.!?]+)", text, re.I)
    if location_match:
        patch["preferred_locations"] = [part.strip(" ,") for part in re.split(r",|\band\b", location_match.group(1), flags=re.I) if part.strip(" ,")]
    work_modes = [name for name in ("remote", "hybrid", "on-site", "onsite") if re.search(r"\b" + name + r"\b", low)]
    if work_modes:
        patch["preferred_work_modes"] = list(dict.fromkeys(work_modes))
    return patch


PREFERENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_preferences",
        "description": "Save structured job preferences. Preserve every explicit hard constraint across turns.",
        "parameters": {
            "type": "object", "properties": {
                "preferred_roles": {"type": "array", "items": {"type": "string"}},
                "preferred_skills": {"type": "array", "items": {"type": "string"}},
                "preferred_locations": {"type": "array", "items": {"type": "string"}},
                "preferred_work_modes": {"type": "array", "items": {"type": "string"}},
                "seniority": {"type": "string"},
                "hard_constraints": {"type": "object", "additionalProperties": {"type": "boolean"}},
            }, "additionalProperties": False,
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_jobs",
        "description": "Search current postings using the saved profile. Hard constraints are applied before ranking.",
        "parameters": {"type": "object", "properties": {
            "k": {"type": "integer", "minimum": 1, "maximum": 10},
        }, "additionalProperties": False},
    },
}


def run_agent(user_id: str, message: str) -> tuple[str, list[dict[str, Any]] | None]:
    patch = extract_preferences(message)
    if patch:
        update_preferences(user_id, patch)
    if not os.getenv("OPENAI_API_KEY"):
        if re.search(r"\b(find|search|show|recommend|match)\b", message, re.I):
            results = search_jobs(user_id, 5)
            return (f"I found {len(results)} matches. Your saved constraints were applied.", results)
        return ("I saved your preferences. Ask me to find jobs when you’re ready.", None)

    try:
        from openai import OpenAI
        client = OpenAI()
        messages = [{
            "role": "system",
            "content": (
                "You are a job search assistant. Use update_preferences to save user preferences in structured form. "
                "Treat negative constraints as durable; never delete an existing constraint. Use search_jobs when asked. "
                "Only describe returned jobs and their cited evidence; never invent facts."
            ),
        }, *history(user_id)]
        results = None
        for _ in range(3):
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), messages=messages,
                tools=[PREFERENCE_TOOL, SEARCH_TOOL], tool_choice="auto", temperature=0.2,
            )
            assistant = response.choices[0].message
            messages.append(assistant.model_dump(exclude_none=True))
            if not assistant.tool_calls:
                return (assistant.content or "I’ve updated your job search profile.", results)
            for call in assistant.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                if call.function.name == "update_preferences":
                    tool_result: Any = {"saved_preferences": update_preferences(user_id, args)}
                elif call.function.name == "search_jobs":
                    results = search_jobs(user_id, min(max(int(args.get("k", 5)), 1), 10))
                    tool_result = [{key: job.get(key) for key in ("title", "company", "location", "url", "score", "why")} for job in results]
                else:
                    tool_result = {"error": "Unknown tool"}
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(tool_result, ensure_ascii=False)})
        return ("I’ve updated your profile. Use the search endpoint to see your ranked matches.", results)
    except Exception as exc:
        # Log only the exception class: SDK error messages can echo fragments of a bad key.
        error_type = type(exc).__name__
        logger.error("OpenAI agent request failed (%s)", error_type)
        if error_type == "AuthenticationError":
            detail = "OpenAI rejected the API key. Check that the replacement key was entered correctly."
        elif error_type in {"RateLimitError", "InsufficientQuotaError"}:
            detail = "The API request hit a quota or rate limit. Check API billing, credits, and project limits."
        elif error_type in {"APIConnectionError", "APITimeoutError"}:
            detail = "Could not reach the OpenAI API. Check the network and try again."
        else:
            detail = f"The API request failed ({error_type}). Check the server terminal for its status."
        return (f"I saved your preferences locally, but the model call failed. {detail}", None)


@app.get("/health", tags=["System"])
def health() -> dict[str, Any]:
    database = "postgresql" if os.getenv("DATABASE_URL", "").startswith(("postgresql", "postgres://")) else "sqlite"
    return {"status": "ok", "job_count": len(get_jobs()), "database": database,
            "llm_enabled": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/chat", tags=["Conversation"])
def chat(request: ChatRequest) -> dict[str, Any]:
    user_id = request.user_id.strip()
    if not user_id:
        raise HTTPException(422, "user_id cannot be blank")
    add_turn(user_id, "user", request.message)
    reply, results = run_agent(user_id, request.message)
    add_turn(user_id, "assistant", reply)
    return {"reply": reply, "profile": get_profile(user_id), "results": results}


@app.post("/search", tags=["Matching"])
def search(request: SearchRequest) -> dict[str, Any]:
    profile = get_profile(request.user_id)
    return {"results": search_jobs(request.user_id, request.k, request.mode), "profile": profile}


@app.get("/profile/{user_id}", tags=["Conversation"])
def profile(user_id: str) -> dict[str, Any]:
    return get_profile(user_id)
