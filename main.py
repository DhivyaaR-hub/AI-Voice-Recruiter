import json
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="AI Voice Recruiter Backend",
    version="1.0.0",
    description="AI-powered interview and answer evaluation API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Configure models with automatic fallback
PRIMARY_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")
FALLBACK_MODEL_ID = "gemini-3.5-flash-lite"

client = genai.Client(api_key=API_KEY) if API_KEY else None

REFERENCE_FILE = os.environ.get("PLAGIARISM_REFERENCE_FILE", "")
SIMILARITY_THRESHOLD = float(
    os.environ.get("SIMILARITY_THRESHOLD", "0.82")
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class CandidateModel(BaseModel):
    name: str
    email: str
    jobRole: str


class AnswerEvaluationModel(BaseModel):
    candidate_name: str
    job_role: str
    question: str
    answer: str
    previous_answers: List[str] = Field(default_factory=list)


# --------------------------------------------------
# TEXT SIMILARITY CHECK
# --------------------------------------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def similarity_percent(a: str, b: str) -> int:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)

    if not a_norm or not b_norm:
        return 0

    return round(
        SequenceMatcher(None, a_norm, b_norm).ratio() * 100
    )


def load_reference_text() -> str:
    if not REFERENCE_FILE:
        return ""

    try:
        return Path(REFERENCE_FILE).read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        print(f"Could not load reference text: {exc}")
        return ""


def check_similarity(
    answer: str,
    previous_answers: List[str],
) -> dict:
    candidates = [
        ("an earlier answer in this interview", old)
        for old in previous_answers
        if old and old.strip()
    ]

    reference_text = load_reference_text()

    if reference_text.strip():
        chunks = [
            chunk.strip()
            for chunk in re.split(
                r"\n\s*\n|(?<=[.!?])\s+",
                reference_text,
            )
            if len(chunk.strip()) >= 35
        ]

        candidates.extend(
            ("configured reference material", chunk)
            for chunk in chunks
        )

    if len(normalize_text(answer).split()) < 8:
        return {
            "similarity_flag": False,
            "similarity_score": 0,
            "similarity_reason": (
                "Answer is too short for a meaningful similarity check."
            ),
        }

    best_score = 0
    best_source = ""

    for source_name, candidate in candidates:
        score = similarity_percent(answer, candidate)

        if score > best_score:
            best_score = score
            best_source = source_name

    flagged = (
        bool(best_source)
        and best_score >= round(SIMILARITY_THRESHOLD * 100)
    )

    if flagged:
        reason = (
            f"High text similarity ({best_score}%) with {best_source}. "
            "Review manually; this is not proof of cheating."
        )
    elif best_source:
        reason = (
            f"Highest measured similarity was {best_score}% "
            f"with {best_source}."
        )
    else:
        reason = (
            "No comparison text was available. "
            "This does not establish whether the answer is original."
        )

    return {
        "similarity_flag": flagged,
        "similarity_score": best_score,
        "similarity_reason": reason,
    }


# --------------------------------------------------
# GEMINI HELPER WITH FALLBACK & BACKOFF
# --------------------------------------------------

def ask_gemini(prompt: str) -> str:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini API key is not configured. "
                "Set the GEMINI_API_KEY environment variable."
            ),
        )

    # Try primary model first, fallback to secondary model if unavailable
    models_to_try = [PRIMARY_MODEL_ID, FALLBACK_MODEL_ID]

    for model_id in models_to_try:
        for attempt in range(3):
            start = time.perf_counter()

            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.4,
                        response_mime_type="application/json",
                    ),
                )

                elapsed = time.perf_counter() - start
                print(f"Gemini API call ({model_id}): {elapsed:.2f}s")

                result = response.text

                if not result:
                    raise HTTPException(
                        status_code=502,
                        detail="Gemini returned an empty response.",
                    )

                return result

            except HTTPException:
                raise

            except Exception as exc:
                error_text = str(exc)
                print(
                    f"Model {model_id} attempt {attempt + 1}/3 failed: "
                    f"{error_text}"
                )

                retryable = any(
                    code in error_text
                    for code in ("429", "500", "502", "503", "504")
                )

                if not retryable or attempt == 2:
                    break  # Break inner loop to try fallback model

                # Exponential backoff (2s, 4s)
                time.sleep(2.0 * (2 ** attempt))

    raise HTTPException(
        status_code=502,
        detail="AI service is temporarily unavailable due to high demand. Please try again shortly.",
    )


# --------------------------------------------------
# JSON RESPONSE PARSER
# --------------------------------------------------

def parse_json_response(text: str) -> dict:
    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group(0))

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise HTTPException(
        status_code=502,
        detail="AI returned an invalid JSON response.",
    )


# --------------------------------------------------
# HOME AND HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Voice Recruiter Backend is running",
        "docs": "/docs",
        "api_key_configured": client is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "AI Voice Recruiter",
        "ai_configured": client is not None,
    }


# --------------------------------------------------
# START INTERVIEW
# --------------------------------------------------

@app.post("/start-interview")
def start_interview(candidate: CandidateModel):
    if not candidate.name.strip():
        raise HTTPException(
            status_code=422,
            detail="Candidate name is required.",
        )

    if not candidate.jobRole.strip():
        raise HTTPException(
            status_code=422,
            detail="Job role is required.",
        )

    prompt = f"""
You are a professional interviewer conducting a job interview.

Candidate name: {candidate.name}
Job role: {candidate.jobRole}

Generate the first interview question appropriate for this job role.
Ask exactly one clear, relevant question.
Do not provide the answer or evaluate the candidate yet.

Return valid JSON:
{{
    "question": "Your first interview question"
}}
"""

    result = parse_json_response(ask_gemini(prompt))
    question = str(result.get("question", "")).strip()

    if not question:
        raise HTTPException(
            status_code=502,
            detail="AI did not generate an interview question.",
        )

    return {
        "success": True,
        "candidate_name": candidate.name,
        "job_role": candidate.jobRole,
        "question": question,
    }


# --------------------------------------------------
# EVALUATE ANSWER
# --------------------------------------------------

@app.post("/evaluate-answer")
def evaluate_answer(data: AnswerEvaluationModel):
    request_start = time.perf_counter()
    answer = data.answer.strip()

    if not answer:
        raise HTTPException(
            status_code=422,
            detail="Answer cannot be empty.",
        )

    similarity_start = time.perf_counter()

    similarity = check_similarity(
        answer,
        data.previous_answers,
    )

    print(
        f"Similarity check: "
        f"{time.perf_counter() - similarity_start:.2f}s"
    )

    prompt = f"""
You are a fair and professional AI interview evaluator.

Candidate: {data.candidate_name}
Job role: {data.job_role}
Interview question: {data.question}
Candidate's answer: {answer}

Evaluate only the answer provided.
Do not assume the candidate is dishonest.
Do not reward an empty, irrelevant, or non-answer response.
Do not give a positive score simply because the candidate said
"thank you", "I don't know", or similar words.

Consider:
- Relevance to the question
- Technical or role-related accuracy
- Clarity and completeness
- Practical understanding

Give a score from 0 to 100 based on the answer's actual quality.
Give specific, constructive feedback.
Generate one appropriate next interview question.
Keep the feedback concise and the next question to one sentence.

Return valid JSON using exactly this structure:
{{
    "score": 0,
    "feedback": "Specific feedback about the answer",
    "strengths": ["Strength 1"],
    "improvements": ["Improvement 1"],
    "next_question": "The next interview question"
}}

Be honest and consistent. If the answer does not address the
question, the score should be low and the feedback should explain why.
"""

    result = parse_json_response(ask_gemini(prompt))

    try:
        score = int(result.get("score", 0))
    except (ValueError, TypeError):
        score = 0

    score = max(0, min(100, score))

    feedback = str(
        result.get("feedback", "No feedback was provided.")
    ).strip()

    strengths = result.get("strengths", [])
    improvements = result.get("improvements", [])

    if not isinstance(strengths, list):
        strengths = []

    if not isinstance(improvements, list):
        improvements = []

    next_question = str(
        result.get("next_question", "")
    ).strip()

    print(
        f"Total evaluation request: "
        f"{time.perf_counter() - request_start:.2f}s"
    )

    return {
        "success": True,
        "candidate_name": data.candidate_name,
        "job_role": data.job_role,
        "question": data.question,
        "answer": answer,
        "score": score,
        "feedback": feedback,
        "strengths": strengths,
        "improvements": improvements,
        "next_question": next_question,
        **similarity,
    }