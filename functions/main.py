import json
import os
import re
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from firebase_admin import firestore, initialize_app
from firebase_functions import firestore_fn
from google import genai
from google.genai import types


PROJECT_ID = "deep-sync-production"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
initialize_app(options={"projectId": PROJECT_ID})
db = firestore.client()


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("LLM returned empty text.")

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        raise ValueError("LLM response did not contain a JSON object.")
    return json.loads(match.group(0))


def _get_genai_client() -> genai.Client:
    load_dotenv()
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")
    return genai.Client(api_key=api_key)


def _build_cv_prompt() -> str:
    return """
You are an expert CV parser.
Read the full uploaded CV/Resume document and extract the following fields.
Return only valid JSON:

{
  "name": "string or null",
  "contact": {
    "email": "string or null",
    "phone": "string or null",
    "linkedin": "string or null"
  },
  "qualifications": ["list of education/certification items"],
  "summary": ["write a brief summary"]
}

Rules:
- Return strictly JSON, no markdown and no commentary.
- If a field is not found, return null or an empty list.
- Qualifications should focus on degree, diploma, certification, and formal education.
""".strip()


def extract_cv_fields_from_pdf_bytes(
    pdf_bytes: bytes,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    if not pdf_bytes:
        raise ValueError("Uploaded PDF bytes are empty.")

    client = _get_genai_client()
    response = client.models.generate_content(
        model=model_name,
        contents=[
            _build_cv_prompt(),
            types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf",
            ),
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    parsed = _extract_json_object(response.text or "")
    contact = parsed.get("contact")
    if not isinstance(contact, dict):
        contact = {}

    qualifications = parsed.get("qualifications")
    if not isinstance(qualifications, list):
        qualifications = []

    summary = parsed.get("summary")
    if not isinstance(summary, list):
        summary = []

    return {
        "name": parsed.get("name"),
        "contact": {
            "email": contact.get("email"),
            "phone": contact.get("phone"),
            "linkedin": contact.get("linkedin"),
        },
        "qualifications": qualifications,
        "summary": summary,
    }


def _validate_firebase_storage_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("resumeUrl must be https.")
    if parsed.netloc.lower() != "firebasestorage.googleapis.com":
        raise ValueError("resumeUrl must be a Firebase Storage URL.")
    if "/b/deep-sync-production.firebasestorage.app/o/" not in parsed.path:
        raise ValueError("resumeUrl bucket does not match deep-sync-production.")


def _download_pdf_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=90) as response:
        content = response.read()

    if not content.startswith(b"%PDF-"):
        raise ValueError("Downloaded content is not a valid PDF file.")
    return content


def _get_career_description(career_id: str) -> str:
    career_doc = db.collection("career").document(career_id).get()
    if not career_doc.exists:
        raise ValueError(f"Career document not found: {career_id}")

    description = ((career_doc.to_dict() or {}).get("description") or "").strip()
    if not description:
        raise ValueError(f"Missing description in career/{career_id}")
    return description


def _evaluate_candidate_with_llm(
    description: str,
    extracted_cv: Dict[str, Any],
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    client = _get_genai_client()
    prompt = f"""
You are an expert hiring screener.
Compare this job description with this candidate profile and decide interview eligibility.
Return only valid JSON:
{{
  "decision": "interview or reject",
  "match_score": 0,
  "reason": "short reason",
  "highlights": ["max 5 concise points"]
}}

Rules:
- "decision" must be exactly "interview" or "reject".
- "match_score" must be an integer from 0 to 100.
- Use only evidence from the provided description and candidate data.

Job Description:
{description}

Candidate Data:
{json.dumps(extracted_cv, ensure_ascii=False)}
""".strip()

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    parsed = _extract_json_object(response.text or "")
    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in {"interview", "reject"}:
        decision = "reject"

    try:
        match_score = int(parsed.get("match_score", 0))
    except (TypeError, ValueError):
        match_score = 0
    match_score = max(0, min(100, match_score))

    reason = str(parsed.get("reason", "")).strip()
    highlights = parsed.get("highlights")
    if not isinstance(highlights, list):
        highlights = []

    return {
        "decision": decision,
        "match_score": match_score,
        "reason": reason,
        "highlights": highlights,
    }


@firestore_fn.on_document_created(
    document="career/{careerId}/applications/{applicationId}",
)
def on_application_created(
    event: firestore_fn.Event[firestore_fn.DocumentSnapshot | None],
) -> None:
    snapshot = event.data
    if snapshot is None:
        return

    app_ref = snapshot.reference
    app_data = snapshot.to_dict() or {}

    try:
        resume_url = (app_data.get("resumeUrl") or "").strip()
        if not resume_url:
            raise ValueError("Missing resumeUrl in application document.")

        _validate_firebase_storage_url(resume_url)
        pdf_bytes = _download_pdf_bytes(resume_url)
        extracted = extract_cv_fields_from_pdf_bytes(
            pdf_bytes=pdf_bytes,
            model_name=DEFAULT_MODEL,
        )

        career_id = event.params["careerId"]
        description = _get_career_description(career_id)
        screening = _evaluate_candidate_with_llm(
            description=description,
            extracted_cv=extracted,
            model_name=DEFAULT_MODEL,
        )

        app_ref.update(
            {
                "interviewEligible": screening["decision"] == "interview",
            }
        )
    except Exception as exc:
        app_ref.update(
            {
                "interviewEligible": False,
            }
        )