import json
import os
import re
from typing import Any, Dict
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from firebase_admin import firestore, initialize_app, credentials
from firebase_functions import firestore_fn
from google import genai
from google.genai import types

PROJECT_ID = "deep-sync-production"
DEFAULT_MODEL = "gemini-2.5-flash-lite"

_db = None

def get_db():
    global _db
    if _db is None:
        try:
            initialize_app()
        except ValueError:
            pass 
        _db = firestore.client()
    return _db

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
    
    return {
        "name": parsed.get("name"),
        "contact": parsed.get("contact") if isinstance(parsed.get("contact"), dict) else {},
        "qualifications": parsed.get("qualifications") if isinstance(parsed.get("qualifications"), list) else [],
        "summary": parsed.get("summary") if isinstance(parsed.get("summary"), list) else [],
    }

def _download_pdf_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        content = response.read()
    if not content.startswith(b"%PDF-"):
        raise ValueError("Downloaded content is not a valid PDF file.")
    return content

def _evaluate_candidate_with_llm(
    description: str,
    extracted_cv: Dict[str, Any],
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    client = _get_genai_client()
    prompt = f"Job Description:\n{description}\n\nCandidate Data:\n{json.dumps(extracted_cv)}"
    
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    parsed = _extract_json_object(response.text or "")
    return {
        "decision": str(parsed.get("decision", "reject")).lower(),
        "match_score": int(parsed.get("match_score", 0)),
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

    db = get_db()
    app_ref = snapshot.reference
    app_data = snapshot.to_dict() or {}

    try:
        resume_url = app_data.get("resumeUrl", "").strip()
        if not resume_url:
            return

        pdf_bytes = _download_pdf_bytes(resume_url)
        extracted = extract_cv_fields_from_pdf_bytes(pdf_bytes)

        career_id = event.params["careerId"]
        career_doc = db.collection("career").document(career_id).get()
        
        if not career_doc.exists:
            raise ValueError("Career not found")

        description = career_doc.to_dict().get("description", "")
        screening = _evaluate_candidate_with_llm(description, extracted)

        app_ref.update({
            "interviewEligible": screening["decision"] == "interview",
            "extractedData": extracted # Useful for debugging
        })
        
    except Exception as exc:
        print(f"Error processing application: {exc}")
        app_ref.update({"interviewEligible": False})