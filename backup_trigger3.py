import json
import os
import re
from typing import Any, Dict
from urllib.request import Request, urlopen
from dotenv import load_dotenv
from firebase_admin import firestore, initialize_app, credentials
from firebase_functions import firestore_fn
from google import genai
from google.genai import types

PROJECT_ID = "deep-sync-production"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json" 

_db_client = None

def get_db():
    global _db_client
    if _db_client is None:
        try:
            if os.path.exists(SERVICE_ACCOUNT_PATH):
                cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
                initialize_app(cred, {'projectId': PROJECT_ID})
            else:
                initialize_app()
            _db_client = firestore.client()
        except ValueError:
            _db_client = firestore.client()
        except Exception as e:
            print(f"Failed to initialize Firebase Admin: {e}")
            raise e
    return _db_client

def _get_genai_client() -> genai.Client:
    load_dotenv()
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment variables.")
    return genai.Client(api_key=api_key)

def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    raw_text = (raw_text or "").strip()
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict): return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        raise ValueError("LLM response did not contain a valid JSON object.")
    return json.loads(match.group(0))

def _download_pdf_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        content = response.read()
    if not content.startswith(b"%PDF-"):
        raise ValueError("Downloaded content is not a valid PDF file.")
    return content

def extract_cv_fields_from_pdf_bytes(pdf_bytes: bytes, model_name: str = DEFAULT_MODEL) -> Dict[str, Any]:
    client = _get_genai_client()
    prompt = (
        "You are an expert CV parser. Extract the candidate's name, contact info, "
        "qualifications, and a professional summary. Return strictly valid JSON."
    )
    response = client.models.generate_content(
        model=model_name,
        contents=[
            prompt,
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    return _extract_json_object(response.text or "")

def _evaluate_candidate_against_requirements(
    requirements: str, 
    extracted_cv: Dict[str, Any], 
    model_name: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    client = _get_genai_client()
    prompt = (
        f"Job Requirements:\n{requirements}\n\n"
        f"Candidate Data:\n{json.dumps(extracted_cv)}\n\n"
        "Instructions: Compare the candidate's qualifications against the job requirements. "
        "Provide a match score (0-100) and a decision ('interview' or 'reject'). "
        "Return strictly valid JSON."
    )
    
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
            raise ValueError(f"Career document {career_id} not found")
        
        job_requirements = career_doc.to_dict().get("requirements", "No requirements listed.")

        screening = _evaluate_candidate_against_requirements(job_requirements, extracted)

        app_ref.update({
            "interviewEligible": screening["decision"] == "interview",
            "matchScore": screening["match_score"],
            "extractedData": extracted,
            "processedAt": firestore.SERVER_TIMESTAMP,
            "status": "completed"
        })
        
    except Exception as exc:
        print(f"Error: {exc}")
        app_ref.update({
            "interviewEligible": False, 
            "error": str(exc),
            "status": "failed"
        })