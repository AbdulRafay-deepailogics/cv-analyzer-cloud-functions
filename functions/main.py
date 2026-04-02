import json
import os
import re
from typing import Any
from urllib.request import Request, urlopen

from dotenv import load_dotenv
load_dotenv()

from firebase_admin import firestore, initialize_app, credentials
from firebase_functions import https_fn  # Changed from firestore_fn
from google import genai
from google.genai import types

PROJECT_ID = "deep-sync-production"
DEFAULT_MODEL = "gemini-2.5-flash-lite" 
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
ELIGIBILITY_THRESHOLD = 50

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
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment variables.")
    return genai.Client(api_key=api_key)

def _download_pdf_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        content = response.read()
    if not content.startswith(b"%PDF-"):
        raise ValueError("Downloaded content is not a valid PDF file.")
    return content

def _get_match_score(pdf_bytes: bytes, requirements: str, model_name: str = DEFAULT_MODEL) -> int:
    client = _get_genai_client()
    
    prompt = f"""
Compare the attached Resume (PDF) against these Job Requirements:
{requirements}
Return only a match score (0-100) based on how well the candidate fits.
Return strictly valid JSON: {{"match_score": <int>}}
""".strip()

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
    
    try:
        data = json.loads(response.text)
        return int(data.get("match_score", 0))
    except:
        match = re.search(r"(\d+)", response.text)
        return int(match.group(1)) if match else 0

@https_fn.on_request()
def on_application_triggered(req: https_fn.Request) -> https_fn.Response:

    career_id = req.args.get("careerId")
    app_id = req.args.get("applicationId")

    if not career_id or not app_id:
        return https_fn.Response("Missing careerId or applicationId", status=400)

    db = get_db()
    
    try:
        app_ref = db.collection("career").document(career_id).collection("applications").document(app_id)
        snapshot = app_ref.get()

        if not snapshot.exists:
            return https_fn.Response("Application not found", status=404)

        app_data = snapshot.to_dict()
        resume_url = app_data.get("resumeUrl", "").strip()

        if not resume_url:
            return https_fn.Response("No resume URL found in document", status=400)

        career_doc = db.collection("career").document(career_id).get()
        job_requirements = career_doc.to_dict().get("requirements", "")

        pdf_bytes = _download_pdf_bytes(resume_url)
        match_score = _get_match_score(pdf_bytes, job_requirements)

        app_ref.update({
            "matchScore": match_score,
            "interviewEligible": match_score >= ELIGIBILITY_THRESHOLD,
            "processedAt": firestore.SERVER_TIMESTAMP
        })

        return https_fn.Response(json.dumps({
            "status": "success",
            "matchScore": match_score,
            "eligible": match_score >= ELIGIBILITY_THRESHOLD
        }), mimetype="application/json")

    except Exception as exc:
        print(f"Error: {exc}")
        return https_fn.Response(f"Processing failed: {str(exc)}", status=500)