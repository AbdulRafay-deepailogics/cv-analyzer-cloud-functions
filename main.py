import json
import os
import re
import time
from typing import Any
from urllib.request import Request, urlopen
from dotenv import load_dotenv

# Load local .env variables
load_dotenv()

from firebase_admin import firestore, credentials, initialize_app
from google import genai
from google.genai import types

PROJECT_ID = "deep-sync-production"
DEFAULT_MODEL = "gemini-2.5-flash-lite" 
SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
ELIGIBILITY_THRESHOLD = 50

if os.path.exists(SERVICE_ACCOUNT_PATH):
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    initialize_app(cred, {'projectId': PROJECT_ID})
else:
    initialize_app()

db = firestore.client()

def _get_genai_client() -> genai.Client:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from environment.")
    return genai.Client(api_key=api_key)

def _download_pdf_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        content = response.read()
    if not content.startswith(b"%PDF-"):
        raise ValueError("Downloaded content is not a valid PDF file.")
    return content

def _get_match_score(pdf_bytes: bytes, requirements: str) -> int:
    client = _get_genai_client()
    prompt = f"""
Compare the Resume (PDF) against these Job Requirements:
{requirements}
Return only a match score (0-100).
Return strictly valid JSON: {{"match_score": <int>}}
""".strip()

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
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

def process_document(doc_snapshot):
    """The core logic to process a single application document."""
    app_ref = doc_snapshot.reference
    app_data = doc_snapshot.to_dict() or {}

    # Critical: Prevent infinite loops by checking if already processed
    if "matchScore" in app_data:
        return

    try:
        resume_url = app_data.get("resumeUrl", "").strip()
        if not resume_url:
            return
        
        # Extract career_id from the document path: career/{careerId}/applications/{appId}
        path_segments = app_ref.path.split('/')
        career_id = path_segments[1]
        
        career_doc = db.collection("career").document(career_id).get()
        if not career_doc.exists:
            print(f"Career doc {career_id} not found.")
            return
        
        job_requirements = career_doc.to_dict().get("requirements", "")
        
        print(f"Processing application {doc_snapshot.id} for career {career_id}...")
        pdf_bytes = _download_pdf_bytes(resume_url)
        match_score = _get_match_score(pdf_bytes, job_requirements)

        app_ref.update({
            "matchScore": match_score,
            "interviewEligible": match_score >= ELIGIBILITY_THRESHOLD,
        })
        print(f"Successfully updated {doc_snapshot.id} with score: {match_score}")
        
    except Exception as exc:
        print(f"Error processing {doc_snapshot.id}: {exc}")
        app_ref.update({
            "interviewEligible": False, 
            "matchScore": 0,
        })

def start_listener():
    """Starts a persistent listener on all 'applications' sub-collections."""
    print("Listening for new applications in Firestore...")
    
    # Collection group watches every 'applications' collection regardless of parent
    query = db.collection_group("applications")
    
    def on_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            # We process 'ADDED' for new docs or 'MODIFIED' if you want to re-run on updates
            if change.type.name == 'ADDED':
                process_document(change.document)

    # Watch the query
    query.on_snapshot(on_snapshot)

    # Keep the main thread alive for the container
    while True:
        time.sleep(1)

if __name__ == "__main__":
    start_listener()