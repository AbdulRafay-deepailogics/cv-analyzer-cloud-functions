import json
import os
import re
import time
from typing import Any
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv()

from firebase_admin import firestore, credentials, initialize_app
from google import genai
from google.genai import types

PROJECT_ID = "deep-sync-production"
DEFAULT_MODEL = "gemini-1.5-flash" 
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
        raise ValueError("GEMINI_API_KEY is missing.")
    return genai.Client(api_key=api_key)

def _download_pdf_bytes(url: str) -> bytes:
    """Downloads PDF into RAM."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()

def _get_match_score(pdf_bytes: bytes, requirements: str) -> int:
    """Sends PDF to Gemini AI and returns score."""
    client = _get_genai_client()
    prompt = f"Compare Resume PDF against: {requirements}. Return JSON: {{'match_score': <int>}}"
    
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
    """Processes only the document passed to it."""
    app_ref = doc_snapshot.reference
    app_data = doc_snapshot.to_dict() or {}

    if "matchScore" in app_data:
        return

    try:
        resume_url = app_data.get("resumeUrl", "").strip()
        if not resume_url: return
        
        # Get Career ID from path
        career_id = app_ref.path.split('/')[1]
        career_doc = db.collection("career").document(career_id).get()
        
        if not career_doc.exists: return
        requirements = career_doc.to_dict().get("requirements", "")
        
        print(f"Processing NEW application: {doc_snapshot.id}")
        
        # Calling the helpers defined above
        pdf_bytes = _download_pdf_bytes(resume_url)
        score = _get_match_score(pdf_bytes, requirements)

        app_ref.update({
            "matchScore": score,
            "interviewEligible": score >= ELIGIBILITY_THRESHOLD,
        })
        print(f"Success! Score: {score}")
        
    except Exception as e:
        print(f"Error: {e}")

def start_listener():
    print("Listening for NEW applications only...")
    
    query = db.collection_group("applications")
    
    is_initial_load = True 

    def on_snapshot(col_snapshot, changes, read_time):
        nonlocal is_initial_load
        
        if is_initial_load:
            is_initial_load = False
            print("Ignoring existing database entries. Waiting for new applicants...")
            return 

        for change in changes:
            if change.type.name == 'ADDED':
                process_document(change.document)

    query.on_snapshot(on_snapshot)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    start_listener()