import os
import re
import io
import base64
import json
import time
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse

# --- File Parsing Imports ---
import pdfplumber
from docx import Document

# --- AI Imports ---
import spacy
import google.generativeai as genai

# --- Google API Imports ---
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.message import EmailMessage # For sending email

# --- GitHub/LinkedIn API Imports ---
import requests
from thefuzz import fuzz

# --- Supabase Import ---
from supabase import create_client, Client

# -----------------------------------------------------------------
# --- PART 1: CONFIGURATIONS
# -----------------------------------------------------------------

# --- Supabase Config ---
SUPABASE_URL = "https://bigehkexbbkdogwaehzs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJpZ2Voa2V4YmJrZG9nd2FlaHpzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTc0MTkyNiwiZXhwIjoyMDc3MzE3OTI2fQ.tKBKF81QeyQWwG9kSazoF6-1F-r0Zy7uAMCOozrxYmY"
supabase: Client = None
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Supabase client: {e}")

# --- Gemini API Config ---
try:
    API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBNt6C9WSGQ7TSZLd0tywYwR1c4618REvk")
    if API_KEY == "AIzaSyBNt6C9WSGQ7TSZLd0tywYwR1c4618REvk": print("Warning: Using fallback Gemini Key.")
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

# --- GitHub API Config ---
GITHUB_API_TOKEN = "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" 
if not GITHUB_API_TOKEN or "YOUR_GITHUB_API_TOKEN_HERE" in GITHUB_API_TOKEN:
    print("Warning: GitHub API Token not found or not set. GitHub verification may fail.")
    GITHUB_HEADERS = {}
else:
    print("GitHub API Token loaded.")
    GITHUB_HEADERS = {"Authorization": f"token {GITHUB_API_TOKEN}"}
GITHUB_BASE_URL = "https://api.github.com"

# --- Hugging Face AI Detector Config ---
HF_API_TOKEN = "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"
HF_API_ENDPOINT = "https://api-inference.huggingface.co/models/Hello-SimpleAI/chatgpt-detector-roberta"
if not HF_API_TOKEN or HF_API_TOKEN.startswith("hf_"):
    print("Hugging Face API Token loaded.")
else:
    print("Warning: Hugging Face API Token seems invalid or is missing.")


# --- Google API Config ---
# --- MODIFIED: Added .modify scope to mark emails as read ---
# --- REMEMBER TO DELETE YOUR token.json FILE AND RE-AUTHENTICATE ---
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
RESUME_DIR = './resumes'

# --- LinkedIn/Certificate Config ---
FUZZY_MATCH_THRESHOLD = 85

# -----------------------------------------------------------------
# --- PART 2: LOAD SPACY MODEL
# -----------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: Spacy 'en_core_web_sm' model not found. Please run:")
    print("\n    python -m spacy download en_core_web_sm\n")
    exit()

# -----------------------------------------------------------------
# --- PART 3: ANALYSIS & VERIFICATION FUNCTIONS
# -----------------------------------------------------------------

# --- MODIFIED: Added marketing and content skills ---
SKILL_KEYWORDS = [
    # Tech
    'python', 'javascript', 'java', 'c#', 'c++', 'sql', 'nosql', 'react', 'angular',
    'vue', 'node.js', 'django', 'flask', 'spring', 'mongodb', 'postgresql', 'mysql',
    'redis', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'ci/cd',
    'machine learning', 'data analysis', 'tensorflow', 'pytorch',
    # Marketing
    'seo', 'sem', 'google analytics', 'google ads', 'hubspot', 'salesforce', 
    'content strategy', 'email marketing', 'social media marketing',
    # Content
    'copywriting', 'wordpress', 'content marketing', 'blogging'
]

# --- MODIFIED: Added marketing and content roles ---
ROLE_KEYWORDS = {
    'Backend Developer': ['django', 'flask', 'node.js', 'spring', 'sql', 'nosql', 'java'],
    'Frontend Developer': ['react', 'angular', 'vue', 'javascript', 'css', 'html'],
    'Full-Stack Developer': ['react', 'node.js', 'python', 'sql', 'javascript'],
    'DevOps Engineer': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'git'],
    'Data Scientist': ['machine learning', 'python', 'tensorflow', 'pytorch', 'data analysis'],
    'Digital Marketer': ['seo', 'sem', 'google analytics', 'google ads', 'content strategy', 'content marketing', 'email marketing'],
    'Content Writer': ['copywriting', 'seo', 'wordpress', 'content strategy', 'content marketing', 'blogging']
}

def extract_text_from_file_data(filename, file_data):
    """ (From Script 1) Parses PDF (pdfplumber) and DOCX (python-docx) """
    text = ""
    file_like_object = io.BytesIO(file_data)
    try:
        if filename.lower().endswith('.pdf'):
            with pdfplumber.open(file_like_object) as pdf:
                for page in pdf.pages: text += page.extract_text(x_tolerance=1, y_tolerance=1) or ""
        elif filename.lower().endswith('.docx'):
            doc = Document(file_like_object)
            for para in doc.paragraphs: text += para.text + "\n"
        print(f"  -> Successfully extracted text from '{filename}'.")
        return text.strip()
    except Exception as e: print(f"  -> Error reading file {filename}: {e}"); return None

def check_ai_generation(text_to_check):
    """ (From Script 2) Sends text to the Hugging Face Inference API. """
    print("\n[AI Detector] Checking text for AI generation (Hugging Face)...")
    
    max_length = 510 # Model limit
    truncated_text = text_to_check
    if len(text_to_check) > max_length:
        truncated_text = text_to_check[:max_length]
        print(f"[AI Detector] Warning: Text truncated to {max_length} chars for this model.")
    
    if not truncated_text.strip():
        print("[AI Detector] Text is empty, skipping check.")
        return {"status": "error", "message": "Input text was empty."}

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }
    
    payload = json.dumps({
        "inputs": truncated_text
    })

    try:
        response = requests.post(HF_API_ENDPOINT, headers=headers, data=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            if result and isinstance(result, list) and isinstance(result[0], list):
                scores = result[0]
                ai_score = 0.0
                for item in scores:
                    if item.get('label') == 'LABEL_1': # LABEL_1 is AI-generated
                        ai_score = item.get('score', 0)
                        break
                print("[AI Detector] Check successful.")
                return {"ai_probability": ai_score, "status": "success"}
            else:
                print(f"[AI Detector] API returned unexpected structure: {result}")
                return {"status": "error", "message": "Unexpected API response structure"}

        elif response.status_code == 401:
            print("[AI Detector] FAILED: Invalid API Key (401). Check HF_API_TOKEN.")
            return {"status": "error", "message": "Invalid API Key"}
        elif response.status_code == 503:
            print("[AI Detector] FAILED: Model is loading (503). Retrying in 15 seconds...")
            time.sleep(15) 
            return check_ai_generation(text_to_check) # Retry
        else:
            print(f"[AI Detector] FAILED: Error {response.status_code}")
            print(f"Response: {response.text}")
            return {"status": "error", "message": f"API Error {response.status_code}"}

    except requests.exceptions.Timeout:
        print("[AI Detector] FAILED: Request timed out.")
        return {"status": "error", "message": "Request timed out"}
    except requests.exceptions.RequestException as e:
        print(f"[AI Detector] FAILED: Network error. {e}")
        return {"status": "error", "message": f"Network Error: {e}"}

def extract_urls(text):
    urls = {'linkedin': None, 'github': None}
    linkedin_match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_-]+)/?', text, re.IGNORECASE)
    github_match = re.search(r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)/?', text, re.IGNORECASE)
    if linkedin_match: urls['linkedin'] = f"https://linkedin.com/in/{linkedin_match.group(1)}"
    if github_match: urls['github'] = f"https://github.com/{github_match.group(1)}"
    return urls

def extract_certificates(text):
    """
    Extracts certificate claims (URLs and text) from resume text.
    """
    claims = []
    
    try:
        urls = re.findall(r'https?://[^\s]+', text, re.IGNORECASE)
        cert_urls = [url for url in urls if 'credly.com' in url or 'coursera.org' in url or 'udemy.com' in url]
        claims.extend(cert_urls)
    except Exception as e:
        print(f"  [Warn] Error parsing certificate URLs: {e}")

    try:
        section_match = re.search(r'(?i)^\s*(certifications?|licenses?|badges|credentials)\s*[:\n-]', text, re.MULTILINE)
        if section_match:
            start_index = section_match.end()
            end_match = re.search(r'(?i)(^\s*([A-Z][a-z]+(\s[A-Z][a-z]+)*|EXPERIENCE|EDUCATION|SKILLS)\s*[:\n-])|(\n\s*\n)', text[start_index:], re.MULTILINE)
            end_index = end_match.start() + start_index if end_match else len(text)
            section_text = text[start_index:end_index]
            text_claims = [line.strip() for line in section_text.split('\n') if line.strip() and len(line.strip()) > 4 and 'http' not in line]
            claims.extend(text_claims)
    except Exception as e:
        print(f"  [Warn] Could not parse text-based certificates: {e}")
        
    return list(set(claims)) # Return unique claims

def get_gemini_description(resume_text, candidate_name):
    max_len = 15000; truncated_text = resume_text[:max_len] + ("..." if len(resume_text) > max_len else "")
    prompt = f"You are an expert HR recruitment analyst. Based *only* on the following resume text, write a 2-3 sentence professional description for the candidate, {candidate_name}. Speak in the third person. Focus on their primary role, years of experience (if mentioned), and strongest technologies.\nResume:\n---\n{truncated_text}\n---"
    try: response = gemini_model.generate_content(prompt); return response.text.strip()
    except Exception as e: print(f"\nWarning: Gemini API call failed. Fallback used. Error: {e}"); return f"Resume analysis for {candidate_name}."

def extract_education_experience(resume_text):
    """
    Uses Gemini to extract structured education and experience data.
    """
    print("\n[Gemini] Extracting Education & Experience...")
    max_len = 15000
    truncated_text = resume_text[:max_len] + ("..." if len(resume_text) > max_len else "")

    prompt = f"""
    Based *only* on the following resume text, extract the candidate's education and work experience.
    Respond *only* with a single, minified JSON object with the keys "education" and "experience".
    
    Example format:
    {{"education": [{{"institution": "University of Example", "degree": "B.S. in Computer Science", "date": "2018 - 2022"}}],"experience": [{{"company": "Example Inc.", "role": "Software Engineer", "date": "2022 - Present"}}]}}

    If no education or experience is found, return an empty list [] for that key.

    Resume Text:
    ---
    {truncated_text}
    ---
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(json_text)
        if 'education' not in data: data['education'] = []
        if 'experience' not in data: data['experience'] = []
        print("  -> Successfully extracted education and experience.")
        return data
    except Exception as e:
        print(f"  -> Warning: Failed to extract education/experience via Gemini. Error: {e}")
        return {"education": [], "experience": []}

def get_credly_data_from_id(badge_id_or_url):
    """
    Mock Credly API function.
    """
    print(f"[MOCK Credly API] Request for badge: {badge_id_or_url}")
    try:
        with open('mock_credly_data.json', 'r', encoding='utf-8') as f:
            all_badges_data = json.load(f) 
        
        if not isinstance(all_badges_data, list):
            print("[MOCK Credly API] ⚠️ Invalid JSON format: Expected a LIST of badge objects.")
            return None

        for badge_data in all_badges_data:
            if badge_id_or_url in badge_data.get("badge_id", "") or badge_id_or_url in badge_data.get("evidence_url", ""):
                print("[MOCK Credly API] ✅ Found matching badge in mock data.")
                return badge_data 
        
        print("[MOCK Credly API] ❌ Badge not found in mock data.")
        return None

    except FileNotFoundError:
        print("[MOCK Credly API] ⚠️ File 'mock_credly_data.json' not found. Cannot verify Credly badges.")
        return None
    except json.JSONDecodeError:
        print("[MOCK Credly API] ⚠️ Invalid JSON format in mock file.")
        return None
    except Exception as e:
        print(f"[MOCK Credly API] ⚠️ Unexpected error: {e}")
        return None

def verify_certificates(resume_data):
    """
    Verifies certificate claims from a parsed resume.
    """
    verification_results = []
    name_on_resume = resume_data.get('name', '')
    all_claims_from_resume = resume_data.get('certificates', [])

    for claim in all_claims_from_resume:
        result = {
            "claim": claim, "status": "Unverified",
            "method": "Unknown", "message": "No verification method applied."
        }

        if "credly.com/badges/" in claim:
            result["method"] = "Credly API (Mock)"
            badge_data = get_credly_data_from_id(claim)
            if badge_data:
                name_on_badge = badge_data.get('recipient_name', '')
                title_on_badge = badge_data.get('badge_template', {}).get('name', '')
                name_score = fuzz.token_set_ratio(name_on_resume, name_on_badge)
                best_title_score = 0
                for text_claim in all_claims_from_resume:
                    if not ("http" in text_claim or "/" in text_claim):
                        score = fuzz.partial_ratio(text_claim.lower(), title_on_badge.lower())
                        best_title_score = max(best_title_score, score)

                if name_score > FUZZY_MATCH_THRESHOLD and best_title_score > 75:
                    result["status"] = "Verified"
                    result["message"] = f"✅ Verified via Credly: Name '{name_on_badge}', Title '{title_on_badge}'."
                elif name_score <= FUZZY_MATCH_THRESHOLD:
                    result["status"] = "Discrepancy"
                    result["message"] = f"⚠️ Name mismatch: Resume='{name_on_resume}' vs Badge='{name_on_badge}'."
                else:
                    result["status"] = "Discrepancy"
                    result["message"] = f"⚠️ Title mismatch: Badge title '{title_on_badge}' not found (Best score: {best_title_score})."
            else:
                result["status"] = "Not Found"
                result["message"] = "❌ Badge not found in mock Credly data."

        elif "coursera.org/verify" in claim:
            result["method"] = "URL Scraping (Placeholder)"; result["status"] = "Pending"; result["message"] = "⚙️ Coursera logic not implemented."
        elif "udemy.com/certificate" in claim:
            result["method"] = "URL Scraping (Placeholder)"; result["status"] = "Pending"; result["message"] = "⚙️ Udemy logic not implemented."
        elif not ("http" in claim or "/" in claim or "." in claim):
            already_verified = any(
                r['status'] == 'Verified' and fuzz.partial_ratio(claim.lower(), r['message'].lower()) > 75 
                for r in verification_results
            )
            if not already_verified:
                result["method"] = "Manual Check"; result["status"] = "Unverified"; result["message"] = "📋 Text-only claim."
            else:
                continue 
        else:
            result["method"] = "Unknown URL"; result["status"] = "Unverified"; result["message"] = "🔍 Unrecognized URL format."

        verification_results.append(result)
    return verification_results

def analyze_resume(resume_text):
    """ Main analysis pipeline for a resume text """
    doc = nlp(resume_text); name = "Candidate"
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON'];
    if persons: name = persons[0]
    
    ai_check_result = check_ai_generation(resume_text)
    urls = extract_urls(resume_text); resume_text_lower = resume_text.lower()
    found_skills = list(set([skill for skill in SKILL_KEYWORDS if re.search(r'\b' + re.escape(skill) + r'\b', resume_text_lower)]))
    
    print("\nGenerating AI description..."); 
    gemini_description = get_gemini_description(resume_text, name); 
    print("...Description generated.")

    certificate_claims = extract_certificates(resume_text)
    resume_data_for_certs = {'name': name.strip(), 'certificates': certificate_claims}
    print("\nVerifying certificates...")
    certificate_verification_report = verify_certificates(resume_data_for_certs)
    print("...Certificate verification complete.")
    
    extracted_data = extract_education_experience(resume_text)

    report = {
        'name': name.strip(),
        'linkedin_url': urls.get('linkedin'),
        'github_url': urls.get('github'),
        'description': gemini_description,
        'ai_detection': ai_check_result,
        'certificate_verification': certificate_verification_report,
        'expertise': found_skills,
        'education': extracted_data.get('education', []),
        'experience': extracted_data.get('experience', [])
    }
    
    skill_counts = Counter(re.findall(r'\b(' + '|'.join(SKILL_KEYWORDS) + r')\b', resume_text_lower))
    report['strong_fields'] = [skill[0] for skill in skill_counts.most_common(3)]
    role_scores = Counter({role: sum(1 for skill in report['expertise'] if skill in keywords) for role, keywords in ROLE_KEYWORDS.items() if sum(1 for skill in report['expertise'] if skill in keywords) > 0})
    report['suitable_role'] = role_scores.most_common(1)[0][0] if role_scores else 'Not determined'
    
    if report['suitable_role'] != 'Not determined': 
        target_skills = ROLE_KEYWORDS[report['suitable_role']]
        gaps = [skill for skill in target_skills if skill not in report['expertise']]
        report['potential_gaps'] = gaps if gaps else ["None."]
    else:
        report['potential_gaps'] = []
        
    return report

def get_username_from_link(link_or_username):
    if not link_or_username: return None
    if link_or_username.startswith("http"):
        try: parts = urlparse(link_or_username).path.strip().split('/'); username = parts[1] if len(parts) > 1 and parts[1] else None
        except Exception as e: print(f"  [Error] Invalid GitHub link ('{link_or_username}'): {e}"); return None
        if username: print(f"  [Info] Extracted GitHub username '{username}'."); return username
        else: print(f"  [Error] Could not parse username from link path: '{link_or_username}'"); return None
    if '/' not in link_or_username and '.' not in link_or_username: print(f"  [Info] Assuming '{link_or_username}' is GitHub username."); return link_or_username
    else: print(f"  [Error] Invalid format for GitHub: '{link_or_username}'"); return None

def calculate_github_profile_rating(report):
    score, reasons = 50, []; collaboration_check = report.get("collaboration_check", {}); originality_check = report.get("originality_check", {}); consistency_check = report.get("consistency_check", {})
    if collaboration_check.get("is_collaborator"): score += 30; reasons.append("+30: Collaborator")
    if originality_check.get("red_flag_empty_repos"): score -= 25; reasons.append("-25: Empty Profile")
    elif originality_check.get("total_code_bytes", 0) > 50000: score += 10; reasons.append("+10: Coder")
    if consistency_check.get("red_flag_bunched_activity"): score -= 30; reasons.append("-30: Bunched Activity")
    elif consistency_check.get("activity_spread_days", 0) > 30: score += 20; reasons.append("+20: Consistent")
    elif consistency_check.get("activity_spread_days", 0) > 7: score += 10; reasons.append("+10: Active")
    if consistency_check.get("recent_activity_count", 0) < 5: score -= 10; reasons.append("-10: Inactive")
    if score >= 75: rating = "Excellent"
    elif score >= 45: rating = "Good"
    else: rating = "Bad"
    return {"score": max(0, min(score, 100)), "rating": rating, "reasoning": reasons}

def verify_github_profile(username):
    if not GITHUB_API_TOKEN or "YOUR_GITHUB_API_TOKEN_HERE" in GITHUB_API_TOKEN:
        print("  [Error] GitHub API Token not configured for verification.")
        return {"error": "GitHub API Token not configured."}
    print(f"\nStarting GitHub verification for: {username}"); repos_url, events_url, search_pr_url = f"{GITHUB_BASE_URL}/users/{username}/repos", f"{GITHUB_BASE_URL}/users/{username}/events?per_page=100", f"{GITHUB_BASE_URL}/search/issues?q=author:{username}+type:pr+is:public"
    repos_data, events_data, public_pr_count = [], [], 0; repo_response = None
    try: repo_response = requests.get(repos_url, headers=GITHUB_HEADERS, timeout=15); repo_response.raise_for_status(); repos_data = repo_response.json()
    except requests.exceptions.Timeout: print(f"  [Error] Timeout fetching repos for {username}."); return {"error": "Timeout fetching user repositories."}
    except requests.exceptions.RequestException as e:
        status_code = repo_response.status_code if repo_response else None; print(f"  [Error] Could not fetch repos: {e}" + (f" (Status: {status_code})" if status_code else ""));
        if status_code == 403: return {"error": f"Failed repo fetch: GitHub API rate limit likely exceeded (403). Check token or wait."}
        if status_code == 401: return {"error": f"Failed repo fetch: Invalid GitHub credentials (401). Check token."}
        return {"error": f"Failed repo fetch: {e}"}
    try: events_response = requests.get(events_url, headers=GITHUB_HEADERS, timeout=15); events_response.raise_for_status(); events_data = events_response.json()
    except requests.exceptions.Timeout: print(f"  [Error] Timeout fetching events."); events_data = []
    except requests.exceptions.RequestException as e: print(f"  [Error] Could not fetch events: {e}"); events_data = []
    try: pr_response = requests.get(search_pr_url, headers=GITHUB_HEADERS, timeout=15); pr_response.raise_for_status(); public_pr_count = pr_response.json().get('total_count', 0)
    except requests.exceptions.Timeout: print(f"  [Error] Timeout fetching PRs.")
    except requests.exceptions.RequestException as e: print(f"  [Error] Could not fetch PRs: {e}")
    original_repos, lang_stats, total_bytes = [], {}, 0; print(f"  - Found {len(repos_data)} repos. Analyzing...")
    for repo in repos_data:
        if not repo['fork']:
            original_repos.append(repo['name'])
            try: lang_resp = requests.get(repo['languages_url'], headers=GITHUB_HEADERS, timeout=10); lang_resp.raise_for_status(); languages = lang_resp.json()
            except requests.exceptions.RequestException: continue
            if languages:
                for lang, b in languages.items():
                    if isinstance(b, int): total_bytes += b; lang_stats[lang] = lang_stats.get(lang, 0) + b
    timestamps = [e['created_at'] for e in events_data if e.get('type') in ['PushEvent', 'PullRequestEvent', 'CreateEvent', 'IssuesEvent'] and e.get('created_at')]
    spread, bunched = 0, False
    if len(timestamps) > 1:
        try: timestamps.sort(reverse=True); last = datetime.strptime(timestamps[0], '%Y-%m-%dT%H:%M:%SZ'); first = datetime.strptime(timestamps[-1], '%Y-%m-%dT%H:%M:%SZ'); spread = (last - first).days; bunched = spread < 2 and len(timestamps) > 10
        except ValueError as dt_error: print(f"  [Warn] Could not parse event timestamps: {dt_error}")
    lang_perc = {lang: round((b / total_bytes) * 100, 2) for lang, b in lang_stats.items()} if total_bytes > 0 else {}
    report = {"username": username, "verification_status": "Completed", "originality_check": {"original_repo_count": len(original_repos), "primary_languages": sorted(lang_perc.items(), key=lambda i: i[1], reverse=True)[:5], "total_code_bytes": total_bytes, "red_flag_empty_repos": total_bytes == 0 and len(original_repos) > 0}, "consistency_check": {"recent_activity_count": len(timestamps), "activity_spread_days": spread, "red_flag_bunched_activity": bunched}, "collaboration_check": {"public_pull_requests_made": public_pr_count, "is_collaborator": public_pr_count > 0}}; print("-> Generating final rating..."); report["profile_rating"] = calculate_github_profile_rating(report); return report

def get_linkedin_data_from_url(linkedin_url):
    print(f"\n[MOCK LinkedIn API] Reading from 'my_profile_data.json'...");
    try:
        with open('my_profile_data.json', 'r', encoding='utf-8') as f: data = json.load(f)
        if isinstance(data, list) and data: return data[0]
        if isinstance(data, dict): return data
        print("[MOCK] FAILED: JSON empty/wrong format."); return None
    except FileNotFoundError: print("[MOCK] FAILED: 'my_profile_data.json' not found!"); return None
    except json.JSONDecodeError: print("[MOCK] FAILED: Could not read JSON."); return None
    except Exception as e: print(f"[MOCK] FAILED: Error: {e}"); return None

def analyze_linkedin_post_activity(posts):
    if not posts: return {"status": "Not Genuine", "message": "No post activity.", "score_boost": -10}
    orig, likes, total = 0, 0, len(posts)
    for post in posts:
        if post.get('type') == 'PERSONAL_POST': orig += 1
        likes_count = post.get('likesCount', 0);
        if isinstance(likes_count, int): likes += likes_count
    ratio = (orig / total) if total > 0 else 0; msg = f"Found {total} posts. "; boost = 10; msg += f"Originality {int(ratio*100)}%. " + ("High." if ratio >= 0.5 else "Low.");
    if ratio >= 0.5: boost += 10
    if likes > 100: msg += " High engagement."; boost += 10
    return {"status": "Genuine", "message": msg, "score_boost": boost}

def verify_linkedin_profile(resume_data, linkedin_data):
    print("-> Starting LinkedIn verification..."); report = {"overall_score": 0, "name_match": {"status": False, "message": ""}, "plausibility": {"message": ""}, "activity_audit": {}, "job_history_matches": [], "skill_matches": [], "discrepancies": []}
    name_score = fuzz.token_set_ratio(resume_data.get('name',''), linkedin_data.get('fullName', ''));
    if name_score > FUZZY_MATCH_THRESHOLD: report['name_match']['status']=True; report['name_match']['message']=f"Name matches '{linkedin_data.get('fullName')}'"; report['overall_score']+=20
    else: report['name_match']['message']=f"Name mismatch: Res '{resume_data.get('name','')}', LI '{linkedin_data.get('fullName')}'"
    if linkedin_data.get('verified', False): report['overall_score']+=30; report['plausibility']['message']="Verified Profile. "
    conn = linkedin_data.get('connectionCount', 0); msg_conn = f"{conn} connections. " if conn > 100 else f"Small network ({conn}). "; report['plausibility']['message']+=msg_conn;
    if conn > 100: report['overall_score'] += 10
    recs = linkedin_data.get('recommendationCount', 0);
    if recs > 0: report['overall_score']+=10; report['plausibility']['message']+=f"{recs} recs."
    
    resume_experience = resume_data.get('experience', [])
    if not resume_experience:
        print("  [Warn] No structured experience extracted from resume for LI matching.")
    
    for res_job in resume_experience:
        matched = False
        for li_job in linkedin_data.get('experience', []):
            comp_score = fuzz.ratio(res_job.get('company',''), li_job.get('companyName', '')); title_score = fuzz.token_set_ratio(res_job.get('role',''), li_job.get('title', ''))
            if comp_score > FUZZY_MATCH_THRESHOLD and title_score > FUZZY_MATCH_THRESHOLD: 
                report['job_history_matches'].append(f"Verified: {res_job.get('role','')}@{res_job.get('company','')}"); 
                report['overall_score']+=15; matched=True; break
            elif comp_score > FUZZY_MATCH_THRESHOLD: 
                disc_msg = f"Title Inflation?: Res '{res_job.get('role','')}' vs LI '{li_job.get('title', 'N/A')}'" if title_score > 50 else f"Title Mismatch: Res '{res_job.get('role','')}' vs LI '{li_job.get('title', 'N/A')}'"
                report['discrepancies'].append(f"{disc_msg} at '{li_job.get('companyName', 'N/A')}'"); 
                report['overall_score']-=15; matched=True; break
        if not matched: report['discrepancies'].append(f"Unverified Job: '{res_job.get('role','')} at {res_job.get('company','')}'")

    li_skills = [s.get('name', '').lower() for s in linkedin_data.get('skills', [])]
    for res_skill in resume_data.get('expertise', []):
        found = any(fuzz.partial_ratio(res_skill.lower(), li_skill) > 90 for li_skill in li_skills)
        if found: report['skill_matches'].append(f"Verified Skill: '{res_skill}'"); report['overall_score']+=2
        else: report['discrepancies'].append(f"Unverified Skill: '{res_skill}'")
    post_report = analyze_linkedin_post_activity(linkedin_data.get('posts', [])); report['activity_audit'] = post_report; report['overall_score'] += post_report['score_boost']; report['overall_score'] = max(0, min(100, report['overall_score'])); print("-> LinkedIn verification complete."); return report

# -----------------------------------------------------------------
# --- PART 6: PRINTING, SCORING & MAIN GMAIL FUNCTION
# -----------------------------------------------------------------
def calculate_final_score(github_report, linkedin_report):
    """ Calculates a weighted final score (simple average for now). """
    gh_score, li_score = -1, -1
    if github_report and not github_report.get('error'): gh_score = github_report.get('profile_rating', {}).get('score', -1)
    if linkedin_report and not linkedin_report.get('error'): li_score = linkedin_report.get('overall_score', -1)
    scores = [s for s in [gh_score, li_score] if s != -1]
    if not scores: return 0
    return int(round(sum(scores) / len(scores)))

def print_combined_report(report):
    """ MODIFIED: Prints the combined analysis report, including AI check and Certs. """
    print("\n--- COMBINED CANDIDATE REPORT ---")
    print("==============================\n")
    print(f"**Candidate Name:** {report.get('name', 'N/A')}")
    print(f"**LinkedIn URL:** {'Yes' if report.get('linkedin_url') else 'No'}")
    print(f"**GitHub URL:** {'Yes' if report.get('github_url') else 'No'}")
    
    print("\n## AI-Generated Description"); print(report.get('description', 'N/A'))
    
    print("\n## AI-Generated Text Check"); ai_check = report.get('ai_detection')
    if ai_check:
        if ai_check.get('status') == 'success':
            prob = ai_check.get('ai_probability', 0)
            prob_percent = prob * 100
            status = "🔴 Likely AI" if prob > 0.75 else ("🟡 Possibly AI" if prob > 0.5 else "🟢 Likely Human")
            print(f" **Status:** {status} ({prob_percent:.2f}% AI probability)")
        else:
            print(f" Error: {ai_check.get('message', 'Check failed.')}")
    else:
        print(" Not performed.")
    
    print("\n## Core Expertise (Skills Identified)"); print(", ".join(report.get('expertise', [])) if report.get('expertise') else "N/A")
    print("\n## Suggested Role"); print(report.get('suitable_role', 'N/A'))
    
    print("\n## 🎓 Education (from AI)");
    if report.get('education'):
        for edu in report['education']:
            print(f" - {edu.get('degree', 'N/A')} at {edu.get('institution', 'N/A')} ({edu.get('date', 'N/A')})")
    else: print(" N/A")
    
    print("\n## 💼 Experience (from AI)");
    if report.get('experience'):
        for exp in report['experience']:
            print(f" - {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')} ({exp.get('date', 'N/A')})")
    else: print(" N/A")

    print("\n## 📜 Certificate Verification"); cert_report = report.get('certificate_verification')
    if cert_report:
        if not cert_report:
            print(" No certificate claims found.")
        for item in cert_report:
            claim_display = item['claim']
            if len(claim_display) > 60:
                claim_display = claim_display[:57] + "..."
            
            status_icon = "❓"
            if item['status'] == 'Verified': status_icon = '✅'
            elif item['status'] == 'Discrepancy': status_icon = '⚠️'
            elif item['status'] == 'Not Found': status_icon = '❌'
            elif item['status'] == 'Pending': status_icon = '⚙️'
            elif item['status'] == 'Unverified': status_icon = '📋'
            
            print(f" {status_icon} **{item['status']}**: {claim_display}\n    (Message: {item['message']})")
    else:
        print(" No certificate claims found or verification not performed.")
    
    print("\n## GitHub Verification"); gh = report.get('github_verification')
    if gh:
        if gh.get('error'): print(f" Error: {gh['error']}")
        elif gh.get('verification_status') == "Completed": r = gh.get('profile_rating', {}); print(f" **Trust Score:** {r.get('score', 'N/A')} / 100 ({r.get('rating', 'N/A')})\n **Reasoning:** {', '.join(r.get('reasoning', []))}")
        else: print(" Not completed.")
    else: print(" Not performed (No URL or error?).")
    
    print("\n## LinkedIn Verification (MOCK)"); li = report.get('linkedin_verification')
    if li:
        if li.get('error'): print(f" Error: {li['error']}")
        else:
            print(f" **Trust Score:** {li.get('overall_score', 'N/A')} / 100")
            name_match_info = li.get('name_match', {}); print(f" **Name Match:** {'Yes' if name_match_info.get('status') else 'NO'} ({name_match_info.get('message', 'N/A')})")
            print(f" **Plausibility:** {li.get('plausibility', {}).get('message', 'N/A')}")
            print(f" **Activity Audit:** {li.get('activity_audit', {}).get('message', 'N/A')}")
            if li.get('job_history_matches'): print(f" **Job Matches:** {', '.join(li['job_history_matches'])}")
            if li.get('skill_matches'): print(f" **Skill Matches:** {', '.join(li['skill_matches'])}")
            if li.get('discrepancies'): print(f" **Discrepancies:** {'; '.join(li['discrepancies'])}")
    else: print(" Not performed (No URL or Mock Data?).")
    
    print(f"\n**Calculated Final Score:** {report.get('final_score', 'N/A')} / 100")
    print("\n=============================="); print("--- End of Report ---")

def main():
    """ Main workflow: Gmail -> Analyze -> Verify -> Store -> Report """
    creds = None
    if os.path.exists('token.json'):
        try: creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception as e: print(f"Error loading token: {e}. Re-auth."); creds=None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try: creds.refresh(Request())
            except Exception as e: 
                print(f"Error refreshing token: {e}. Re-auth."); creds=None;
                if os.path.exists('token.json'):
                    try: os.remove('token.json'); print("Removed invalid token file.")
                    except Exception as rm_e: print(f"Failed to remove invalid token file: {rm_e}")
        if not creds or not creds.valid:
            try: flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES); creds = flow.run_local_server(port=0)
            except FileNotFoundError: print("\n[ERROR] 'credentials.json' not found. Please download it from Google Cloud Console and place it in this directory.\n"); return
            except Exception as e: print(f"Auth flow error: {e}"); return
        try:
             with open('token.json', 'w') as token: token.write(creds.to_json())
        except Exception as e: print(f"Error saving token: {e}")

    if not supabase: print("Supabase client not initialized. Cannot save data. Exiting."); return

    try:
        os.makedirs(RESUME_DIR, exist_ok=True)
        service = build('gmail', 'v1', credentials=creds)
        search_query = 'is:unread has:attachment (subject:"job" OR subject:"application" OR "job enquiry" OR subject:"resume")'
        
        list_results = service.users().messages().list(userId='me', q=search_query).execute()
        messages = list_results.get('messages', [])
        if not messages: print('No matching job application emails found.'); return
        print(f'Found {len(messages)} matching emails:')

        for message_summary in messages:
            message_id = message_summary.get('id')
            if not message_id: continue 

            github_verification_result, linkedin_verification_result = None, None
            final_score_calculated, analysis_report = 0, {}

            try:
                msg = service.users().messages().get(userId='me', id=message_id, format='full').execute()
                headers = msg.get('payload', {}).get('headers', [])
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "No Subject")
                from_email = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Unknown Sender")

                print(f"\n\n--- Processing Email From: {from_email} (ID: {message_id}) ---")
                print(f"  Subject: {subject}")

                payload = msg.get('payload', {})
                parts = payload.get('parts', []) 

                if not parts and payload.get('body', {}).get('attachmentId') and payload.get('filename'):
                    parts = [payload] 

                found_resume = False
                for part in parts:
                    filename = part.get('filename')
                    if filename: filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '_', filename) # Sanitize

                    if filename and (filename.lower().endswith('.pdf') or filename.lower().endswith('.docx')):
                        part_body = part.get('body', {}); attachment_id = part_body.get('attachmentId'); data_base64 = part_body.get('data')
                        attachment_data = None
                        if data_base64: print(f"  -> Found inline attachment: '{filename}'"); attachment_data = data_base64
                        elif attachment_id:
                            print(f"  -> Found attachment by ID: '{filename}'")
                            try: attachment = service.users().messages().attachments().get(userId='me', messageId=message_id, id=attachment_id).execute(); attachment_data = attachment.get('data')
                            except HttpError as attach_error: print(f"  [Error] Failed to get attachment {attachment_id}: {attach_error}"); continue
                        else: continue 

                        if attachment_data:
                            try: file_data = base64.urlsafe_b64decode(attachment_data.encode('UTF-8'))
                            except Exception as decode_error: print(f"  [Error] Failed to decode base64 for {filename}: {decode_error}"); continue

                            resume_text = extract_text_from_file_data(filename, file_data)
                            if resume_text:
                                analysis_report = analyze_resume(resume_text)
                                github_url = analysis_report.get('github_url')
                                if github_url:
                                    username = get_username_from_link(github_url)
                                    if username:
                                        try: github_verification_result = verify_github_profile(username)
                                        except Exception as e: github_verification_result = {'error': f'Internal GitHub check error: {e}'}
                                    else: print("  -> Could not parse GitHub username.")
                                else: print("  -> No GitHub URL in resume.")
                                
                                linkedin_url = analysis_report.get('linkedin_url')
                                if linkedin_url:
                                    linkedin_profile_data = get_linkedin_data_from_url(linkedin_url)
                                    if linkedin_profile_data:
                                        resume_data_for_li = {
                                            "name": analysis_report.get('name'), 
                                            "expertise": analysis_report.get('expertise', []), 
                                            "experience": analysis_report.get('experience', [])
                                        } 
                                        try: linkedin_verification_result = verify_linkedin_profile(resume_data_for_li, linkedin_profile_data)
                                        except Exception as e: linkedin_verification_result = {'error': f'Internal LinkedIn check error: {e}'}
                                    else: print("  -> Could not get LinkedIn mock data.")
                                else: print("  -> No LinkedIn URL in resume.")

                                final_score_calculated = calculate_final_score(github_verification_result, linkedin_verification_result)
                                analysis_report['final_score'] = final_score_calculated
                                analysis_report['github_verification'] = github_verification_result
                                analysis_report['linkedin_verification'] = linkedin_verification_result

                                print_combined_report(analysis_report)

                                if supabase:
                                    print("\n-> Saving results to Supabase...")
                                    gh_rating = github_verification_result.get('profile_rating', {}) if github_verification_result and not github_verification_result.get('error') else {}
                                    li_match = linkedin_verification_result.get('name_match', {}) if linkedin_verification_result and not linkedin_verification_result.get('error') else {}
                                    ai_report = analysis_report.get('ai_detection', {})
                                    ai_prob = ai_report.get('ai_probability') if ai_report.get('status') == 'success' else None
                                    cert_report_for_db = analysis_report.get('certificate_verification')
                                    
                                    data_to_save = {
                                        'name': analysis_report.get('name'),
                                        'email': from_email,
                                        'linkedin_url': analysis_report.get('linkedin_url'),
                                        'github_url': analysis_report.get('github_url'),
                                        'ai_summary': analysis_report.get('description'),
                                        'skills': analysis_report.get('expertise'),
                                        'suggested_role': analysis_report.get('suitable_role'),
                                        'education': analysis_report.get('education'),
                                        'experience': analysis_report.get('experience'),
                                        'github_score': gh_rating.get('score'),
                                        'github_rating': gh_rating.get('rating'),
                                        'github_reasoning': gh_rating.get('reasoning'),
                                        'github_verification_error': github_verification_result.get('error') if github_verification_result else None,
                                        'linkedin_score': linkedin_verification_result.get('overall_score') if linkedin_verification_result and not linkedin_verification_result.get('error') else None,
                                        'linkedin_name_match': li_match.get('status'),
                                        'linkedin_discrepancies': linkedin_verification_result.get('discrepancies') if linkedin_verification_result and not linkedin_verification_result.get('error') else None,
                                        'linkedin_verification_error': linkedin_verification_result.get('error') if linkedin_verification_result else None,
                                        'ai_check_status': ai_report.get('status'),
                                        'ai_probability': ai_prob,
                                        'certificate_verification_report': cert_report_for_db,
                                        'final_score': final_score_calculated,
                                        'resume_filename': filename
                                    }
                                    try:
                                        insert_response = supabase.table('candidates').insert(data_to_save).execute()
                                        if hasattr(insert_response, 'data') and insert_response.data: print("  -> Successfully saved to Supabase.")
                                        elif hasattr(insert_response, 'error') and insert_response.error: print(f"  -> Supabase Insert Error: {insert_response.error}")
                                        else: print(f"  -> Supabase response unclear: {insert_response}")
                                    except Exception as db_error: print(f"  -> Supabase insert EXCEPTION: {db_error}")
                                else: print("  -> Supabase client not available. Skipping save.")
                            else: print("  -> Could not extract text.")
                            
                            try: save_path = os.path.join(RESUME_DIR, filename);
                            except Exception as e: print(f"  -> Error creating save path for {filename}: {e}"); continue
                            try:
                                with open(save_path, 'wb') as f: f.write(file_data)
                                print(f"\n  -> Saved original file to: {save_path}")
                            except Exception as e: print(f"  -> Error saving file {filename}: {e}")
                            
                            found_resume = True; break 

                if not found_resume: print("  -> No PDF/DOCX resume attachment found.")
                try: 
                    service.users().messages().modify(userId='me', id=message_id, body={'removeLabelIds': ['UNREAD']}).execute()
                    print(f"  -> Marked email {message_id} as read.")
                except HttpError as label_error: 
                    print(f"  [Warn] Could not mark email {message_id} as read: {label_error}")

            except Exception as e: print(f"!!!--- Error processing email (ID: {message_id}): {e} ---!!!")

    except HttpError as error: print(f'An Google API error occurred: {error}')
    except Exception as e: print(f'An unexpected error occurred in main loop: {e}')

# -----------------------------------------------------------------
# --- PART 7: SCRIPT EXECUTION
# -----------------------------------------------------------------
if __name__ == '__main__':
    main()

