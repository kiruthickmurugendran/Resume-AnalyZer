import os
import re
import io
import base64
import json
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

# --- GitHub/LinkedIn API Imports ---
import requests
from dotenv import load_dotenv
from thefuzz import fuzz # For LinkedIn fuzzy matching

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
    API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAKn72E32CLxWP70WNMB38gX0axNmpyZac") # Fallback, use env var ideally
    if API_KEY == "AIzaSyAKn72E32CLxWP70WNMB38gX0axNmpyZac": print("Warning: Using fallback Gemini Key.") # Inform if fallback is used
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

# --- GitHub API Config ---
load_dotenv()
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")
if not GITHUB_API_TOKEN:
    print("Warning: GitHub API Token not found in .env file. GitHub verification may fail.")
    GITHUB_HEADERS = {}
else:
    GITHUB_HEADERS = {"Authorization": f"token {GITHUB_API_TOKEN}"}
GITHUB_BASE_URL = "https://api.github.com"

# --- Google API Config ---
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
RESUME_DIR = './resumes'

# --- LinkedIn Config ---
FUZZY_MATCH_THRESHOLD = 85

# -----------------------------------------------------------------
# --- PART 2: LOAD SPACY MODEL
# -----------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Error: Spacy 'en_core_web_sm' model not found. Please run:")
    print("\n   python -m spacy download en_core_web_sm\n")
    exit()

# -----------------------------------------------------------------
# --- PART 3, 4, 5: ANALYSIS & VERIFICATION FUNCTIONS
# --- (Keep ALL functions exactly as they were) ---
# -----------------------------------------------------------------
SKILL_KEYWORDS = [
    'python', 'javascript', 'java', 'c#', 'c++', 'sql', 'nosql', 'react', 'angular',
    'vue', 'node.js', 'django', 'flask', 'spring', 'mongodb', 'postgresql', 'mysql',
    'redis', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'ci/cd',
    'machine learning', 'data analysis', 'tensorflow', 'pytorch'
]
ROLE_KEYWORDS = {
    'Backend Developer': ['django', 'flask', 'node.js', 'spring', 'sql', 'nosql', 'java'],
    'Frontend Developer': ['react', 'angular', 'vue', 'javascript', 'css', 'html'],
    'Full-Stack Developer': ['react', 'node.js', 'python', 'sql', 'javascript'],
    'DevOps Engineer': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'git'],
    'Data Scientist': ['machine learning', 'python', 'tensorflow', 'pytorch', 'data analysis']
}
def extract_text_from_file_data(filename, file_data):
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
def extract_urls(text):
    urls = {'linkedin': None, 'github': None}
    linkedin_match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_-]+)/?', text, re.IGNORECASE)
    github_match = re.search(r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)/?', text, re.IGNORECASE)
    if linkedin_match: urls['linkedin'] = f"https://linkedin.com/in/{linkedin_match.group(1)}"
    if github_match: urls['github'] = f"https://github.com/{github_match.group(1)}"
    return urls
def get_gemini_description(resume_text, candidate_name):
    max_len = 15000; truncated_text = resume_text[:max_len] + ("..." if len(resume_text) > max_len else "")
    prompt = f"You are an expert HR recruitment analyst. Based *only* on the following resume text, write a 2-3 sentence professional description for the candidate, {candidate_name}. Speak in the third person. Focus on their primary role, years of experience (if mentioned), and strongest technologies.\nResume:\n---\n{truncated_text}\n---"
    try: response = gemini_model.generate_content(prompt); return response.text.strip()
    except Exception as e: print(f"\nWarning: Gemini API call failed. Fallback used. Error: {e}"); return f"Resume analysis for {candidate_name}."
def analyze_resume(resume_text):
    doc = nlp(resume_text); name = "Candidate"
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON'];
    if persons: name = persons[0]
    urls = extract_urls(resume_text); resume_text_lower = resume_text.lower()
    found_skills = list(set([skill for skill in SKILL_KEYWORDS if re.search(r'\b' + re.escape(skill) + r'\b', resume_text_lower)]))
    print("\nGenerating AI description..."); gemini_description = get_gemini_description(resume_text, name); print("...Description generated.")
    report = {'name': name.strip(), 'linkedin_url': urls.get('linkedin'), 'github_url': urls.get('github'), 'description': gemini_description, 'expertise': found_skills, 'strong_fields': [], 'potential_gaps': []}
    skill_counts = Counter(re.findall(r'\b(' + '|'.join(SKILL_KEYWORDS) + r')\b', resume_text_lower))
    report['strong_fields'] = [skill[0] for skill in skill_counts.most_common(3)]
    role_scores = Counter({role: sum(1 for skill in report['expertise'] if skill in keywords) for role, keywords in ROLE_KEYWORDS.items() if sum(1 for skill in report['expertise'] if skill in keywords) > 0})
    report['suitable_role'] = role_scores.most_common(1)[0][0] if role_scores else 'Not determined'
    if report['suitable_role'] != 'Not determined': target_skills = ROLE_KEYWORDS[report['suitable_role']]; gaps = [skill for skill in target_skills if skill not in report['expertise']]; report['potential_gaps'] = gaps if gaps else ["None."]
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
    if not GITHUB_API_TOKEN: return {"error": "GitHub API Token not configured."}
    print(f"\nStarting GitHub verification for: {username}"); repos_url, events_url, search_pr_url = f"{GITHUB_BASE_URL}/users/{username}/repos", f"{GITHUB_BASE_URL}/users/{username}/events?per_page=100", f"{GITHUB_BASE_URL}/search/issues?q=author:{username}+type:pr+is:public"
    repos_data, events_data, public_pr_count = [], [], 0; repo_response = None
    try: repo_response = requests.get(repos_url, headers=GITHUB_HEADERS, timeout=15); repo_response.raise_for_status(); repos_data = repo_response.json()
    except requests.exceptions.Timeout: print(f"  [Error] Timeout fetching repos for {username}."); return {"error": "Timeout fetching user repositories."}
    except requests.exceptions.RequestException as e:
        status_code = repo_response.status_code if repo_response else None; print(f"  [Error] Could not fetch repos: {e}" + (f" (Status: {status_code})" if status_code else ""));
        if status_code == 403: return {"error": f"Failed repo fetch: GitHub API rate limit likely exceeded (403). Check token or wait."}
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
    mock_resume_experience = resume_data.get('experience', [{"title": "Placeholder Title", "company": "Placeholder Company"}])
    for res_job in mock_resume_experience:
        matched = False
        for li_job in linkedin_data.get('experience', []):
            comp_score = fuzz.ratio(res_job.get('company',''), li_job.get('companyName', '')); title_score = fuzz.token_set_ratio(res_job.get('title',''), li_job.get('title', ''))
            if comp_score > FUZZY_MATCH_THRESHOLD and title_score > FUZZY_MATCH_THRESHOLD: report['job_history_matches'].append(f"Verified: {res_job.get('title','')}@{res_job.get('company','')}"); report['overall_score']+=15; matched=True; break
            elif comp_score > FUZZY_MATCH_THRESHOLD: disc_msg = f"Title Inflation?: Res '{res_job.get('title','')}' vs LI '{li_job.get('title', 'N/A')}'" if title_score > 50 else f"Title Mismatch: Res '{res_job.get('title','')}' vs LI '{li_job.get('title', 'N/A')}'"; report['discrepancies'].append(f"{disc_msg} at '{li_job.get('companyName', 'N/A')}'"); report['overall_score']-=15; matched=True; break
        if not matched: report['discrepancies'].append(f"Unverified Job: '{res_job.get('title','')} at {res_job.get('company','')}'")
    li_skills = [s.get('name', '').lower() for s in linkedin_data.get('skills', [])]
    for res_skill in resume_data.get('skills', []):
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
    """ Prints the combined analysis report. """
    print("\n--- COMBINED CANDIDATE REPORT ---")
    print("==============================\n")
    print(f"**Candidate Name:** {report.get('name', 'N/A')}")
    print(f"**LinkedIn URL:** {'Yes' if report.get('linkedin_url') else 'No'}")
    print(f"**GitHub URL:** {'Yes' if report.get('github_url') else 'No'}")
    print("\n## AI-Generated Description"); print(report.get('description', 'N/A'))
    print("\n## Core Expertise (Skills Identified)"); print(", ".join(report.get('expertise', [])) if report.get('expertise') else "N/A")
    print("\n## Suggested Role"); print(report.get('suitable_role', 'N/A'))
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
            except Exception as e: print(f"Error refreshing token: {e}. Re-auth."); creds=None;
            if not creds and os.path.exists('token.json'):
                try: os.remove('token.json'); print("Removed invalid token file.")
                except Exception as rm_e: print(f"Failed to remove invalid token file: {rm_e}")
        if not creds or not creds.valid:
            try: flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES); creds = flow.run_local_server(port=0)
            except Exception as e: print(f"Auth flow error: {e}"); return
        try:
             with open('token.json', 'w') as token: token.write(creds.to_json())
        except Exception as e: print(f"Error saving token: {e}")

    if not supabase: print("Supabase client not initialized. Cannot save data. Exiting."); return

    try:
        os.makedirs(RESUME_DIR, exist_ok=True)
        service = build('gmail', 'v1', credentials=creds)
        search_query = 'is:unread has:attachment (subject:"job" OR subject:"application" OR "job enquiry" OR subject:"resume")'
        # Get message IDs first
        list_results = service.users().messages().list(userId='me', q=search_query).execute()
        messages = list_results.get('messages', [])
        if not messages: print('No matching job application emails found.'); return
        print(f'Found {len(messages)} matching emails:')

        for message_summary in messages:
            message_id = message_summary.get('id')
            if not message_id: continue # Skip if no ID found

            github_verification_result, linkedin_verification_result = None, None
            final_score_calculated, analysis_report = 0, {}

            try:
                # Fetch full message details
                msg = service.users().messages().get(userId='me', id=message_id, format='full').execute()
                headers = msg.get('payload', {}).get('headers', [])
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "No Subject")
                from_email = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Unknown Sender")

                print(f"\n\n--- Processing Email From: {from_email} (ID: {message_id}) ---")
                print(f"  Subject: {subject}")

                payload = msg.get('payload', {})
                parts = payload.get('parts', []) # Directly get parts list

                # If no parts, check if the main body might be an attachment
                if not parts and payload.get('body', {}).get('attachmentId') and payload.get('filename'):
                    parts = [payload] # Treat the main payload as the only part

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
                        else: continue # Skip part if no data or ID

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
                                        resume_data_for_li = {"name": analysis_report.get('name'), "skills": analysis_report.get('expertise', []), "experience": []}
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
                                    data_to_save = { 'name': analysis_report.get('name'), 'email': from_email, 'linkedin_url': analysis_report.get('linkedin_url'), 'github_url': analysis_report.get('github_url'), 'ai_summary': analysis_report.get('description'), 'skills': analysis_report.get('expertise'), 'suggested_role': analysis_report.get('suitable_role'), 'github_score': gh_rating.get('score'), 'github_rating': gh_rating.get('rating'), 'github_reasoning': gh_rating.get('reasoning'), 'github_verification_error': github_verification_result.get('error') if github_verification_result else None, 'linkedin_score': linkedin_verification_result.get('overall_score') if linkedin_verification_result and not linkedin_verification_result.get('error') else None, 'linkedin_name_match': li_match.get('status'), 'linkedin_discrepancies': linkedin_verification_result.get('discrepancies') if linkedin_verification_result and not linkedin_verification_result.get('error') else None, 'linkedin_verification_error': linkedin_verification_result.get('error') if linkedin_verification_result else None, 'final_score': final_score_calculated, 'resume_filename': filename }
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
                try: service.users().messages().modify(userId='me', id=message_id, body={'removeLabelIds': ['UNREAD']}).execute(); print(f"  -> Marked email {message_id} as read.")
                except HttpError as label_error: print(f"  [Warn] Could not mark email {message_id} as read: {label_error}")

            except Exception as e: print(f"!!!--- Error processing email (ID: {message_id}): {e} ---!!!")

    except HttpError as error: print(f'An Google API error occurred: {error}')
    except Exception as e: print(f'An unexpected error occurred in main loop: {e}')

# -----------------------------------------------------------------
# --- PART 7: SCRIPT EXECUTION
# -----------------------------------------------------------------
if __name__ == '__main__':
    main()