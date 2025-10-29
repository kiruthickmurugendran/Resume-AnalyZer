import os
import re
import io
import base64
import json
from collections import Counter
from datetime import datetime, timedelta # <-- Added timedelta just in case, though not used in GitHub code provided
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

# -----------------------------------------------------------------
# --- PART 1: CONFIGURATIONS
# -----------------------------------------------------------------

# --- Gemini API Config ---
try:
    # --- Corrected Model Name ---
    API_KEY = "AIzaSyAKn72E32CLxWP70WNMB38gX0axNmpyZac" # Your Gemini Key - Use environment variables in production!
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp') # Corrected model name
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

# --- GitHub API Config ---
load_dotenv()
GITHUB_API_TOKEN = os.getenv("GITHUB_API_TOKEN")
if not GITHUB_API_TOKEN:
    print("Warning: GitHub API Token not found. GitHub verification may fail.")
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
    print("Error: 'en_core_web_sm' model not found. Please run:")
    print("\n   python -m spacy download en_core_web_sm\n")
    exit()

# -----------------------------------------------------------------
# --- PART 3: AI & RESUME ANALYSIS FUNCTIONS
# -----------------------------------------------------------------

# Keywords remain the same as before
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
    """ Extracts raw text from in-memory file data (bytes). """
    text = ""
    file_like_object = io.BytesIO(file_data)
    try:
        if filename.lower().endswith('.pdf'):
            with pdfplumber.open(file_like_object) as pdf:
                for page in pdf.pages: text += page.extract_text() or ""
        elif filename.lower().endswith('.docx'):
            doc = Document(file_like_object)
            for para in doc.paragraphs: text += para.text + "\n"
        print(f"  -> Successfully extracted text from '{filename}'.")
        return text
    except Exception as e:
        print(f"  -> Error reading file {filename}: {e}")
        return None

def extract_urls(text):
    """ Uses regex to find LinkedIn and GitHub URLs. """
    urls = {'linkedin': None, 'github': None}
    linkedin_match = re.search(r'linkedin\.com/in/([a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    github_match = re.search(r'github\.com/([a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    if linkedin_match: urls['linkedin'] = f"https://linkedin.com/in/{linkedin_match.group(1)}"
    if github_match: urls['github'] = f"https://github.com/{github_match.group(1)}"
    return urls

def get_gemini_description(resume_text, candidate_name):
    """ Uses the Gemini API to generate a professional summary. """
    prompt = f"You are an expert HR recruitment analyst. Based *only* on the following resume text, write a 2-3 sentence professional description for the candidate, {candidate_name}. Speak in the third person. Focus on their primary role, years of experience, and strongest technologies.\nResume:\n---\n{resume_text}\n---"
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"\nWarning: Could not call Gemini API. Using fallback. Error: {e}")
        return f"Resume for {candidate_name}, showing software development experience."

def analyze_resume(resume_text):
    """ Main AI function to parse text and build a report. """
    doc = nlp(resume_text)
    name = "Candidate"
    for ent in doc.ents:
        if ent.label_ == 'PERSON': name = ent.text; break
    urls = extract_urls(resume_text)
    resume_text_lower = resume_text.lower()
    found_skills = list(set([skill for skill in SKILL_KEYWORDS if re.search(r'\b' + re.escape(skill) + r'\b', resume_text_lower)]))

    print("\nGenerating AI description...")
    gemini_description = get_gemini_description(resume_text, name)
    print("...Description generated.")

    report = {'name': name, 'linkedin_url': urls.get('linkedin'), 'github_url': urls.get('github'),
              'description': gemini_description, 'expertise': found_skills, 'strong_fields': [], 'potential_gaps': []}

    skill_counts = Counter(re.findall(r'\b(' + '|'.join(SKILL_KEYWORDS) + r')\b', resume_text_lower))
    report['strong_fields'] = [skill[0] for skill in skill_counts.most_common(3)]

    role_scores = Counter({role: sum(1 for skill in report['expertise'] if skill in keywords)
                           for role, keywords in ROLE_KEYWORDS.items() if sum(1 for skill in report['expertise'] if skill in keywords) > 0})
    report['suitable_role'] = role_scores.most_common(1)[0][0] if role_scores else 'Not determined'

    if report['suitable_role'] != 'Not determined':
        target_skills = ROLE_KEYWORDS[report['suitable_role']]
        gaps = [skill for skill in target_skills if skill not in report['expertise']]
        report['potential_gaps'] = gaps if gaps else ["None found for this role."]
    return report

# -----------------------------------------------------------------
# --- PART 4: GITHUB VERIFICATION FUNCTIONS (REPLACED)
# -----------------------------------------------------------------

# --- NEW Function from your working GitHub script ---
def get_username_from_link(link_or_username):
    """
    Extracts GitHub username from a full URL or returns it if it's already a username.
    """
    if not link_or_username: # Handle None case
        return None
    # Check if it looks like a URL
    if link_or_username.startswith("http://") or link_or_username.startswith("https://"):
        try:
            path = urlparse(link_or_username).path
            # The path will be '/username', so we split by '/' and take the 2nd item
            parts = path.strip().split('/')
            if len(parts) > 1 and parts[1]:
                username = parts[1]
                print(f"  [Info] Extracted GitHub username '{username}' from link.")
                return username
            else:
                raise ValueError("Could not parse username from link path.")
        except Exception as e:
            print(f"  [Error] Invalid GitHub link provided ('{link_or_username}'): {e}")
            return None

    # If not a link, assume it's already a username (basic check)
    if '/' not in link_or_username and '.' not in link_or_username:
         print(f"  [Info] Assuming '{link_or_username}' is a GitHub username.")
         return link_or_username
    else:
        print(f"  [Error] Invalid format for GitHub username or link: '{link_or_username}'")
        return None


# --- NEW Function from your working GitHub script (renamed from calculate_profile_rating)---
def calculate_github_profile_rating(report):
    """
    Analyzes the verification report to generate a score and rating.
    This is the "Trust Score" logic.
    """
    score = 50  # Start with a neutral score
    reasons = []

    # Check if essential keys exist before accessing them
    collaboration_check = report.get("collaboration_check", {})
    originality_check = report.get("originality_check", {})
    consistency_check = report.get("consistency_check", {})

    # 1. Collaboration Check (Gold Standard)
    if collaboration_check.get("is_collaborator"):
        score += 30
        reasons.append("+30: 'Collaborator' (Contributed to public projects)")

    # 2. Originality Check
    if originality_check.get("red_flag_empty_repos"):
        score -= 25
        reasons.append("-25: 'Empty Profile' (Repos have no code)")
    elif originality_check.get("total_code_bytes", 0) > 50000: # 50k bytes
        score += 10
        reasons.append("+10: 'Coder' (Has a reasonable amount of original code)")

    # 3. Consistency Check
    if consistency_check.get("red_flag_bunched_activity"):
        score -= 30
        reasons.append("-30: 'Bunched Activity' (High-activity in < 2 days, looks fake)")
    elif consistency_check.get("activity_spread_days", 0) > 30:
        score += 20
        reasons.append("+20: 'Consistent' (Activity spread over 30+ days)")
    elif consistency_check.get("activity_spread_days", 0) > 7:
        score += 10
        reasons.append("+10: 'Active' (Activity spread over 7+ days)")

    if consistency_check.get("recent_activity_count", 0) < 5:
        score -= 10
        reasons.append("-10: 'Inactive' (Very few recent events)")

    # Final Rating
    if score >= 75:
        rating = "Excellent"
    elif score >= 45:
        rating = "Good"
    else:
        rating = "Bad"

    return {
        "score": max(0, min(score, 100)), # Clamp score between 0 and 100
        "rating": rating,
        "reasoning": reasons
    }

# --- NEW Function from your working GitHub script ---
def verify_github_profile(username):
    """
    Performs an "Activity and Quality Audit" on a GitHub user.
    Fetches all data and returns a final report including a profile rating.
    """
    if not GITHUB_API_TOKEN:
        return {"error": "GitHub API Token is not configured."}

    print(f"\nStarting GitHub verification for: {username}")

    # --- Check 1: Originality (The "Empty Repo" Check) ---
    print("-> Checking Originality (Repos)...")
    repos_url = f"{GITHUB_BASE_URL}/users/{username}/repos"
    try:
        repo_response = requests.get(repos_url, headers=GITHUB_HEADERS, timeout=15) # Add timeout
        repo_response.raise_for_status()
        repos_data = repo_response.json()
    except requests.exceptions.Timeout:
         print(f"  [Error] Timeout fetching repos for {username}.")
         return {"error": f"Timeout fetching user repositories."}
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Could not fetch repos: {e}")
        # Check for rate limit error specifically
        if repo_response and repo_response.status_code == 403:
             return {"error": f"Failed repo fetch: GitHub API rate limit likely exceeded. Check token or wait. Details: {e}"}
        return {"error": f"Failed repo fetch: {e}"}


    original_repos = []
    language_stats = {}
    total_code_bytes = 0

    print(f"  - Found {len(repos_data)} repos. Analyzing original ones...")
    for repo in repos_data:
        if not repo['fork']:
            original_repos.append(repo['name'])
            lang_url = repo['languages_url']
            try:
                lang_response = requests.get(lang_url, headers=GITHUB_HEADERS, timeout=10) # Shorter timeout for languages
                lang_response.raise_for_status()
                languages = lang_response.json()

                if not languages:
                    #print(f"  - Repo '{repo['name']}' has no code detected via languages API.") # Optional: less verbose
                    continue

                for lang, bytes_of_code in languages.items():
                    total_code_bytes += bytes_of_code
                    language_stats[lang] = language_stats.get(lang, 0) + bytes_of_code

            except requests.exceptions.Timeout:
                #print(f"  - Timeout getting languages for '{repo['name']}'. Skipping lang analysis for this repo.") # Optional: less verbose
                pass # Continue without languages for this repo
            except requests.exceptions.RequestException:
                #print(f"  - Could not get languages for '{repo['name']}'. Skipping lang analysis for this repo.") # Optional: less verbose
                pass # Continue without languages for this repo


    # --- Check 2: Consistency (The "Anti-Fake" Check) ---
    print("-> Checking Consistency (Events)...")
    events_url = f"{GITHUB_BASE_URL}/users/{username}/events?per_page=100" # Max 100 events
    try:
        events_response = requests.get(events_url, headers=GITHUB_HEADERS, timeout=15)
        events_response.raise_for_status()
        events_data = events_response.json()
    except requests.exceptions.Timeout:
         print(f"  [Error] Timeout fetching events for {username}.")
         events_data = [] # Treat as no events found
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Could not fetch events: {e}")
        events_data = []

    push_events_timestamps = []
    if events_data: # Only process if we got data
        for event in events_data:
            # Focus on significant contribution events
            if event['type'] in ['PushEvent', 'PullRequestEvent', 'CreateEvent', 'IssuesEvent']:
                timestamp_str = event.get('created_at')
                if timestamp_str:
                     push_events_timestamps.append(timestamp_str)


    # --- Check 3: Collaboration (The "Gold Standard" Check) ---
    print("-> Checking Collaboration (Public PRs)...")
    search_pr_url = f"{GITHUB_BASE_URL}/search/issues?q=author:{username}+type:pr+is:public"
    public_pr_count = 0
    try:
        pr_response = requests.get(search_pr_url, headers=GITHUB_HEADERS, timeout=15)
        pr_response.raise_for_status()
        pr_data = pr_response.json()
        public_pr_count = pr_data.get('total_count', 0)
        print(f"  - Found {public_pr_count} public pull requests.")
    except requests.exceptions.Timeout:
         print(f"  [Error] Timeout fetching public PRs for {username}.")
         # Allow score calculation without this data point
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Could not fetch public PRs: {e}")
        pass # Allow score calculation without this data point

    # --- 4. Analyze and Build the Intermediate Report ---
    print("-> Analyzing data...")
    activity_spread_days = 0
    is_bunched_activity = False

    if len(push_events_timestamps) > 1:
        # Sort timestamps just in case API doesn't guarantee order (newest first)
        push_events_timestamps.sort(reverse=True)
        try:
             # Use datetime.strptime for robustness with GitHub's format
             last_push = datetime.strptime(push_events_timestamps[0], '%Y-%m-%dT%H:%M:%SZ')
             first_push = datetime.strptime(push_events_timestamps[-1], '%Y-%m-%dT%H:%M:%SZ')
             activity_spread_days = (last_push - first_push).days

             # Bunched activity check: more than 10 events within 2 days (exclusive)
             if activity_spread_days < 2 and len(push_events_timestamps) > 10:
                 is_bunched_activity = True
        except ValueError as dt_error:
             print(f"  [Warn] Could not parse event timestamps: {dt_error}")
             # Cannot calculate spread if parsing fails

    language_percentage = {}
    if total_code_bytes > 0:
        for lang, bytes_of_code in language_stats.items():
            percentage = round((bytes_of_code / total_code_bytes) * 100, 2)
            language_percentage[lang] = percentage

    # --- Build the data-only report first ---
    report = {
        "username": username,
        "verification_status": "Completed", # Assume completed unless error occurred earlier
        "originality_check": {
            "total_public_repos": len(repos_data), # Total count from initial fetch
            "original_repo_count": len(original_repos),
            "forked_repo_count": len(repos_data) - len(original_repos),
            "primary_languages": sorted(language_percentage.items(), key=lambda item: item[1], reverse=True)[:5], # Top 5 langs
            "total_code_bytes": total_code_bytes,
            "red_flag_empty_repos": total_code_bytes == 0 and len(original_repos) > 0
        },
        "consistency_check": {
            "recent_activity_count": len(push_events_timestamps),
            "activity_spread_days": activity_spread_days,
            "red_flag_bunched_activity": is_bunched_activity
        },
        "collaboration_check": {
            "public_pull_requests_made": public_pr_count,
            "is_collaborator": public_pr_count > 0
        }
    }

    # --- 5. Final Step: Calculate Rating and Add to Report ---
    print("-> Generating final rating...")
    report["profile_rating"] = calculate_github_profile_rating(report)

    return report


# -----------------------------------------------------------------
# --- PART 5: LINKEDIN VERIFICATION FUNCTIONS (MOCK)
# -----------------------------------------------------------------
# (LinkedIn functions remain unchanged from previous version)

def get_linkedin_data_from_url(linkedin_url):
    """ MOCK: Reads LinkedIn data from 'my_profile_data.json'. """
    print(f"\n[MOCK LinkedIn API] Reading from 'my_profile_data.json'...")
    try:
        with open('my_profile_data.json', 'r', encoding='utf-8') as f: data = json.load(f)
        if isinstance(data, list) and data: return data[0]
        if isinstance(data, dict): return data
        print("[MOCK LinkedIn API] FAILED: JSON empty or wrong format.")
        return None
    except FileNotFoundError: print("[MOCK LinkedIn API] FAILED: 'my_profile_data.json' not found!"); return None
    except json.JSONDecodeError: print("[MOCK LinkedIn API] FAILED: Could not read JSON."); return None
    except Exception as e: print(f"[MOCK LinkedIn API] FAILED: Error: {e}"); return None

def analyze_linkedin_post_activity(posts):
    """ Analyzes LinkedIn posts for activity score boost. """
    if not posts: return {"status": "Not Genuine", "message": "No post activity.", "score_boost": -10}
    orig, likes, total = 0, 0, len(posts)
    for post in posts:
        if post.get('type') == 'PERSONAL_POST': orig += 1
        likes += post.get('likesCount', 0)
    ratio = (orig / total) if total > 0 else 0
    msg = f"Found {total} posts. "; boost = 10
    msg += f"Originality {int(ratio*100)}%. " + ("High." if ratio >= 0.5 else "Low (mostly reshares).")
    if ratio >= 0.5: boost += 10
    if likes > 100: msg += " High engagement."; boost += 10
    return {"status": "Genuine", "message": msg, "score_boost": boost}

def verify_linkedin_profile(resume_data, linkedin_data):
    """ Analyzes resume against MOCK LinkedIn profile JSON. """
    print("-> Starting LinkedIn verification...")
    report = {"overall_score": 0, "name_match": {"status": False, "message": ""}, "plausibility": {"status": "Not Plausible", "message": ""},
              "activity_audit": {"status": "Not Genuine", "message": ""}, "job_history_matches": [], "skill_matches": [], "discrepancies": []}

    name_score = fuzz.token_set_ratio(resume_data['name'], linkedin_data.get('fullName', ''))
    if name_score > FUZZY_MATCH_THRESHOLD: report['name_match']['status']=True; report['name_match']['message']=f"Name matches '{linkedin_data.get('fullName')}'"; report['overall_score']+=20
    else: report['name_match']['message']=f"Name mismatch: Resume '{resume_data['name']}', LinkedIn '{linkedin_data.get('fullName')}'"

    if linkedin_data.get('verified', False): report['overall_score']+=30; report['plausibility']['message']="Verified Profile. "
    conn = linkedin_data.get('connectionCount', 0)
    if conn > 100: report['overall_score']+=10; report['plausibility']['message']+=f"{conn} connections. "
    else: report['plausibility']['message']+=f"Small network ({conn}). "
    recs = linkedin_data.get('recommendationCount', 0)
    if recs > 0: report['overall_score']+=10; report['plausibility']['message']+=f"{recs} recommendations."

    # Using skills from AI analysis for comparison
    mock_resume_experience = [{"title": "Placeholder Title", "company": "Placeholder Company"}] # Keep placeholder for now
    if 'experience' in resume_data and resume_data['experience']: # Check if AI provides this structure later
        mock_resume_experience = resume_data['experience']

    for resume_job in mock_resume_experience:
        matched = False
        for li_job in linkedin_data.get('experience', []):
            comp_score = fuzz.ratio(resume_job['company'], li_job.get('companyName', ''))
            title_score = fuzz.token_set_ratio(resume_job['title'], li_job.get('title', ''))
            if comp_score > FUZZY_MATCH_THRESHOLD and title_score > FUZZY_MATCH_THRESHOLD:
                report['job_history_matches'].append(f"Verified: {resume_job['title']} at {resume_job['company']}"); report['overall_score']+=15; matched=True; break
            elif comp_score > FUZZY_MATCH_THRESHOLD:
                # Be more specific about potential inflation vs simple mismatch
                if title_score > 50: # If title is somewhat similar, suggest inflation
                     report['discrepancies'].append(f"Potential Title Inflation?: Res '{resume_job['title']}' vs LI '{li_job.get('title', 'N/A')}' at '{li_job.get('companyName', 'N/A')}'")
                else: # Otherwise just note the mismatch
                     report['discrepancies'].append(f"Job Title Mismatch: Res '{resume_job['title']}' vs LI '{li_job.get('title', 'N/A')}' at '{li_job.get('companyName', 'N/A')}'")
                report['overall_score']-=15 # Penalize less severely than a completely missing job
                matched=True; break
        if not matched: report['discrepancies'].append(f"Unverified Job: '{resume_job['title']} at {resume_job['company']}'")

    li_skills = [s.get('name', '').lower() for s in linkedin_data.get('skills', [])]
    # Use skills extracted by AI ('expertise')
    for res_skill in resume_data.get('skills', []): # skills key comes from resume_data_for_li now
        found = False
        for li_skill in li_skills:
            if fuzz.partial_ratio(res_skill.lower(), li_skill) > 90: report['skill_matches'].append(f"Verified Skill: '{res_skill}'"); report['overall_score']+=2; found=True; break
        if not found: report['discrepancies'].append(f"Unverified Skill: '{res_skill}'")

    post_report = analyze_linkedin_post_activity(linkedin_data.get('posts', []))
    report['activity_audit'] = post_report
    report['overall_score'] += post_report['score_boost']
    report['overall_score'] = max(0, min(100, report['overall_score']))
    print("-> LinkedIn verification complete.")
    return report

# -----------------------------------------------------------------
# --- PART 6: PRINTING AND MAIN GMAIL FUNCTION
# -----------------------------------------------------------------

def print_combined_report(report):
    """ Prints the combined analysis report. """
    print("\n--- COMBINED CANDIDATE REPORT ---")
    print("==============================\n")
    print(f"**Candidate Name:** {report.get('name', 'N/A')}") # Use .get for safety
    print(f"**LinkedIn URL:** {'Yes' if report.get('linkedin_url') else 'No'}")
    print(f"**GitHub URL:** {'Yes' if report.get('github_url') else 'No'}")

    print("\n## AI-Generated Description")
    print(report.get('description', 'N/A'))

    print("\n## Core Expertise (Skills Identified)")
    print(", ".join(report.get('expertise', [])) if report.get('expertise') else "N/A")

    print("\n## Suggested Role")
    print(report.get('suitable_role', 'N/A'))

    print("\n## GitHub Verification")
    gh = report.get('github_verification')
    if gh:
        if gh.get('error'): print(f" Error: {gh['error']}")
        elif gh.get('verification_status') == "Completed":
            r = gh.get('profile_rating', {})
            print(f" **Trust Score:** {r.get('score', 'N/A')} / 100 ({r.get('rating', 'N/A')})")
            print(f" **Reasoning:** {', '.join(r.get('reasoning', []))}")
        else: print(" Not completed.")
    else: print(" Not performed (No URL or error?).") # Clarified message

    print("\n## LinkedIn Verification (MOCK)")
    li = report.get('linkedin_verification')
    if li:
        if li.get('error'): print(f" Error: {li['error']}")
        else:
            print(f" **Trust Score:** {li.get('overall_score', 'N/A')} / 100")
            # Use .get for nested dicts too
            name_match_info = li.get('name_match', {})
            print(f" **Name Match:** {'Yes' if name_match_info.get('status') else 'NO'} ({name_match_info.get('message', 'N/A')})")
            print(f" **Plausibility:** {li.get('plausibility', {}).get('message', 'N/A')}")
            print(f" **Activity Audit:** {li.get('activity_audit', {}).get('message', 'N/A')}")
            if li.get('job_history_matches'): print(f" **Job Matches:** {', '.join(li['job_history_matches'])}")
            if li.get('skill_matches'): print(f" **Skill Matches:** {', '.join(li['skill_matches'])}")
            if li.get('discrepancies'): print(f" **Discrepancies:** {'; '.join(li['discrepancies'])}")
    else: print(" Not performed (No URL or Mock Data?).")

    print("\n==============================")
    print("--- End of Report ---")


def main():
    """ Main workflow: Gmail -> Analyze -> Verify GitHub -> Verify LinkedIn -> Report """
    creds = None
    # Google Auth flow (remains the same)
    if os.path.exists('token.json'):
        try: creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        except Exception as e: print(f"Error loading token.json: {e}. Re-authenticating."); creds=None # Handle corrupted token
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try: creds.refresh(Request())
            except Exception as e: print(f"Error refreshing token: {e}. Re-authenticating."); creds=None; os.remove('token.json') # Remove bad token
        else:
            try:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES);
                creds = flow.run_local_server(port=0)
            except Exception as e: print(f"Error during auth flow: {e}"); return # Exit if auth fails
        try:
             with open('token.json', 'w') as token: token.write(creds.to_json())
        except Exception as e: print(f"Error saving token: {e}") # Non-fatal, but warn user

    # --- Start Main Logic ---
    try:
        os.makedirs(RESUME_DIR, exist_ok=True)
        service = build('gmail', 'v1', credentials=creds)
        search_query = 'is:unread has:attachment (subject:"job" OR subject:"application" OR "job enquiry" OR subject:"resume")'
        results = service.users().messages().list(userId='me', q=search_query).execute()
        messages = results.get('messages', [])
        if not messages: print('No matching job application emails found.'); return
        print(f'Found {len(messages)} matching emails:')

        for message in messages:
            try: # Add try/except around each email processing step
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                headers = msg.get('payload', {}).get('headers', [])
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), "No Subject")
                from_email = next((h['value'] for h in headers if h['name'].lower() == 'from'), "Unknown Sender")

                print(f"\n\n--- Processing Email From: {from_email} ---")
                print(f"  Subject: {subject}")

                parts = msg.get('payload', {}).get('parts', [])
                found_resume = False
                for part in parts:
                    filename = part.get('filename')
                    if filename and (filename.lower().endswith('.pdf') or filename.lower().endswith('.docx')):
                        attachment_id = part.get('body', {}).get('attachmentId')
                        if attachment_id:
                            print(f"  -> Found resume file: '{filename}'")
                            attachment = service.users().messages().attachments().get(userId='me', messageId=message['id'], id=attachment_id).execute()
                            data = attachment.get('data')
                            if data:
                                # 1. Decode & Extract Text
                                file_data = base64.urlsafe_b64decode(data.encode('UTF-8'))
                                resume_text = extract_text_from_file_data(filename, file_data)
                                if resume_text:
                                    # 2. AI Resume Analysis
                                    analysis_report = analyze_resume(resume_text)

                                    # 3. GitHub Verification
                                    github_url = analysis_report.get('github_url')
                                    if github_url:
                                        username = get_username_from_link(github_url)
                                        if username:
                                            try: analysis_report['github_verification'] = verify_github_profile(username)
                                            except Exception as e: print(f"  -> GitHub verification function error: {e}"); analysis_report['github_verification'] = {'error': f'Internal error during GitHub check: {e}'}
                                        else: print("  -> Could not parse GitHub username from URL.")
                                    else: print("  -> No GitHub URL found in resume text.")

                                    # 4. LinkedIn Verification (MOCK)
                                    linkedin_url = analysis_report.get('linkedin_url')
                                    if linkedin_url:
                                        linkedin_profile_data = get_linkedin_data_from_url(linkedin_url) # Reads from JSON
                                        if linkedin_profile_data:
                                            # Prepare resume data for LinkedIn comparison using AI analysis results
                                            resume_data_for_li = {
                                                "name": analysis_report.get('name', 'Candidate'),
                                                # Pass actual extracted skills to LinkedIn verifier
                                                "skills": analysis_report.get('expertise', []),
                                                # Still using placeholder experience until AI extracts it
                                                "experience": [{"title": "Placeholder Title", "company": "Placeholder Company"}],
                                            }
                                            try: analysis_report['linkedin_verification'] = verify_linkedin_profile(resume_data_for_li, linkedin_profile_data)
                                            except Exception as e: print(f"  -> LinkedIn verification function error: {e}"); analysis_report['linkedin_verification'] = {'error': f'Internal error during LinkedIn check: {e}'}
                                        else: print("  -> Could not get LinkedIn mock data (check my_profile_data.json).")
                                    else: print("  -> No LinkedIn URL found in resume text.")

                                    # 5. Print Combined Report
                                    print_combined_report(analysis_report)
                                else: print("  -> Could not extract text from resume file.")

                                # 6. Save Original File
                                try:
                                    save_path = os.path.join(RESUME_DIR, filename)
                                    with open(save_path, 'wb') as f: f.write(file_data)
                                    print(f"\n  -> Saved original file to: {save_path}")
                                except Exception as e: print(f"  -> Error saving file {filename}: {e}") # Catch save errors

                                found_resume = True
                                break # Process only the first valid resume found in an email
                if not found_resume: print("  -> No PDF/DOCX resume attachment found in this email.")

            except Exception as e: # Catch errors processing a single email
                print(f"!!!--- Error processing email (ID: {message.get('id', 'N/A')}): {e} ---!!!")
                # Continue to the next email

    except HttpError as error: print(f'An Google API error occurred: {error}')
    except Exception as e: print(f'An unexpected error occurred in main loop: {e}') # Catch other potential errors

# -----------------------------------------------------------------
# --- PART 7: SCRIPT EXECUTION
# -----------------------------------------------------------------
if __name__ == '__main__':
    main()