import requests
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from urllib.parse import urlparse # New import for parsing links

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
API_TOKEN = os.getenv("GITHUB_API_TOKEN")
if not API_TOKEN:
    raise ValueError("GitHub API Token not found. Please set GITHUB_API_TOKEN in your .env file.")

HEADERS = {"Authorization": f"token {API_TOKEN}"}
BASE_URL = "https://api.github.com"

# --- NEW: Helper function to get username from URL ---
def get_username_from_link(link_or_username):
    """
    Extracts GitHub username from a full URL or returns it if it's already a username.
    """
    # Check if it looks like a URL
    if link_or_username.startswith("http://") or link_or_username.startswith("https://"):
        try:
            path = urlparse(link_or_username).path
            # The path will be '/username', so we split by '/' and take the 2nd item
            username = path.strip().split('/')[1]
            if username:
                print(f"  [Info] Extracted username '{username}' from link.")
                return username
            else:
                raise ValueError("Could not parse username from link.")
        except Exception as e:
            print(f"  [Error] Invalid GitHub link provided: {e}")
            return None
    
    # If not a link, assume it's already a username
    print(f"  [Info] Using provided username '{link_or_username}'.")
    return link_or_username

# --- NEW: Helper function to calculate the profile rating ---
def calculate_profile_rating(report):
    """
    Analyzes the verification report to generate a score and rating.
    This is the "Trust Score" logic.
    """
    score = 50  # Start with a neutral score
    reasons = []

    # 1. Collaboration Check (Gold Standard)
    if report["collaboration_check"]["is_collaborator"]:
        score += 30
        reasons.append("+30: 'Collaborator' (Contributed to public projects)")
    
    # 2. Originality Check
    if report["originality_check"]["red_flag_empty_repos"]:
        score -= 25
        reasons.append("-25: 'Empty Profile' (Repos have no code)")
    elif report["originality_check"]["total_code_bytes"] > 50000: # 50k bytes
        score += 10
        reasons.append("+10: 'Coder' (Has a reasonable amount of original code)")

    # 3. Consistency Check
    if report["consistency_check"]["red_flag_bunched_activity"]:
        score -= 30
        reasons.append("-30: 'Bunched Activity' (High-activity in < 2 days, looks fake)")
    elif report["consistency_check"]["activity_spread_days"] > 30:
        score += 20
        reasons.append("+20: 'Consistent' (Activity spread over 30+ days)")
    elif report["consistency_check"]["activity_spread_days"] > 7:
        score += 10
        reasons.append("+10: 'Active' (Activity spread over 7+ days)")
    
    if report["consistency_check"]["recent_activity_count"] < 5:
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


def verify_github_profile(username):
    """
    Performs an "Activity and Quality Audit" on a GitHub user.

    Fetches all data and returns a final report including a profile rating.
    """
    print(f"Starting verification for: {username}")
    
    # --- Check 1: Originality (The "Empty Repo" Check) ---
    print("-> Checking Originality (Repos)...")
    repos_url = f"{BASE_URL}/users/{username}/repos"
    try:
        repo_response = requests.get(repos_url, headers=HEADERS)
        repo_response.raise_for_status() 
        repos_data = repo_response.json()
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Could not fetch repos: {e}")
        return {"error": f"Failed to fetch user repositories. {e}"}

    original_repos = []
    language_stats = {}
    total_code_bytes = 0
    
    for repo in repos_data:
        if not repo['fork']:
            original_repos.append(repo['name'])
            lang_url = repo['languages_url']
            try:
                lang_response = requests.get(lang_url, headers=HEADERS)
                lang_response.raise_for_status()
                languages = lang_response.json() 
                
                if not languages:
                    print(f"  - Repo '{repo['name']}' has no code.")
                    continue
                    
                for lang, bytes_of_code in languages.items():
                    total_code_bytes += bytes_of_code
                    language_stats[lang] = language_stats.get(lang, 0) + bytes_of_code
                    
            except requests.exceptions.RequestException:
                print(f"  - Could not get languages for '{repo['name']}'. Skipping.")
                pass

    # --- Check 2: Consistency (The "Anti-Fake" Check) ---
    print("-> Checking Consistency (Events)...")
    events_url = f"{BASE_URL}/users/{username}/events?per_page=100"
    try:
        events_response = requests.get(events_url, headers=HEADERS)
        events_response.raise_for_status()
        events_data = events_response.json()
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Could not fetch events: {e}")
        events_data = []

    push_events_timestamps = []
    for event in events_data:
        if event['type'] in ['PushEvent', 'PullRequestEvent']:
            push_events_timestamps.append(event['created_at'])

    # --- Check 3: Collaboration (The "Gold Standard" Check) ---
    print("-> Checking Collaboration (Public PRs)...")
    search_pr_url = f"{BASE_URL}/search/issues?q=author:{username}+type:pr+is:public"
    public_pr_count = 0
    try:
        pr_response = requests.get(search_pr_url, headers=HEADERS)
        pr_response.raise_for_status()
        pr_data = pr_response.json()
        public_pr_count = pr_data.get('total_count', 0)
        print(f"  - Found {public_pr_count} public pull requests.")
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Could not fetch public PRs: {e}")
        pass 

    # --- 4. Analyze and Build the Intermediate Report ---
    print("-> Analyzing data...")
    activity_spread_days = 0
    is_bunched_activity = False
    
    if len(push_events_timestamps) > 1:
        last_push = datetime.fromisoformat(push_events_timestamps[0].replace('Z', ''))
        first_push = datetime.fromisoformat(push_events_timestamps[-1].replace('Z', ''))
        activity_spread_days = (last_push - first_push).days

        if activity_spread_days < 2 and len(push_events_timestamps) > 10:
            is_bunched_activity = True

    language_percentage = {}
    if total_code_bytes > 0:
        for lang, bytes_of_code in language_stats.items():
            percentage = round((bytes_of_code / total_code_bytes) * 100, 2)
            language_percentage[lang] = percentage

    # --- Build the data-only report first ---
    report = {
        "username": username,
        "verification_status": "Completed",
        "originality_check": {
            "total_public_repos": len(repos_data),
            "original_repo_count": len(original_repos),
            "forked_repo_count": len(repos_data) - len(original_repos),
            "primary_languages": sorted(language_percentage.items(), key=lambda item: item[1], reverse=True),
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
    report["profile_rating"] = calculate_profile_rating(report)
    
    return report

# --- How to Run This File Directly for Testing ---
if __name__ == "__main__":
    
    # --- TEST CASE 1: Your Profile (using the link) ---
    my_input = "https://github.com/kiruthickmurugendran/" # <-- TEST WITH THE LINK
    
    print(f"--- Verifying {my_input} ---")
    username_to_verify = get_username_from_link(my_input)
    
    if username_to_verify:
        my_report = verify_github_profile(username_to_verify)
        import json 
        print("\n--- FINAL REPORT ---")
        print(json.dumps(my_report, indent=2))
        
    # --- TEST CASE 2: A very active, real developer (Linus Torvalds) ---
    # real_user_input = "torvalds" # <-- TEST WITH USERNAME
    
    # print(f"\n--- Verifying {real_user_input} ---")
    # username_to_verify = get_username_from_link(real_user_input)
    
    # if username_to_verify:
    #     report = verify_github_profile(username_to_verify)
    #     import json
    #     print("\n--- FINAL REPORT ---")
    #     print(json.dumps(report, indent=2))