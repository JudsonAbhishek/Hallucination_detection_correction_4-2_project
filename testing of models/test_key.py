import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    print("[X] OPENROUTER_API_KEY is not set.")
    exit(1)

print(f"[OK] Key found: {api_key[:10]}... (Total length: {len(api_key)})")

url = "https://openrouter.ai/api/v1/models"
headers = {
    "Authorization": f"Bearer {api_key}"
}

try:
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        print("[OK] API key is valid (Models list retrieved)")
        models = r.json().get("data", [])
        if any(m.get("id") == "google/gemma-3-12b-it:free" for m in models):
            print("[OK] 'google/gemma-3-12b-it:free' found in model list.")
        else:
            print("[!] Warning: 'google/gemma-3-12b-it:free' not found in model list (it might be temporarily unavailable or requires credit).")
    else:
        print(f"[X] API Key Test Failed with status code: {r.status_code}")
        print(f"Response: {r.text}")
except Exception as e:
    print(f"[X] Request Error: {e}")
