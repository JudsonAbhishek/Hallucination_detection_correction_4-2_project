import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

# Testing keys from .env
OR_KEYS = [os.getenv(f"OPENROUTER_API_KEY_{i}") for i in range(1, 6)]
OR_KEYS = [k for k in OR_KEYS if k] # Filter out missing keys
if not OR_KEYS and os.getenv("OPENROUTER_API_KEY"):
    OR_KEYS = [os.getenv("OPENROUTER_API_KEY")]

url = "https://openrouter.ai/api/v1/chat/completions"
model = "google/gemini-2.0-flash-lite-preview-02-05:free" # A known 100% free model

for key in OR_KEYS:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
    r = requests.post(url, headers=headers, json=data)
    print(f"Key {key[:10]}... Status: {r.status_code}")
    if r.status_code == 200:
        print("Success!")
        break
    else:
        print(r.text)
