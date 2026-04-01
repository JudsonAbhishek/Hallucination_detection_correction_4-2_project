import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Test Groq
print("Testing Groq...")
groq_url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
data = {"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
r = requests.post(groq_url, headers=headers, json=data)
print("Groq Status:", r.status_code)
if r.status_code != 200: print(r.text)

# Test Gemini
print("\nTesting Gemini...")
gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
headers = {"Content-Type": "application/json"}
payload = {"contents": [{"parts": [{"text": "hi"}]}]}
r = requests.post(gemini_url, headers=headers, json=payload)
print("Gemini Status:", r.status_code)
if r.status_code != 200: print(r.text)
