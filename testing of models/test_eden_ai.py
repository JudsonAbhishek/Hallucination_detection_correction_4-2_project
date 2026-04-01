import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

EDEN_AI_API_KEY = os.getenv("EDEN_AI_API_KEY")

url = "https://api.edenai.run/v2/text/chat"
headers = {
    "Authorization": f"Bearer {EDEN_AI_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "providers": "openai",
    "text": "Hello, how are you?",
    "chatbot_global_action": "",
    "previous_history": [],
    "max_tokens": 100
}

r = requests.post(url, headers=headers, json=data)
print("Status:", r.status_code)
print("Response:", r.text)
