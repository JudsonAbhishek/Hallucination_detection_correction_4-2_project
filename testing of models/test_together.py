import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

url = "https://api.together.xyz/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}
data = {
    "model": "meta-llama/Llama-2-7b-chat-hf", # Using a generic small model to test
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 10
}

r = requests.post(url, headers=headers, json=data)
print("Status:", r.status_code)
print("Response:", r.text)
