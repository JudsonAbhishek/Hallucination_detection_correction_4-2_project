import requests
import os
from dotenv import load_dotenv

load_dotenv()

models = [
    'openrouter/free',
    'google/gemma-3-12b-it:free', 
    'meta-llama/llama-3-8b-instruct:free', 
    'cognitivecomputations/dolphin3.0-r1-mistral-24b:free',
    'deepseek/deepseek-r1:free',
    'qwen/qwen-vl-plus:free',
    'google/gemma-2-9b-it:free'
]

url = 'https://openrouter.ai/api/v1/chat/completions'
headers = {
    'Authorization': f'Bearer {os.environ.get("OPENROUTER_API_KEY")}'
}

for m in models:
    r = requests.post(url, headers=headers, json={'model': m, 'messages': [{'role': 'user', 'content': 'hi'}]})
    print(f'{m}: {r.status_code}')
