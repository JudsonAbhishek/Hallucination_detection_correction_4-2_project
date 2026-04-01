import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()
url = 'https://openrouter.ai/api/v1/models'
res = requests.get(url).json()

free_models = []
for m in res['data']:
    costs = m['pricing']
    if float(costs.get('prompt', 1)) == 0 and float(costs.get('completion', 1)) == 0:
        free_models.append(m['id'])

print(f'Found {len(free_models)} free models.')
test_count = 0
headers = {'Authorization': 'Bearer ' + os.environ.get("OPENROUTER_API_KEY")}

for m in free_models:
    r = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json={'model': m, 'messages': [{'role': 'user', 'content': 'say hi'}]})
    print(f'{m} -> HTTP {r.status_code}')
    test_count += 1
    if test_count > 10: break
