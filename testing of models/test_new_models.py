import os
import requests
from dotenv import load_dotenv

load_dotenv()

def call_llm(model):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "MISSING_KEY"
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "MedHallu Test"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5
    }
    
    try:
        r = requests.post(url, headers=headers, json=data, timeout=10)
        if r.status_code == 200:
            return "[OK]"
        else:
            return f"[ERROR {r.status_code}]"
    except Exception as e:
        return f"[EXCEPTION]: {str(e)[:50]}"

def test_new_models():
    print("Testing Alternative Free Models...")
    print("=" * 60)
    
    # New models NOT in the current rate-limited list
    new_models = [
        "openrouter/aurora-alpha",
        "openrouter/free",
        "arcee-ai/trinity-large-preview:free"
    ]
    
    working_models = []
    
    for i, model in enumerate(new_models, 1):
        print(f"{i}. {model}... ", end="", flush=True)
        status = call_llm(model)
        print(status)
        if "[OK]" in status:
            working_models.append(model)
    
    print("\n" + "=" * 60)
    print(f"Working Models: {len(working_models)}/{len(new_models)}")
    if working_models:
        print("\nRECOMMENDED REPLACEMENTS:")
        for m in working_models:
            print(f"  - {m}")
    
    return working_models

if __name__ == "__main__":
    test_new_models()
