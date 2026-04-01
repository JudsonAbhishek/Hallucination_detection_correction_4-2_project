import os
import requests
import json
import time
from dotenv import load_dotenv

load_dotenv()

def call_llm(model, prompt, max_tokens=100):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "ERROR: No API Key"

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "MedHallu-Diagnostic"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        if r.status_code == 429:
            return "RATE_LIMIT"
        if r.status_code != 200:
            return f"ERROR: {r.status_code} - {r.text[:100]}"
            
        res = r.json()
        if "choices" in res:
            return "ALIVE"
        return "ERROR: Malformed Response"
    except Exception as e:
        return f"ERROR: {str(e)}"

FREE_MODELS = [
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "google/gemma-3-4b-it:free",
    "openrouter/free"
]

EXPERT_MODELS = {
    "Fever Expert": "deepseek/deepseek-chat",
    "Symptom Expert": "qwen/qwen-2.5-7b-instruct",
    "Disease Expert": "meta-llama/llama-3-8b-instruct",
    "Diagnosis Expert": "mistralai/mistral-7b-instruct",
    "Drug Expert": "mistralai/mistral-nemo",
    "Lab Expert": "openai/gpt-4o-mini",
    "Risk Expert": "anthropic/claude-3.5-haiku"
}

def diagnostic():
    print("=== MedHallu LLM Diagnostic Tool ===\n")
    print(f"API Key: {'Found' if os.getenv('OPENROUTER_API_KEY') else 'MISSING'}\n")

    results = []

    print("Testing FREE Fallback Models...")
    for model in FREE_MODELS:
        print(f"  - Testing {model}...", end="", flush=True)
        status = call_llm(model, "Say hello.")
        print(f" [{status}]")
        results.append((model, status))
        time.sleep(1) # Safety gap

    print("\nTesting Council of Expert Models...")
    for expert, model in EXPERT_MODELS.items():
        print(f"  - Testing {expert} ({model})...", end="", flush=True)
        status = call_llm(model, "State 'alive'.")
        print(f" [{status}]")
        results.append((f"{expert} ({model})", status))
        time.sleep(1)

    print("\n" + "="*40)
    print("FINAL SUMMARY")
    print("="*40)
    working = [m for m, s in results if s == "ALIVE"]
    failed = [m for m, s in results if s != "ALIVE"]
    
    print(f"TOTAL MODELS: {len(results)}")
    print(f"WORKING: {len(working)}")
    print(f"FAILED/RATE-LIMITED: {len(failed)}")
    
    if failed:
        print("\nFailed Models Log:")
        for m, s in results:
            if s != "ALIVE":
                print(f"  [!] {m}: {s}")

if __name__ == "__main__":
    diagnostic()
