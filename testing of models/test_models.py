import os
import requests
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

load_dotenv()

def test_local_models():
    print("\n--- Testing Local Models ---")
    try:
        print("1. Loading SBERT (sentence-transformers/all-MiniLM-L6-v2)... ", end="", flush=True)
        AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        print("[OK]")
    except Exception as e:
        print(f"[FAILED]: {e}")

    try:
        print("2. Loading Flan-T5 (google/flan-t5-small)... ", end="", flush=True)
        AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        print("[OK]")
    except Exception as e:
        print(f"[FAILED]: {e}")

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

def test_remote_models():
    print("\n--- Testing OpenRouter Free Models ---")
    models = [
        "stepfun/step-3.5-flash:free",
        "arcee-ai/trinity-large-preview:free",
        "openrouter/aurora-alpha",
        "google/gemma-3-4b-it:free",
        "openrouter/free"
    ]
    
    for i, model in enumerate(models, 3):
        print(f"{i}. {model}... ", end="", flush=True)
        status = call_llm(model)
        print(status)

if __name__ == "__main__":
    print("Starting System Health Check...")
    test_local_models()
    test_remote_models()
    print("\nDone.")
