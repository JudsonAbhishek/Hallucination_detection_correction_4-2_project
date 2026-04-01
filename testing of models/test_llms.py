import os
import json
from dotenv import load_dotenv
from part2_llm import call_llm

load_dotenv()

models_to_test = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "z-ai/glm-4.5-air:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun/step-3.5-flash:free"
]

print("Testing NEWly updated Models on OpenRouter...\n")

for model in models_to_test:
    print(f"Testing {model}...")
    try:
        response = call_llm(model, "Say hello world.", max_tokens=10)
        if response and response != "RATE_LIMIT_HIT":
            print(f"SUCCESS: {model} -> {response.strip()}")
        elif response == "RATE_LIMIT_HIT":
            print(f"RATE_LIMITED: {model}")
        else:
            print(f"FAILED: {model} -> returned '{response}'")
    except Exception as e:
        print(f"ERROR: {model} -> {str(e)}")
    print("-" * 40)
