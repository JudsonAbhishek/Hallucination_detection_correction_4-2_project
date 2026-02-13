import requests
import json
import os

def list_openrouter_models():
    url = "https://openrouter.ai/api/v1/models"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", [])
            print(f"Found {len(models)} models.")
            
            free_models = []
            for m in models:
                id_name = m.get("id")
                pricing = m.get("pricing", {})
                prompt_price = pricing.get("prompt")
                completion_price = pricing.get("completion")
                
                # Check for "free" in ID or zero pricing
                is_free = False
                if "free" in id_name.lower():
                    is_free = True
                elif prompt_price == "0" and completion_price == "0":
                    is_free = True
                    
                if is_free:
                    free_models.append(id_name)
                    
            print("\n--- FREE MODELS ---")
            for fm in free_models:
                print(fm)
                
        else:
            print(f"Error fetching models: {r.status_code}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    list_openrouter_models()
