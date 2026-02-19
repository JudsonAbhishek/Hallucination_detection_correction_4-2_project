from part2_llm import fetch_expert_evidence
import part2_llm
from unittest.mock import MagicMock
import time

# Mock the call_llm function to avoid network calls and verify model selection
def mock_call_llm(model, prompt, max_tokens=200):
    return f"Mock response from {model}"

# Apply the mock
part2_llm.call_llm = mock_call_llm

def test_persona(name, claim, expected_model_substring):
    print(f"\n--- TEST: {name} ---")
    print(f"Claim: {claim}")
    
    # We need to make sure the specific expert is called. 
    # fetch_expert_evidence calls ALL experts now.
    results = fetch_expert_evidence(claim)
    
    found = False
    model_correct = False
    
    for e in results:
        # e is formatted as: "[Expert Name]: Response"
        if f"[{name}]" in e:
            found = True
            # The mock response contains the model name
            if expected_model_substring in e:
                model_correct = True
            else:
                print(f"   [!] Wrong model used? Output: {e}")
            break
            
    if found:
        if model_correct:
            print(f"   [SUCCESS] {name} responded using correct model ({expected_model_substring}).")
        else:
            print(f"   [WARNING] {name} responded but model check failed.")
    else:
        print(f"   [FAILURE] {name} did not respond.")
        
    return found and model_correct

print("=== STARTING 7-PERSONA MAPPING INTERCEPT TEST ===")

# 1. Generalist -> stepfun
test_persona("Generalist", "Bones", "stepfun")

# 2. Pharmacologist -> trinity
test_persona("Pharmacologist", "Meds", "trinity")

# 3. Symptom Expert -> stepfun
test_persona("Symptom Expert", "Pain", "stepfun")

# 4. Diagnostic Expert -> gemma
test_persona("Diagnostic Expert", "Tests", "gemma")

# 5. Treatment Expert -> trinity
test_persona("Treatment Expert", "Surgery", "trinity")

# 6. Epidemiologist -> stepfun
test_persona("Epidemiologist", "Stats", "stepfun")

# 7. Lifestyle/Nutrition -> gemma
test_persona("Lifestyle/Nutrition", "Diet", "gemma")

print("\n=== TEST COMPLETE ===")
