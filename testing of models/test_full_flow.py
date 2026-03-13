
from part3_main_pipeline import extract_atomic_claims
from part2_llm import refine_text_for_verification

original_text = "Ginger is a root. It treats nausea. It is also believed to cure cancer."

print(f"Original Text: {original_text}\n")

# Step 1: Refine (LLM) - Use the updated prompt logic
print("--- Step 1: Refining Text (LLM) ---")
refined_text = refine_text_for_verification(original_text)
print(f"Refined Text: {refined_text}\n")

# Step 2: Extract (T5) - Now using T5 on the REFINED text
print("--- Step 2: Extracting Claims (T5) ---")
claims = extract_atomic_claims(refined_text)

for i, claim in enumerate(claims):
    print(f"Claim {i+1}: {claim}")
    
# Check for resolved pronouns in T5 output
unresolved = any(claim.lower().startswith("it ") or claim.lower().startswith("he ") for claim in claims)

if unresolved:
    print("\nFAILURE: Pronouns still present in extracted claims.")
else:
    print("\nSUCCESS: Pronouns resolved before extraction.")
