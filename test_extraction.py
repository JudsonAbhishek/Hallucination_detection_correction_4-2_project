
from part3_main_pipeline import extract_atomic_claims

text = "Ginger is a root. It is used to treat nausea. It is also believed to cure cancer."

print(f"Original Text: {text}\n")
print("--- T5 Extraction Output ---")

claims = extract_atomic_claims(text)

for i, claim in enumerate(claims):
    print(f"Claim {i+1}: {claim}")
    
# Check for unresolved pronouns
unresolved = any(claim.lower().startswith("it ") or claim.lower().startswith("he ") for claim in claims)

if unresolved:
    print("\nFAILURE: Pronouns NOT resolved (e.g., 'It is used...').")
else:
    print("\nSUCCESS: Pronouns resolved (e.g., 'Ginger is used...').")
