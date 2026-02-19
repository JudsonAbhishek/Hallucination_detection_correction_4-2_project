
from part2_llm import verify_claim_with_gemini
import json

print("\n--- TEST CASE: Semantic Equivalence (Green Tea) ---")
claim_tea = "Green tea improves cardiovascular health."
evidence_tea = ["Several clinical studies suggest that regular consumption of green tea is associated with improved cardiovascular markers."]

status, reason, correction = verify_claim_with_gemini(claim_tea, evidence_tea)
print(f"Status: {status}")
print(f"Reason: {reason}")
if correction:
    print(f"Correction: '{correction}'")

if status == "VERIFIED":
    print("SUCCESS: Semantic match confirmed (Health == Markers).")
else:
    print("FAILURE: Semantic match failed.")
