
from part2_llm import fetch_expert_evidence

print("\n--- TEST CASE 1: Pharmacology Expert Routing ---")
claim_drug = "Taking too much Ibuprofen causes stomach ulcers."
print(f"Claim: {claim_drug}")
evidence_drug = fetch_expert_evidence(claim_drug)
print(f"Expert Response: {evidence_drug}")

if any("Pharmacology Expert" in e for e in evidence_drug):
    print("SUCCESS: Routed to Pharmacology Expert.")
else:
    print("FAILURE: Wrong expert selected.")

print("\n--- TEST CASE 2: Symptom Expert Routing ---")
claim_sym = "High fever is a common symptom of malaria."
print(f"Claim: {claim_sym}")
evidence_sym = fetch_expert_evidence(claim_sym)
print(f"Expert Response: {evidence_sym}")

if any("Symptom Expert" in e for e in evidence_sym):
    print("SUCCESS: Routed to Symptom Expert.")
else:
    print("FAILURE: Wrong expert selected.")
