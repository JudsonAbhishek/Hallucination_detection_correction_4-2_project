
from part2_llm import fetch_expert_evidence
import re

print("\n--- TEST CASE: 7 Experts & 14 Evidences ---")
claim_all = "Green tea consumption affects cardiovascular health, digestion, and sleep patterns."
print(f"Claim: {claim_all}\n")

evidence = fetch_expert_evidence(claim_all)

print(f"\nTotal Experts Consulted: {len(evidence)}")
total_points = 0
for e in evidence:
    print(e)
    # Count bullets or lines
    points = len(re.findall(r"[-•*] ", e))
    if points == 0: points = 1 # Fallback if no bullets
    total_points += points

print(f"\nTotal Evidence Points (Target ~14): {total_points}")

if len(evidence) >= 5:
    print("SUCCESS: Majority of Council convened.")
else:
    print("FAILURE: Council did not convene fully.")
    
if total_points >= 10:
    print("SUCCESS: High volume of evidence collected.")
else:
    print("FAILURE: Low evidence count.")
