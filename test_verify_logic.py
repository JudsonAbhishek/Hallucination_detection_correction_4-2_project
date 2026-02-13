import os
import time
from part2_llm import fetch_expert_evidence, verify_claim_with_gemini

def test_verification_logic():
    print("Testing Consolidated Expert Evidence...")
    claim = "Gargling with warm salt water can help soothe a sore throat."
    
    start_time = time.time()
    evidence = fetch_expert_evidence(claim)
    end_time = time.time()
    
    print(f"\nTime taken: {end_time - start_time:.2f}s")
    print(f"Evidence Found: {len(evidence)}")
    for e in evidence:
        print(f"- {e[:100]}...")
        
    if evidence and evidence[0] != "RATE_LIMIT_HIT":
        print("\nTesting Verification (Judge) Logic...")
        status, reason, correction = verify_claim_with_gemini(claim, evidence)
        print(f"STATUS: {status}")
        print(f"REASON: {reason}")
    elif evidence and evidence[0] == "RATE_LIMIT_HIT":
        print("\nRATE LIMIT HIT as expected (or unexpectedly). Fallback confirmed.")

if __name__ == "__main__":
    test_verification_logic()
