from part2_llm import check_common_knowledge_fallback, verify_claim_with_gemini
from unittest.mock import MagicMock
import part2_llm

# Mock the LLM to test the fallback logic without hitting the API (or hit it partially?)
# Let's hit the actual definition via mock to simulate 'VERIFIED' response first

print("--- Testing Common Knowledge Fallback Logic ---")

# 1. Test the function directly with a mock
original_call = part2_llm.call_free_llm_with_fallback

def mock_llm_verified(prompt, max_tokens=200):
    return "VERIFIED"

part2_llm.call_free_llm_with_fallback = mock_llm_verified
status = check_common_knowledge_fallback("Consult your doctor")
print(f"Test 1 (Mock Verified): Expected 'VERIFIED', Got '{status}' -> {'PASS' if status == 'VERIFIED' else 'FAIL'}")

def mock_llm_unverified(prompt, max_tokens=200):
    return "UNVERIFIED"

part2_llm.call_free_llm_with_fallback = mock_llm_unverified
status = check_common_knowledge_fallback("Drug X cures everything")
print(f"Test 2 (Mock Unverified): Expected 'UNVERIFIED', Got '{status}' -> {'PASS' if status == 'UNVERIFIED' else 'FAIL'}")

# 2. Test integration in verify_claim_with_gemini (No Evidence case)
part2_llm.call_free_llm_with_fallback = mock_llm_verified # Assume it's common knowledge
status, reason, correction = verify_claim_with_gemini("Consult your doctor", []) # Empty evidence
print(f"Test 3 (Integration - Verified): Expected 'VERIFIED', Got '{status}' -> {'PASS' if status == 'VERIFIED' else 'FAIL'}")
print(f"   Reason: {reason}")

part2_llm.call_free_llm_with_fallback = mock_llm_unverified # Assume it's specific claim
status, reason, correction = verify_claim_with_gemini("Drug X cures everything", []) # Empty evidence
print(f"Test 4 (Integration - Unverified): Expected 'IRRELEVANT', Got '{status}' -> {'PASS' if status == 'IRRELEVANT' else 'FAIL'}")
print(f"   Reason: {reason}")


# Restore
part2_llm.call_free_llm_with_fallback = original_call
print("\n--- Test Complete ---")
