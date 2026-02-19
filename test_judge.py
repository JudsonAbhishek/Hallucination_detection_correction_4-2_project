from part2_llm import verify_claim_with_gemini
import part2_llm
from unittest.mock import MagicMock

# Test Mode 3 Verification Logic
print("--- Testing Verification Judge (Mode 3) ---")

# Mock the LLM to return VERIFIED for a negative claim
def mock_llm_judge(prompt, max_tokens=300):
    print("\n[PROMPT SENT TO JUDGE]:")
    print(prompt) # Verify the new instructions are in
    return '{"status": "Verified", "reason": "Consistent with medical consensus that fasting is not a permanent cure.", "correction": null}'

original_call = part2_llm.call_free_llm_with_fallback
part2_llm.call_free_llm_with_fallback = mock_llm_judge

# Test Case: Negative Claim related to Diabetes (Simulating the user's issue)
claim = "Intermittent fasting alone cannot permanently cure diabetes."
evidence = ["Intermittent fasting improves insulin sensitivity.", "Studies show weight loss benefits.", "Data on long-term remission is limited."]

status, reason, correction = verify_claim_with_gemini(claim, evidence)

print(f"\nResult: Status='{status}'")
print(f"Reason: {reason}")

if status == "VERIFIED":
    print("✅ SUCCESS: Judge Verified the negative claim.")
else:
    print("❌ FAILURE: Judge did not verify.")

# Restore
part2_llm.call_free_llm_with_fallback = original_call
