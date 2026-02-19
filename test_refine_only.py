
from part2_llm import refine_text_for_verification

print("\n--- TEST CASE: Refinement Only (No Correction) ---")
false_text = "Garlic is a plant. It completely cures AIDS and restores dead cells."

print(f"Original Text: {false_text}\n")

refined_text = refine_text_for_verification(false_text)
print(f"Refined Text: {refined_text}\n")

# Logic Check: The refined text MUST still contain the false claim.
if "cures AIDS" in refined_text or "treats AIDS" in refined_text:
    print("SUCCESS: False claim preserved (Refinement only).")
else:
    print("FAILURE: The LLM corrected the fact prematurely!")
