import sys
import os
import json

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from part2_llm import refine_text_for_verification, generate_refined_answer_preview

def test_preview_logic():
    print("\n--- Testing Preview Logic (Lightweight) ---")
    
    # 1. Mode 1: Generation Preview
    q = "What are the symptoms of flu?"
    print(f"Testing Generation Preview for: '{q}'")
    try:
        res = generate_refined_answer_preview(q)
        print("Result Keys:", res.keys())
        if "refined_question" in res and "generated_answer" in res:
            print("[PASS] Generation Preview structure valid.")
            print(f"Refined Q: {res['refined_question']}")
            print(f"Answer Start: {res['generated_answer'][:50]}...")
        else:
            print("[FAIL] Invalid structure for Generation Preview.")
    except Exception as e:
        print(f"[FAIL] Exception: {e}")

    # 2. Mode 2: Refinement Preview
    text = "paitent has high fevr and caugh."
    print(f"\nTesting Refinement Preview for: '{text}'")
    try:
        res = refine_text_for_verification(text)
        print(f"Refined Text: {res}")
        if res and len(res) > 5 and "fever" in res.lower():
            print("[PASS] Refinement Logic valid (corrected spelling).")
        else:
            print("[FAIL] Refinement failed or returned empty/unchanged.")
    except Exception as e:
        print(f"[FAIL] Exception in Refinement: {e}")

if __name__ == "__main__":
    test_preview_logic()
