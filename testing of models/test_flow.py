import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from part2_llm import refine_text_for_verification, generate_refined_answer_preview
from part3_main_pipeline import run_medhallu_pipeline

def test_mode_1_flow():
    print("\n--- Testing Mode 1 (Ask & Verify) ---")
    question = "Is metformin good for weight loss?"
    
    # Step 1: Preview (Generation)
    print(f"1. Calling Preview for: '{question}'...")
    try:
        preview_data = generate_refined_answer_preview(question)
        print("   [scan] Preview Data received keys:", preview_data.keys())
        refined_q = preview_data.get("refined_question")
        generated_a = preview_data.get("generated_answer")
        
        if not refined_q or not generated_a:
            print("   [FAIL] Missing refined question or answer.")
            return
        print(f"   [PASS] Generated Answer length: {len(generated_a)}")
    except Exception as e:
        print(f"   [FAIL] Exception in Preview: {e}")
        return

    # Step 2: Analyze (Verification)
    # Simulate user sending back the refined info
    print("2. Calling Verification with generated answer...")
    try:
        result = run_medhallu_pipeline(refined_q, generated_a)
        print("   [scan] Pipeline Result Status:", result.get("status"))
        print("   [scan] Final Hallucination Score:", result.get("final_hallucination_score"))
        if result.get("status") in ["PASSED", "FAILED"]:
            print("   [PASS] Mode 1 Flow complete.")
        else:
            print("   [FAIL] Unexpected status.")
    except Exception as e:
        print(f"   [FAIL] Exception in Verification: {e}")

def test_mode_2_flow():
    print("\n--- Testing Mode 2 (Verify Text) ---")
    raw_text = "Metformin is a drug used to treat type 2 diabetes and has shown potential for weight loss."
    
    # Step 1: Preview (Refinement)
    print(f"1. Calling Refinement for text length: {len(raw_text)}...")
    try:
        refined_text = refine_text_for_verification(raw_text)
        print(f"   [scan] Refined Text received. Length: {len(refined_text)}")
        
        if not refined_text or len(refined_text) < 10:
            print("   [FAIL] Refined text seems empty or too short.")
            return
        print("   [PASS] Refinement successful.")
    except Exception as e:
        print(f"   [FAIL] Exception in Refinement: {e}")
        return

    # Step 2: Analyze (Verification)
    print("2. Calling Verification with refined text...")
    try:
        # In Mode 2, question might be empty or implicit
        result = run_medhallu_pipeline("", refined_text)
        print("   [scan] Pipeline Result Status:", result.get("status"))
        if result.get("status") in ["PASSED", "FAILED"]:
            print("   [PASS] Mode 2 Flow complete.")
        else:
            print("   [FAIL] Unexpected status.")
    except Exception as e:
        print(f"   [FAIL] Exception in Verification: {e}")

if __name__ == "__main__":
    test_mode_1_flow()
    test_mode_2_flow()
