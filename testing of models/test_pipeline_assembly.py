
# Simulating the pipeline logic from part3_main_pipeline.py
# to verify "final_claims_text" assembly.

claims_data = [
    {
        "claim": "Ginger treats nausea.", 
        "status": "VERIFIED", 
        "correction": None
    },
    {
        "claim": "Advanced AI biosensors measure ginger in hospitals.", 
        "status": "HALLUCINATED", 
        "correction": "There is no evidence that AI biosensors are used for this purpose."
    },
    {
        "claim": "Blockchain is used in ginger products.", 
        "status": "HALLUCINATED", 
        "correction": None # Missing correction case
    }
]

final_claims_text = []

print("--- Simulating Pipeline Logic ---")

for item in claims_data:
    status = item["status"]
    correction_claim = item["correction"]
    claim = item["claim"]
    
    final_text = claim # Default
    
    if status == "VERIFIED":
        final_text = correction_claim if correction_claim else claim
        print(f"VERIFIED -> kept/refined: '{final_text}'")
        
    elif status == "HALLUCINATED":
        # Critical Logic Check: matches part3_main_pipeline.py Step 227
        final_text = correction_claim if correction_claim else "No clinical evidence supports this claim."
        print(f"HALLUCINATED -> corrected: '{final_text}'")
        
    else:
        print(f"OTHER -> kept: '{final_text}'")
        
    final_claims_text.append(final_text)

full_paragraph = " ".join(final_claims_text)
print("\n--- Final Assembled Text ---")
print(full_paragraph)

expected_text = "Ginger treats nausea. There is no evidence that AI biosensors are used for this purpose. No clinical evidence supports this claim."

if full_paragraph == expected_text:
    print("\nSUCCESS: Paragraph assembly logic is correct.")
else:
    print("\nFAILURE: Paragraph assembly logic is incorrect.")
