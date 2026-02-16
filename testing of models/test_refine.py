from part2_llm import refine_text_for_verification

# Test with misspelled medical text
text = "trunmeric hads many postiifvies and it cures cacner permananetly"
print(f"Original Text: {text}\n")

refined = refine_text_for_verification(text)

print("\n--- REFINED ---")
print(refined)
