from part2_llm import generate_refined_answer_preview
import json

# Test with a question that typically triggers conversational filler
question = "can trunmric cure cancer or nott"
print(f"Testing Question: {question}\n")

result = generate_refined_answer_preview(question)

print("\n--- RESULT ---")
print(json.dumps(result, indent=2))
print(f"\nAnswer Length: {len(result.get('generated_answer', ''))} characters")
