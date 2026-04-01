import os
from part2_llm import generate_refined_answer_preview

print("Testing...")
try:
    res = generate_refined_answer_preview("can turmeirc cures cancerr")
    print("Success:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
