
from part3_main_pipeline import run_medhallu_pipeline
import json

text = "However, some reports claim that drinking silver-infused water can completely cure diabetes within three days, which has no scientific evidence."
result = run_medhallu_pipeline(None, text)

print("\n--- REPRODUCTION RESULT ---")
print(json.dumps(result, indent=2))
