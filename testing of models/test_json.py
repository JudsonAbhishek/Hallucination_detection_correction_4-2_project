import re
import json

def test_extraction(response_text):
    print(f"--- Processing: {response_text[:50]}... ---")
    try:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            print("SUCCESS:", data)
        else:
            print("NO JSON FOUND")
    except Exception as e:
        print("ERROR:", e)

# Test cases
test_extraction('{"foo": "bar"}')
test_extraction('```json\n{"foo": "bar"}\n```')
test_extraction('Here is the json: {"foo": "bar"} hope it helps')
test_extraction('{"foo": "bar"} with some text after')
