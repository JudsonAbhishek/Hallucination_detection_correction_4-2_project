
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
print("Loading models...")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)
t5_model.eval()

def test_flan_division(text, prompt_type):
    if prompt_type == 1:
        prompt = (
            f"Break the following medical text into a list of short, atomic facts. "
            f"Use <SEP> between each fact.\n"
            f"Text: {text}\n"
            f"Facts:"
        )
    elif prompt_type == 2:
        prompt = (
            f"List all distinct medical facts in the text below. Separate them with <SEP>.\n"
            f"Text: {text}"
        )
    elif prompt_type == 3:
        prompt = (
            f"Split this paragraph into medical claims. Output format: claim1 <SEP> claim2 <SEP> ...\n"
            f"Paragraph: {text}"
        )

    inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs, 
            max_length=512, 
            num_beams=4, 
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        
    generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

test_text = "Bitter gourd juice can permanently cure diabetes by normalizing blood sugar levels naturally. Herbal remedies can stop the progression of diabetes. Diabetes is a chronic metabolic disorder that cannot be completely cured but can be effectively managed. Diabetes management requires long-term medical supervision, lifestyle modifications, regular monitoring of blood glucose levels, and, in some cases, medication or insulin therapy. Diabetes treatment requires medical supervision to prevent complications such as nerve damage, kidney disease, and cardiovascular problems."

print(f"Prompt 1 Results:\n{test_flan_division(test_text, 1)}\n")
print(f"Prompt 2 Results:\n{test_flan_division(test_text, 2)}\n")
print(f"Prompt 3 Results:\n{test_flan_division(test_text, 3)}\n")
