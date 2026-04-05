
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import nltk
from nltk.tokenize import sent_tokenize

# Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except:
    pass

# Load models (same as in part3_main_pipeline.py)
print("Loading models...")
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)
t5_model.eval()

def is_valid_medical_claim(text):
    text = text.strip()
    if len(text.split()) < 3:
        return False
    if text.endswith("?"):
        return False
    return True

def extract_atomic_claims(text):
    raw_sentences = []
    try:
        raw_sentences = [s.strip() for s in sent_tokenize(text) if is_valid_medical_claim(s)]
    except Exception:
        raw_sentences = [s.strip() for s in re.split(r'\.|\n', text) if is_valid_medical_claim(s)]

    if not raw_sentences:
        return [text] if is_valid_medical_claim(text) else []

    final_claims = []
    for sentence in raw_sentences:
        if len(sentence.split()) < 10:
            final_claims.append(sentence)
            continue
        atomized = []
        if t5_model and t5_tokenizer:
            try:
                prompt = f"Break this medical sentence into short, atomic facts. Use <SEP> between facts: {sentence}"
                inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
                with torch.no_grad():
                    outputs = t5_model.generate(**inputs, max_length=256)
                gen_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "<SEP>" in gen_text:
                    atomized = [c.strip() for c in gen_text.split("<SEP>") if len(c.strip()) > 5]
                elif "," in gen_text and len(gen_text.split(",")) > 2:
                    atomized = [c.strip() for c in gen_text.split(",") if len(c.strip()) > 5]
            except: pass
        if atomized:
            final_claims.extend(atomized)
        else:
            final_claims.append(sentence)
    return final_claims

test_text = "Bitter gourd juice can permanently cure diabetes by normalizing blood sugar levels naturally. Herbal remedies can stop the progression of diabetes. Diabetes is a chronic metabolic disorder that cannot be completely cured but can be effectively managed. Diabetes management requires long-term medical supervision, lifestyle modifications, regular monitoring of blood glucose levels, and, in some cases, medication or insulin therapy. Diabetes treatment requires medical supervision to prevent complications such as nerve damage, kidney disease, and cardiovascular problems."

extracted = extract_atomic_claims(test_text)
print("\nExtracted Claims:")
for i, c in enumerate(extracted, 1):
    print(f"{i}. {c}")
