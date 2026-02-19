import os
import requests
import json
import time
import re
from dotenv import load_dotenv

load_dotenv()

# Configuration for Expert LLMs (Switched to FREE models due to credit limit)
expert_models = {
    "fever_expert": "stepfun/step-3.5-flash:free",
    "symptom_expert": "arcee-ai/trinity-large-preview:free",
    "disease_expert": "stepfun/step-3.5-flash:free",
    "diagnosis_expert": "openrouter/aurora-alpha",
    "drug_expert": "arcee-ai/trinity-large-preview:free",
    "lab_expert": "stepfun/step-3.5-flash:free",
    "risk_expert": "openrouter/aurora-alpha"
}

expert_prompts = {
    "fever_expert": "Verify fever-related claims using WHO & CDC.",
    "symptom_expert": "Verify symptom-related claims using PubMed & Medline.",
    "disease_expert": "Verify disease mapping using Mayo Clinic & PubMed.",
    "diagnosis_expert": "Verify diagnosis logic using clinical rules.",
    "drug_expert": "Verify treatment using FDA & DrugBank.",
    "lab_expert": "Verify lab interpretations using NIH references.",
    "risk_expert": "Verify risk & red flags using NICE guidelines."
}

def call_llm(model, prompt, max_tokens=200):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("WARNING: OPENROUTER_API_KEY not found. Skipping Expert LLM fallback.")
        return ""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://medhallu.app", # Optional, for OpenRouter rankings
        "X-Title": "MedHallu"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": max_tokens  # Configurable limit
    }

    # 2s Rate Limiting backoff for Free Tier (Safety)
    time.sleep(2)

    try:
        # Reduced timeout to avoid hanging the pipeline too long during fallback
        r = requests.post(url, headers=headers, json=data, timeout=20)
        
        # Immediate 429 Detection
        if r.status_code == 429:
            print(f"CRITICAL: Rate limit hit (429) for {model}. Switching to fallback mode.")
            return "RATE_LIMIT_HIT"
            
        if r.status_code != 200:
            print(f"OpenRouter Error ({model}): {r.status_code} - {r.text}")
            return ""
            
        res = r.json()
        if "choices" not in res:
            return ""
        return res["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"LLM Call Error ({model}): {e}")
        return ""

def generate_medical_answer(question, context=None):
    print(f"DEBUG: Generating medical answer for: {question}")
    
    context_str = ""
    if context:
        context_str = f"CONTEXT: {context}\n"
        
    prompt = (
        "You are a helpful and knowledgeable medical AI assistant. "
        "The user has asked a medical question. Provide a clear, concise, and fact-based answer.\n"
        f"{context_str}"
        "Do not hallucinate or make up facts. If you are unsure, state that.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    
    # Use free model fallback
    try:
        answer = call_free_llm_with_fallback(prompt, max_tokens=400)
        if not answer:
            return "Error: Could not generate an answer at this time (Service Busy)."
        return answer
    except Exception as e:
        print(f"Generation Error: {e}")
        return "Error: An exception occurred during answer generation."

def classify_and_rewrite_query(claim):
    """
    Uses Gemini to:
    1. Classify claim (Guideline vs Study vs General)
    2. Rewrite claim into a keyword-based PubMed query
    3. Suggest MeSH terms
    """
    # Use Free Models for query rewriting
    prompt = (
        "You are an expert search query optimizer for PubMed. Analyze the following medical claim:\n\n"
        f"CLAIM: \"{claim}\"\n\n"
        "TASKS:\n"
        "1. CLASSIFY the claim type:\n"
        "   - 'GUIDELINE': Mentions specific organizations (WHO, AHA, ESC), guidelines, recommendations, or standard of care.\n"
        "   - 'STUDY': Mentions specific trials, p-values, sample sizes, or 'recent study'.\n"
        "   - 'GENERAL': General medical fact or mechanism.\n"
        "2. REWRITE for PubMed Search:\n"
        "   - Strip noise words (e.g., 'according to', 'shows that', 'year 2021').\n"
        "   - Focus on medical entities (Drug, Disease, Outcome).\n"
        "   - RETURN A STRING of 3-6 keywords joined by spaces.\n"
        "3. SUGGEST MeSH Terms (optional, up to 3).\n\n"
        "OUTPUT FORMAT (JSON):\n"
        "{ \"type\": \"GUIDELINE\" | \"STUDY\" | \"GENERAL\", \"query\": \"keyword1 keyword2...\", \"mesh\": [\"Term1\", \"Term2\"] }"
    )
    
    try:
        response_text = call_free_llm_with_fallback(prompt, max_tokens=200)
        if not response_text:
            raise Exception("Empty response from LLM")
            
        # Robust JSON extraction
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            return json.loads(response_text)
            
    except Exception as e:
        print(f"Query Rewrite Error (Free LLM): {e}")

    # HEURISTIC FALLBACK
    import re
    lower_claim = claim.lower()
    
    # 1. Classify
    ctype = "GENERAL"
    if re.search(r"(guideline|recommendation|standard of care|who|aha|esc|acc|nice)", lower_claim):
        ctype = "GUIDELINE"
    elif re.search(r"(study|trial|rct|randomized|cohort|meta-analysis)", lower_claim):
        ctype = "STUDY"
        
    # 2. Rewrite Query (Remove noise)
    noise_patterns = [
        r"according to the", r"based on", r"guidelines?", r"recommendations?", 
        r"who", r"aha", r"esc", r"acc", r"nice", r"20\d\d", # Years
        r"shows that", r"suggests that", r"found that", r"concluded that",
        r"significant", r"associated with", r"treatment should be", 
        r"initiated in", r"adults with", r"confirmed diagnosis of"
    ]
    
    clean_query = lower_claim
    for p in noise_patterns:
        clean_query = re.sub(p, " ", clean_query)
        
    # Simple stopword removal
    stop_words = {"a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were", "be", "been", "that", "this", "it", "by", "from"}
    query_words = [w for w in clean_query.split() if w not in stop_words and len(w) > 2]
    
    final_query = " ".join(query_words[:8])
    
    print(f"DEBUG: Using Heuristic Fallback. Type: {ctype} | Query: {final_query}")
    
    return {
        "type": ctype, 
        "query": final_query,
        "mesh": []
    }

def fetch_expert_evidence(claim):
    """
    CONSOLIDATED EXPERT: Reducse API calls from 4 to 1 per claim.
    """
    print(f"DEBUG: Fetching evidence from Super Medical Expert for: {claim[:50]}...")
    
    # Use the primary working model for the expert knowledge
    model = expert_models.get("disease_expert", "stepfun/step-3.5-flash:free")
    
    prompt = (
        "You are a Super Medical Expert. Verify the following claim using your internal knowledge base "
        "(WHO, CDC, PubMed, FDA, and clinical guidelines).\n\n"
        f"CLAIM: {claim}\n\n"
        "TASKS:\n"
        "1. Identify if a medical disease, drug, or symptom is mentioned.\n"
        "2. Provide short, factual evidence confirming or correcting the claim.\n"
        "3. Cite the likely source (e.g., 'According to WHO guidelines...').\n\n"
        "RESPONSE:"
    )
    
    out = call_llm(model, prompt, max_tokens=300)
    
    if out == "RATE_LIMIT_HIT":
        return ["RATE_LIMIT_HIT"]
        
    if out:
        return [f"[Source: Super Medical Expert] {out}"]
    
    return []

# FREE MODEL FALLBACK LIST
FREE_MODELS = [
    "stepfun/step-3.5-flash:free",
    "arcee-ai/trinity-large-preview:free",
    "openrouter/aurora-alpha",
    "google/gemma-3-4b-it:free",
    "openrouter/free"
]

def call_free_llm_with_fallback(prompt, max_tokens=200):
    """
    Tries multiple free models in sequence.
    Returns content of first success.
    """
    last_rate_limit_hit = False
    
    for model in FREE_MODELS:
        print(f"DEBUG: Active Model -> {model}")
        res = call_llm(model, prompt, max_tokens=max_tokens)
        
        if res == "RATE_LIMIT_HIT":
            print(f"DEBUG: {model} is rate limited. Trying next fallback...")
            last_rate_limit_hit = True
            continue # Try next model!
            
        if res:
            return res
        
        print(f"DEBUG: Model {model} failed. Trying next...")
            
    return "RATE_LIMIT_HIT" if last_rate_limit_hit else ""

def verify_claim_with_gemini(claim, evidence_list):
    # NOW USES OPENROUTER (GPT-4o-mini equivalent) - Balanced Mode
    if not evidence_list:
        return "IRRELEVANT", "No evidence found.", None
    
    # Use Top-3 Evidence Combined
    evidence_text = "\n\n".join(evidence_list[:3])
    
    prompt = (
        "You are a medical fact verification system.\n\n"
        "Given:\n"
        "1. A medical claim\n"
        "2. Retrieved medical evidence\n\n"
        "INSTRUCTIONS:\n"
        "- **Verified**: The evidence *supports* or *strongly aligns* with the claim. (Allow for slight variations in phrasing).\n"
        "- **Hallucinated**: The evidence *directly contradicts* the claim or states the opposite.\n"
        "- **Irrelevant / No Evidence**: The evidence is unrelated or does not contain enough information to judge.\n\n"
        "IMPORTANT:\n"
        "Do not be overly cynical. If the evidence provides a high degree of confidence for the claim, mark it as Verified.\n"
        "Respond with ONLY a JSON object in this format:\n"
        "{ \"status\": \"Verified\" | \"Hallucinated\" | \"Irrelevant\", \"reason\": \"short explanation\", \"correction\": \"corrected claim if needed\" }\n\n"
        f"Claim: {claim}\n\n"
        f"Evidence: {evidence_text}\n"
    )
    
    # Use Free Models as the Judge
    try:
        print(f"DEBUG: verifying with Free Models (Balanced Mode)...")
        response_text = call_free_llm_with_fallback(prompt, max_tokens=300)
        
        if not response_text or response_text == "RATE_LIMIT_HIT":
             print("DEBUG: API Failed (Rate Limit or No Response). Using Deterministic Fallback.")
             return verify_claim_deterministic(claim, evidence_list)

        print(f"DEBUG: Raw LLM Response: {response_text}")

        # Robust JSON extraction
        import re
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        data = {}
        
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
            except:
                data = {}
        
        # If JSON parsing failed, try text parsing logic
        if not data:
            clean_resp = response_text.strip().lower()
            if "not hallucinated" in clean_resp or "verified" in clean_resp:
                data = {"status": "Verified"}
            elif "hallucinated" in clean_resp:
                data = {"status": "Hallucinated"}
            else:
                data = {"status": "Hallucinated"} # Default fallback
            
        status = data.get("status", "Hallucinated")
        
        # Normalize Status
        if "verified" in status.lower():
            status = "VERIFIED"
        elif "hallucinated" in status.lower():
            status = "HALLUCINATED"
        else:
            status = "HALLUCINATED" 
            
        return status, data.get("reason", "No reason provided."), data.get("correction", None)
        
    except Exception as e:
        print(f"Verification Error (Free LLM): {e}")
        print("DEBUG: Exception occurred. Using Deterministic Fallback.")
        return verify_claim_deterministic(claim, evidence_list)

def verify_claim_deterministic(claim, evidence_list):
    """
    Fallback verification when LLMs are down.
    Uses robust keyword overlap ratio.
    """
    if not evidence_list:
        return "IRRELEVANT", "No evidence found.", None
        
    import string
    # Core medical stop words to ignore in matching
    med_stop = {"treats", "causes", "factor", "increases", "decreases", "lower", "standard", "evidence", "confirmed", "found", "study", "shows"}
    
    # Clean punctuation and filter small/noise words
    claim_words = [w.strip(string.punctuation).lower() for w in claim.split()]
    claim_words = [w for w in claim_words if len(w) > 3 and w not in med_stop]
    
    if not claim_words:
        return "VERIFIED", "Claim consists only of general terms, assuming true.", None
        
    evidence_text = " ".join(evidence_list).lower()
    
    found_count = 0
    missing_words = []
    
    for word in claim_words:
        if word in evidence_text:
            found_count += 1
        else:
            missing_words.append(word)
            
    overlap_ratio = found_count / len(claim_words)
    
    print(f"DEBUG: Deterministic Found: {found_count}/{len(claim_words)} (Ratio: {overlap_ratio:.2f})")
    
    # Threshold: Balanced 25% for medical keywords
    if overlap_ratio >= 0.25:
        return "VERIFIED", f"Deterministic: Found {int(overlap_ratio*100)}% of keywords.", None
    else:
        return "HALLUCINATED", f"Deterministic: Missing key terms: {', '.join(missing_words[:2])}", None

def refine_text_for_verification(text):
    """
    Mode 2 Step 1: Semantically correct and refine the input text.
    """
    print("DEBUG: Refining text for verification...")
    prompt = (
        "You are an expert medical editor. Your task is to semantically refine the following text for clarity and correctness "
        "before it is sent for fact-checking. \n"
        "1. Correct any spelling or grammatical errors.\n"
        "2. Ensure medical terms are used correctly.\n"
        "3. Do NOT add new facts. Only clarity and precision.\n"
        "4. DO NOT include conversational filler like 'Okay', 'Here is the refined text', etc.\n"
        "5. FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:\n\n"
        f"USER_TEXT: {text}\n\n"
        "REFINED_TEXT: [The refined text here]\n"
    )
    
    # Use robust fallback
    response_text = call_free_llm_with_fallback(prompt, max_tokens=300)
    
    if not response_text:
        return text
        
    try:
        if "REFINED_TEXT:" in response_text:
            return response_text.split("REFINED_TEXT:")[1].strip()
        return response_text.strip()
    except Exception as e:
        print(f"Refinement Parse Error: {e}")
        return response_text.strip()

def generate_refined_answer_preview(question, context=None):
    """
    Mode 1 Step 1: Refine the question and generate an initial answer.
    Returns JSON: { "refined_question": "...", "generated_answer": "..." }
    """
    print(f"DEBUG: Generating preview for question: {question}")
    
    context_str = ""
    if context:
        context_str = f"CONTEXT: {context}\n"

    prompt = (
        "You are a medical AI assistant. The user wants to verify a question.\n"
        "1. REFINE the user's question to be precise and professional.\n"
        "2. GENERATE a comprehensive, fact-based answer to the refined question.\n"
        f"{context_str}"
        "3. RULES:\n"
        "   - DO NOT include conversational filler like 'Okay', 'Here is the answer', etc.\n"
        "   - The ANSWER must be the direct medical answer.\n"
        "   - NO markdown formatting in the answer (plain text preferred).\n"
        "   - STRUCTURE the answer in exactly TWO PARAGRAPHS.\n"
        "4. FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:\n\n"
        f"USER_INPUT: {question}\n\n"
        "REFINED_QUESTION: [The refined question here]\n"
        "ANSWER: [The generated answer here]\n"
    )
    
    # Use robust fallback with higher token limit for answer generation
    response_text = call_free_llm_with_fallback(prompt, max_tokens=600)
    
    if not response_text:
        return {"refined_question": question, "generated_answer": "Error: All free models failed to respond. Please try again later."}
        
    # Text-based parsing
    refined_q = question
    gen_answer = response_text
    
    try:
        lines = response_text.split('\n')
        q_found = False
        a_found = False
        
        q_text = []
        a_text = []
        
        current_section = None
        
        for line in lines:
            if line.strip().startswith("REFINED_QUESTION:"):
                current_section = "Q"
                q_text.append(line.replace("REFINED_QUESTION:", "").strip())
                continue
            elif line.strip().startswith("ANSWER:"):
                current_section = "A"
                a_text.append(line.replace("ANSWER:", "").strip())
                continue
                
            if current_section == "Q":
                q_text.append(line)
            elif current_section == "A":
                a_text.append(line)
                
        if q_text:
            refined_q = " ".join(q_text).strip()
        if a_text:
            gen_answer = "\n".join(a_text).strip()
            
        return {"refined_question": refined_q, "generated_answer": gen_answer}

    except Exception as e:
        print(f"Text Parse Error in Preview: {e}")
        return {"refined_question": question, "generated_answer": response_text}
