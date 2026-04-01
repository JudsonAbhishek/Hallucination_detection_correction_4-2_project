import os
import requests
import json
import time
import random
import re
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

load_dotenv()

# --- GLOBALS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- OFFLINE MODELS (Flan-T5) ---
t5_tokenizer = None
t5_model = None

try:
    print("Loading Flan-T5 for Atomic Claim Extraction...")
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(DEVICE)
    t5_model.eval()
except Exception as e:
    print(f"WARNING: Flan-T5 loading failed ({e}). Fallback to NLTK/LLM extraction will be active.")

# --- PROMPTS ---


# --- PROMPTS ---

PROMPT_MODE_1_QA = (
    "You are a Senior Medical AI Consultant. Task: Refine a user inquiry and provide a grounded, evidence-based medical answer.\n\n"
    "GUIDELINES:\n"
    "- Provide a comprehensive, professional response.\n"
    "- Address standard benefits and address potential hallucinations directly with scientific consensus.\n"
    "- Use medical terminology correctly.\n\n"
    "OUTPUT FORMAT (STRICT):\n"
    "REFINED_QUESTION: [Professional research-grade version of the input]\n"
    "ANSWER: [A detailed, one paragraph factual answer with 5-6 sentences. Ensure every detected hallucination or high-risk claim in the input is directly clarified with evidence-based corrections.]\n\n"
    "CONTEXT: {context_str}\n"
    "INPUT: {question}\n"
)

PROMPT_MODE_2_REFINE = (
    "You are an expert medical editor. Your task is to semantically refine text for clinical fact-checking.\n\n"
    "CRITICAL RULE: PRONOUNS ARE FORBIDDEN\n"
    "You MUST NOT use: 'It', 'They', 'He', 'She', 'This', 'This drug', 'The substance', 'The condition'.\n"
    "- You MUST replace every pronoun with the exact name of the subject.\n"
    "- If the text is about 'Turmeric', every sentence must say 'Turmeric'.\n"
    "- Every single sentence in the REFINED_TEXT must be a complete, self-contained medical fact.\n"
    "- **NO DUPLICATE CLAIMS:** Do not repeat the same medical assertion multiple times in different ways. Each claim in the refined text must be distinct and unique.\n"
    "- Example: 'Ginger is a root. It treats nausea.' -> 'Ginger is a root. Ginger treats nausea.'\n\n"
    "INSTRUCTIONS:\n"
    "1. Correct spelling and grammar.\n"
    "2. DO NOT change the medical assertions (assertions must remain exactly as the user provided, even if factually wrong).\n"
    "3. Eliminate any redundant or overlapping sentences that state the same fact.\n\n"
    "FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:\n\n"
    "USER_TEXT: {text}\n\n"
    "REFINED_TEXT: [The refined text with NO PRONOUNS and NO REPETITION here]\n"
)


PROMPT_MODE_3_VERIFY = (
    "You are a medical evidence evaluator.\n\n"
    "Given:\n"
    "1. A medical claim\n"
    "2. Retrieved evidence (abstracts, studies, or expert insights)\n\n"
    "INSTRUCTIONS:\n"
    "First, assess RELEVANCE. Does the evidence take a stance on the topic of the claim? If the evidence is about an unrelated condition or a different drug, it is IRRELEVANT.\n\n"
    "Determine the status using this hierarchy:\n\n"
    "- **VERIFIED**: \n"
    "   - Strong, relevant evidence supports the claim.\n"
    "   - OR it is a fundamental safety guideline (e.g., 'See a doctor').\n"
    "- **CONTRADICTED**: \n"
    "   - Relevant evidence directly DISPROVES the claim.\n"
    "   - OR (CRITICAL): The claim makes a high-risk medical assertion (e.g., 'Cures X', 'Prevents Y') but the evidence explicitly states there is NO evidence for such an effect.\n"
    "- **INSUFFICIENT EVIDENCE**: \n"
    "   - The evidence provided does not contain enough information to confirm or deny the claim.\n"
    "- **IRRELEVANT**: \n"
    "   - The evidence provided is about a completely different medical topic (e.g., claim is about 'survival without water' but evidence is about 'diuretics and sodium').\n\n"
    "Respond ONLY with a valid JSON object:\n"
    "{{ \"status\": \"Verified\" | \"Contradicted\" | \"Insufficient Evidence\" | \"Irrelevant\", \"reason\": \"short explanation\", \"correction\": \"(Required if Contradicted) A professional factual correction of the claim based on the evidence provided.\" }}\n\n"
    "Claim: {claim}\n\n"
    "Evidence: {evidence_text}\n"
)


PROMPT_MODE_4_OMNI_SEARCH = (
    "You are the Chief Medical Intelligence Officer. This claim has already been checked by PubMed and a Council of Experts, but no specific evidence was found.\n\n"
    "TASK:\n"
    "Perform a deep retrieval from your internal medical training data, simulating a search across ALL TRUSTED global sources (e.g., Google Scholar, textbooks, international consensus papers, European Medicines Agency, etc.).\n\n"
    "CLAIM: {claim}\n\n"
    "INSTRUCTIONS:\n"
    "1. Provide 2 highly specific factual evidence points that confirm or deny this claim.\n"
    "2. Cite the likely 'Trusted Source' (e.g., 'Source: European Society of Cardiology', 'Source: Harrison's Principles of Internal Medicine').\n"
    "3. If THIS claim is genuinely unproven or unknown globally, state 'GLOBAL_UNKNOWN'.\n\n"
    "OUTPUT FORMAT:\n"
    "- Evidence Point 1 (Source: [Source Type])\n"
    "- Evidence Point 2 (Source: [Source Type])"
)

def call_groq_llm(model, prompt, max_tokens=600):
    if not GROQ_API_KEY:
        return "RATE_LIMIT_HIT"
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2
    }
    
    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        if r.status_code == 413: return "CONTEXT_EXCEEDED"
        if r.status_code == 429: return "RATE_LIMIT_HIT"
        if r.status_code == 200:
            res = r.json()
            return res['choices'][0]['message']['content']
    except Exception as e:
        print(f"Groq Error: {e}")
    return "RATE_LIMIT_HIT"

def call_gemini_native(prompt, max_tokens=1000):
    """Native Google AI Studio (v1beta) for Gemini 2.5 Flash"""
    if not GEMINI_API_KEY:
        return "RATE_LIMIT_HIT"
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.1-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.1
        }
    }
    
    try:
        r = requests.post(url, json=payload, timeout=25)
        if r.status_code == 429: return "RATE_LIMIT_HIT"
        if r.status_code == 200:
            res = r.json()
            if "candidates" in res and len(res["candidates"]) > 0:
                return res["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Gemini Native Error: {e}")
    return "RATE_LIMIT_HIT"

    return "RATE_LIMIT_HIT"



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
        answer = call_free_llm_with_fallback(prompt, max_tokens=600)
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

import concurrent.futures

def fetch_expert_evidence(claim):
    """
    COUNCIL OF EXPERTS: Calls multiple specialized LLMs in parallel to verify the claim.
    """
    print(f"DEBUG: Convening the Council of Experts for: '{claim[:50]}...'")
    
    # Define the 7 Experts (CONFIRMED 2026 FREE MODELS)
    expert_registry = {
        "Fever Expert": {
            "model": "llama-3.1-8b-instant",
            "provider": "groq",
            "prompt": "Evaluate fever-related clinical reasoning using global guidelines. Consider threshold values, duration, and epidemiological context. Reference evidence from: ['WHO', 'CDC']"
        },
        "Symptom Expert": {
            "model": "llama-3.1-70b-versatile",
            "provider": "groq",
            "prompt": "Analyze presenting symptoms and map patterns to possible etiologies. Consider progression, timeline, and co-occurring features. Reference evidence from: ['PubMed', 'MedlinePlus']"
        },
        "Disease Expert": {
            "model": "mixtral-8x7b-32768",
            "provider": "groq",
            "prompt": "Match symptom clusters to candidate diseases and differentials. Prioritize prevalence, age group, region, and comorbidities. Reference evidence from: ['Mayo Clinic', 'Medscape']"
        },
        "Diagnosis Expert": {
            "model": "gemini",
            "provider": "gemini",
            "prompt": "Evaluate diagnostic likelihood using clinical decision rules and flowcharts. Consider Bayesian reasoning, sensitivity/specificity, and red flags. Reference evidence from: ['UpToDate', 'BMJ Best Practice']"
        },
        "Drug Expert": {
            "model": "llama-3.3-70b-versatile",
            "provider": "groq",
            "prompt": "Propose medication and treatment pathways with dosing & contraindications. Consider age, severity, allergies, pregnancy, interactions, and comorbidities. Reference evidence from: ['Drugs.com', 'FDA Database']"
        },
        "Lab Expert": {
            "model": "gemma2-9b-it",
            "provider": "groq",
            "prompt": "Interpret expected laboratory test findings and deviations. Consider ranges, diagnostic value, and follow-up testing. Reference evidence from: ['LabCorp', 'NIH']"
        },
        "Risk Expert": {
            "model": "llama-3.1-8b-instant",
            "provider": "groq",
            "prompt": "Estimate risk severity, complications, and escalation triggers. Include admission criteria, red-flag symptoms, and clinical thresholds. Reference evidence from: ['NICE Guidelines', 'CDC']"
        }
    }


    
    # Select relevant experts: NOW SELECTING ALL 7 as per user request "7 llms and 14 evidences"
    selected_experts = list(expert_registry.keys())
    
    print(f"DEBUG: Convening the Full Council of Experts ({len(selected_experts)} personas)...")
    
    evidence_collected = []
    
    def call_single_expert(expert_name):
        config = expert_registry.get(expert_name)
        if not config: return None
        
        prompt = (
            f"You are a specialized {expert_name}. {config['prompt']}\n"
            f"CLAIM: {claim}\n"
            "TASK: Provide **2 distinct** short factual evidence points (1 sentence each) confirming or debunking the claim.\n"
            "OUTPUT FORMAT:\n"
            "- Point 1\n"
            "- Point 2"
        )
 
        try:
            provider = config.get("provider")
            if provider == "groq":
                response = call_groq_llm(config['model'], prompt, max_tokens=200)
            elif provider == "gemini":
                response = call_gemini_native(prompt, max_tokens=200)
            else:
                response = call_groq_llm("llama-3.1-8b-instant", prompt, max_tokens=200)

                
            if response and "RATE_LIMIT" not in response and "OUT_OF_CREDITS" not in response:
                return f"[{expert_name}]: {response}"
        except Exception as e:


            print(f"Error calling {expert_name}: {e}")
        return None
 
    # Run sequentially (linear execution) to minimize rate limiting 429 errors
    for name in selected_experts:
        result = call_single_expert(name)
        if result:
            evidence_collected.append(result)
        # Add a delay between linear runs to prevent spamming the free API server
        time.sleep(2)
        
    return evidence_collected
 
def fetch_omni_source_evidence(claim):
    """
    FINAL FAIL-SAFE: The Judge searches global knowledge for evidence when experts fail.
    """
    print(f" -> [FAIL-SAFE] Chief Medical Officer is performing global cross-source retrieval...")
    
    prompt = PROMPT_MODE_4_OMNI_SEARCH.format(claim=claim)
    
    # Use the most capable available model for global knowledge
    response = call_free_llm_with_fallback(prompt, max_tokens=300)
    
    if response and "GLOBAL_UNKNOWN" not in response:
        return [f"[Omni-Source]: {response}"]
    return []
 
# FALLBACK ORDER: Gemini -> Groq
def call_free_llm_with_fallback(prompt, max_tokens=600):
    """
    Tries primary providers (Gemini, Groq) to ensure response.
    """
    # 1. Try Gemini (High Accuracy, Stable v1beta)
    print("DEBUG: Attempting Gemini (Primary)...")
    res = call_gemini_native(prompt, max_tokens=max_tokens)
    if res and res not in ["RATE_LIMIT_HIT", "OUT_OF_CREDITS"] and len(res) > 20:
        return res
    
    # 2. Try Groq (Ultra Fast)
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    for m in models:
        print(f"DEBUG: Attempting Groq {m}...")
        res = call_groq_llm(m, prompt, max_tokens=max_tokens)
        if res and res not in ["RATE_LIMIT_HIT", "OUT_OF_CREDITS"]:
            return res
        
    return "ERROR: All API providers (Gemini, Groq) failed or are rate limited. Please wait 60 seconds."




def verify_claim_with_gemini(claim, evidence_list):
    # NOW USES THE COMBINED ENGINE (Gemini/Groq)


    if not evidence_list:
        # NEW FALLBACK: Check if it's common knowledge or a safety guideline
        print("DEBUG: No direct evidence found. Checking for Common Knowledge / Safety Guideline...")
        common_knowledge_status = check_common_knowledge_fallback(claim)
        
        if common_knowledge_status == "VERIFIED":
            return "VERIFIED", "Standard medical consensus / Safety guideline (Common Knowledge).", None
        else:
            return "IRRELEVANT", "No evidence found and not common knowledge.", None
    
    # Use Top-3 Evidence Combined
    evidence_text = "\n\n".join(evidence_list[:3])
    
    # Mode 3 Prompt (Verification)
    prompt = PROMPT_MODE_3_VERIFY.format(claim=claim, evidence_text=evidence_text)
    
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
            
        status_raw = data.get("status", "Insufficient Evidence")
        
        # Normalize Status to Academic Classes
        status_lower = status_raw.lower()
        if "verified" in status_lower:
            status = "VERIFIED"
        elif "contradicted" in status_lower or "hallucinated" in status_lower:
            status = "CONTRADICTED"
        elif "irrelevant" in status_lower:
            status = "IRRELEVANT"
        else:
            status = "INSUFFICIENT_EVIDENCE"
            
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
    
    # RED FLAG CHECK: High-risk words require much higher evidence confidence
    risk_words = {"cure", "cures", "completely", "prevent", "vaccine", "cancer", "diabetes"}
    is_high_risk = any(w in claim_words for w in risk_words)
    
    print(f"DEBUG: Deterministic Found: {found_count}/{len(claim_words)} (Ratio: {overlap_ratio:.2f})")
    
    # Threshold: Balanced 50% for medical keywords, 75% for high-risk claims
    required_threshold = 0.75 if is_high_risk else 0.50
    
    if overlap_ratio >= required_threshold:
        return "VERIFIED", f"Deterministic: Found {int(overlap_ratio*100)}% of keywords.", None
    else:
        status = "HALLUCINATED" if is_high_risk else "INSUFFICIENT_EVIDENCE"
        return status, f"Deterministic: Missing key terms: {', '.join(missing_words[:2])}", None

def check_common_knowledge_fallback(claim):
    """
    Checks if a claim is 'Common Medical Knowledge' or a 'Standard Safety Guideline'
    (e.g. 'Consult your doctor', 'Drink water', 'Exercise is good').
    """
    prompt = (
        "You are a medical safety judge. Determine if the following claim is considered "
        "**COMMON MEDICAL KNOWLEDGE** or a **STANDARD SAFETY GUIDELINE** that does not require specific new studies to prove.\n\n"
        f"CLAIM: \"{claim}\"\n\n"
        "EXAMPLES OF COMMON KNOWLEDGE / SAFETY GUIDELINES (Should be VERIFIED):\n"
        "- \"Patients should consult their doctor before starting new supplements.\"\n"
        "- \"Smoking is harmful to health.\"\n"
        "- \"Regular exercise improves cardiovascular health.\"\n"
        "- \"If symptoms persist, seek medical attention.\"\n\n"
        "EXAMPLES OF SPECIFIC CLAIMS (Should be UNVERIFIED if no evidence):\n"
        "- \"Drug X cures Cancer Y.\"\n"
        "- \"Taking 5000mg of Vitamin C prevents COVID.\"\n\n"
        "TASK:\n"
        "- If the claim is a general safety advice or widely accepted fact, output: **VERIFIED**\n"
        "- If the claim is specific, controversial, or requires data you don't have, output: **UNVERIFIED**\n\n"
        "OUTPUT ONLY ONE WORD: 'VERIFIED' or 'UNVERIFIED'."
    )
    
    # Use Generalist Model (Stepfun/Gemma)
    response = call_free_llm_with_fallback(prompt, max_tokens=50)
    
    if response and "VERIFIED" in response.upper() and "UNVERIFIED" not in response.upper():
        return "VERIFIED"
    return "UNVERIFIED"

def refine_text_for_verification(text):
    """
    Mode 2 Step 1: Semantically correct and refine the input text.
    """
    print("DEBUG: Refining text for verification...")
    # Mode 2 Prompt (Refinement Only)
    prompt = PROMPT_MODE_2_REFINE.format(text=text)
    
    # Use the robust fallback chain to ensure we get a result even if one model is busy
    response_text = call_free_llm_with_fallback(prompt, max_tokens=600)
    
    if "RATE_LIMIT_HIT" in response_text or "ERROR:" in response_text:
        raise Exception("Cloud API Rate Limit hit. Please wait 60 seconds and try again.")

        
    if response_text == "RATE_LIMIT_HIT" or "ERROR:" in response_text:
        raise Exception("Cloud API Rate Limit hit. Please wait 60 seconds and try again.")
        
    # Robust splitting for Mode 2
    try:
        if "REFINED_TEXT:" in response_text:
            clean_text = response_text.split("REFINED_TEXT:")[-1].strip()
        else:
            # Fallback for case variations
            parts = re.split(r"(?i)REFINED_TEXT\s*[:*-]+", response_text)
            clean_text = parts[-1].strip()
        
        return clean_text.strip('"`* \n')
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

    # Mode 1 Prompt (QA Generation)
    prompt = PROMPT_MODE_1_QA.format(context_str=context_str, question=question)
    
    # Use the robust fallback chain for the preview step (Mode 1)
    response_text = call_free_llm_with_fallback(prompt, max_tokens=800)
    
    if "RATE_LIMIT_HIT" in response_text or "ERROR:" in response_text:
        raise Exception("Cloud API Rate Limit hit. Please wait 60 seconds and try again.")

        
    # Robust Regex Parsing
    refined_q = question
    gen_answer = response_text
    
    try:
        # Robust case-insensitive partitioning
        res_upper = response_text.upper()
        
        # Check if BOTH labels exist
        if "REFINED_QUESTION:" in res_upper and "ANSWER:" in res_upper:
            # Split by REFINED_QUESTION
            q_match = re.search(r"(?i)REFINED_QUESTION\s*[:*-]+\s*(.*?)\s*(?=ANSWER:)", response_text, re.DOTALL)
            a_match = re.search(r"(?i)ANSWER\s*[:*-]+\s*(.*)", response_text, re.DOTALL)
            
            if q_match:
                refined_q = q_match.group(1).strip()
            if a_match:
                gen_answer = a_match.group(1).strip()
        
        elif "REFINED_QUESTION:" in res_upper:
            # Only Refined Question label found, answer might be after it or omitted
            parts = re.split(r"(?i)REFINED_QUESTION\s*[:*-]+", response_text)
            content = parts[1].strip()
            # If there's a large chunk of text, maybe it contains both?
            # We'll split by double newline as a heuristic if Answer label is missing
            if "\n\n" in content:
                sub_parts = content.split("\n\n", 1)
                refined_q = sub_parts[0].strip()
                gen_answer = sub_parts[1].strip()
            else:
                refined_q = content
                gen_answer = "No detailed answer generated. Please check your query."

        elif "ANSWER:" in res_upper:
            # Only Answer label found
            parts = re.split(r"(?i)ANSWER\s*[:*-]+", response_text)
            refined_q = question # Fallback to original
            gen_answer = parts[1].strip()
            
        else:
            # NO LABELS FOUND - Use heuristic split if possible
            if "\n\n" in response_text:
                parts = response_text.split("\n\n", 1)
                refined_q = parts[0].strip()
                gen_answer = parts[1].strip()
            else:
                refined_q = question
                gen_answer = response_text

        # FINAL CLEANUP: Ensure NO labels remain in the strings
        refined_q = re.sub(r"(?i)^REFINED_QUESTION\s*[:*-]+\s*", "", refined_q)
        gen_answer = re.sub(r"(?i)^ANSWER\s*[:*-]+\s*", "", gen_answer)
        
        return {
            "refined_question": refined_q.strip('"`* \n'),
            "generated_answer": gen_answer.strip('"`* \n')
        }

    except Exception as e:
        print(f"Text Parse Error in Preview: {e}")
        return {"refined_question": question, "generated_answer": response_text}

def extract_claims_with_llm(text):
    """
    Consolidated function to split text into claims.
    Uses Flan-T5 by default (User request) with fallback to NLTK and LLM.
    """
    return extract_atomic_claims(text)

def extract_atomic_claims(text):
    """
    Uses T5 to split complex text into individual atomic medical claims.
    Reliant on refined input text for pronoun resolution.
    """
    claims = []
    
    # 1. Try Flan-T5 (High Priority)
    if t5_model and t5_tokenizer:
        try:
            # Enhanced prompt for Flan-T5 splitting
            prompt = (
                f"Break the following medical text into a list of short, atomic facts. "
                f"Use <SEP> between each fact.\n"
                f"Text: {text}\n"
                f"Facts:"
            )
            
            inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            
            with torch.no_grad():
                outputs = t5_model.generate(
                    **inputs, 
                    max_length=512, 
                    num_beams=4, 
                    early_stopping=True
                )
                
            generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Robust split
            if "<SEP>" in generated_text:
                claims = [c.strip() for c in generated_text.split("<SEP>") if len(c.strip()) > 5]
            else:
                # Heuristic split by common patterns if <SEP> is missing
                claims = [c.strip() for c in re.split(r'\d+\.|\*|\n|-', generated_text) if len(c.strip()) > 5]
            
            # If T5 returned a summary or truncated significantly, it's not a success
            if len(claims) <= 1 and len(text.split('.')) > 1:
                claims = []
                
        except Exception as e:
            print(f"DEBUG: Flan-T5 generation failed ({e}). Falling back.")

    # 2. Try LLM Splitting (If T5 failed)
    if not claims:
        print("DEBUG: Using LLM for claim extraction (Fallback)...")
        prompt = (
            "You are an expert medical text analyst. Your task is to split the following text into individual, atomic claims.\n"
            "RULES:\n"
            "1. SPLIT complex sentences into single facts.\n"
            "2. RESOLVE PRONOUNS: Replace 'It', 'He', 'They', 'This method' with the specific noun they refer to.\n"
            "3. MAKE CLAIMS SELF-CONTAINED: Every claim MUST have the full medical subject name in it.\n"
            "4. OUTPUT FORMAT: Join claims with <SEP>.\n\n"
            f"TEXT: \"{text}\"\n\n"
            "ATOMIC CLAIMS (joined by <SEP>):"
        )
        
        try:
            response_text = call_free_llm_with_fallback(prompt, max_tokens=600)
            if response_text and "RATE_LIMIT" not in response_text:
                 # Extract after any label LLM might provide
                 if ":" in response_text and "<SEP>" not in response_text[:30]:
                     response_text = response_text.split(":", 1)[-1]
                 claims = [c.strip() for c in response_text.split("<SEP>") if len(c.strip()) > 5]
        except:
            pass

    # 3. Final Fallback: Sentence Tokenization (NLTK)
    if not claims:
        print("DEBUG: Using NLTK for claim extraction (Final Fallback)...")
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            try:
                nltk.data.find('tokenizers/punkt')
            except:
                nltk.download('punkt', quiet=True)
            
            claims = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 10]
        except:
             # Split by period and newline
             claims = [s.strip() for s in re.split(r'\.|\n', text) if len(s.strip()) > 10]
            
    return claims or [text]
