import os
import requests
import json
import time
import re
from dotenv import load_dotenv

load_dotenv()

# --- PROMPTS ---


# --- PROMPTS ---

PROMPT_MODE_1_QA = (
    "You are a medical AI assistant. Your goal is to provide a grounded medical answer.\n\n"
    "CRITICAL RULE: PRONOUNS ARE FORBIDDEN\n"
    "You MUST NOT use: 'It', 'They', 'These', 'The drug', 'The treatment', 'The condition'.\n"
    "- You MUST repeat the specific name of the medical subject (e.g., 'Metformin', 'Turmeric') in every single sentence.\n"
    "- BAD: 'Turmeric is good. It treats cancer.'\n"
    "- GOOD: 'Turmeric is considered a health supplement. Turmeric has been historically claimed to treat certain conditions.'\n\n"
    "1. REFINE the user's question to be precise and professional.\n"
    "2. GENERATE a comprehensive, fact-based answer to the refined question exactly according to the RULES below.\n\n"
    "{context_str}"
    "3. RULES:\n"
    "   - The ANSWER must be exactly ONE PARAGRAPH.\n"
    "   - NO markdown formatting.\n"
    "   - **NO PRONOUNS**: Every sentence must stand alone with the full subject name. Never use 'It'.\n"
    "   - **SIMPLE TERMS**: Use clear language for laypeople.\n"
    "   - **NO HEDGING**: State facts directly.\n\n"
    "4. FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:\n\n"
    "USER_INPUT: {question}\n\n"
    "REFINED_QUESTION: [The refined question here]\n"
    "ANSWER: [The generated answer here]\n"
)

PROMPT_MODE_2_REFINE = (
    "You are an expert medical editor. Your task is to semantically refine text for clinical fact-checking.\n\n"
    "CRITICAL RULE: PRONOUNS ARE FORBIDDEN\n"
    "You MUST NOT use: 'It', 'They', 'He', 'She', 'This', 'This drug', 'The substance', 'The condition'.\n"
    "- You MUST replace every pronoun with the exact name of the subject.\n"
    "- If the text is about 'Turmeric', every sentence must say 'Turmeric'.\n"
    "- Every single sentence in the REFINED_TEXT must be a complete, self-contained medical fact.\n"
    "- Example: 'Ginger is a root. It treats nausea.' -> 'Ginger is a root. Ginger treats nausea.'\n\n"
    "INSTRUCTIONS:\n"
    "1. Correct spelling and grammar.\n"
    "2. DO NOT change the medical assertions (assertions must remain exactly as the user provided, even if factually wrong).\n\n"
    "FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:\n\n"
    "USER_TEXT: {text}\n\n"
    "REFINED_TEXT: [The refined text with NO PRONOUNS here]\n"
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
    "Respond ONLY with a JSON object:\n"
    "{{ \"status\": \"Verified\" | \"Contradicted\" | \"Insufficient Evidence\" | \"Irrelevant\", \"reason\": \"short explanation\", \"correction\": \"(If Contradicted) The factual correction\" }}\n\n"
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
        "max_tokens": max_tokens
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

import concurrent.futures

def fetch_expert_evidence(claim):
    """
    COUNCIL OF EXPERTS: Calls multiple specialized LLMs in parallel to verify the claim.
    """
    print(f"DEBUG: Convening the Council of Experts for: '{claim[:50]}...'")
    
    # Define the 7 Experts with 100% FREE Models (No Credits Required)
    expert_registry = {
        "Fever Expert": {
            "model": "stepfun/step-3.5-flash:free",
            "prompt": "Evaluate fever-related clinical reasoning using global guidelines. Consider threshold values, duration, and epidemiological context. Reference evidence from: ['WHO', 'CDC']"
        },
        "Symptom Expert": {
            "model": "google/gemma-3-4b-it:free",
            "prompt": "Analyze presenting symptoms and map patterns to possible etiologies. Consider progression, timeline, and co-occurring features. Reference evidence from: ['PubMed', 'MedlinePlus']"
        },
        "Disease Expert": {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "prompt": "Match symptom clusters to candidate diseases and differentials. Prioritize prevalence, age group, region, and comorbidities. Reference evidence from: ['Mayo Clinic', 'Medscape']"
        },
        "Diagnosis Expert": {
            "model": "mistralai/mistral-small-3.1-24b-instruct:free",
            "prompt": "Evaluate diagnostic likelihood using clinical decision rules and flowcharts. Consider Bayesian reasoning, sensitivity/specificity, and red flags. Reference evidence from: ['UpToDate', 'BMJ Best Practice']"
        },
        "Drug Expert": {
            "model": "arcee-ai/trinity-large-preview:free",
            "prompt": "Propose medication and treatment pathways with dosing & contraindications. Consider age, severity, allergies, pregnancy, interactions, and comorbidities. Reference evidence from: ['Drugs.com', 'FDA Database']"
        },
        "Lab Expert": {
            "model": "google/gemma-3-12b-it:free",
            "prompt": "Interpret expected laboratory test findings and deviations. Consider ranges, diagnostic value, and follow-up testing. Reference evidence from: ['LabCorp', 'NIH']"
        },
        "Risk Expert": {
            "model": "meta-llama/llama-3.3-70b-instruct:free",
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
            response = call_llm(config['model'], prompt, max_tokens=200)
            if response and "RATE_LIMIT" not in response:
                return f"[{expert_name}]: {response}"
        except Exception as e:
            print(f"Error calling {expert_name}: {e}")
        return None
 
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(call_single_expert, name): name for name in selected_experts}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                evidence_collected.append(result)
                
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
 
# FREE MODEL FALLBACK LIST
FREE_MODELS = [
    "stepfun/step-3.5-flash:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-12b-it:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "arcee-ai/trinity-large-preview:free"
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

    # Mode 1 Prompt (QA Generation)
    prompt = PROMPT_MODE_1_QA.format(context_str=context_str, question=question)
    
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

def extract_claims_with_llm(text):
    """
    Uses an LLM to split text into atomic, self-contained claims.
    CRITICAL: Resolves pronouns (e.g., "It" -> "Ginger") to ensure context is preserved.
    """
    print(f"DEBUG: Extracting claims with LLM for context-aware resolution...")
    
    prompt = (
        "You are an expert medical text analyst. Your task is to split the following text into individual, atomic claims.\n"
        "RULES:\n"
        "1. SPLIT complex sentences into single facts.\n"
        "2. RESOLVE PRONOUNS: Replace 'It', 'He', 'They', 'This method' with the specific noun they refer to.\n"
        "   - Example Input: 'Ginger is a root. It treats nausea.'\n"
        "   - Example Output: 'Ginger is a root.' <SEP> 'Ginger treats nausea.'\n"
        "3. MAKE CLAIMS SELF-CONTAINED: Every claim MUST have the full medical subject name in it.\n"
        "4. IGNORE questions or conversational filler.\n"
        "5. OUTPUT FORMAT: Join claims with <SEP>.\n\n"
        f"TEXT: \"{text}\"\n\n"
        "ATOMIC CLAIMS (joined by <SEP>):"
    )
    
    response_text = call_free_llm_with_fallback(prompt, max_tokens=600)
    
    if not response_text or response_text == "RATE_LIMIT_HIT":
        print("DEBUG: Extraction LLM failed. Returning original text as single claim.")
        return [text]
        
    # Clean and split
    # Remove any preamble LLM might output
    if ":" in response_text and "<SEP>" not in response_text[:20]:
         # Try to find where the claims start
         parts = response_text.split(":")
         if len(parts) > 1:
             response_text = parts[-1] 
             
    raw_claims = response_text.replace("ATOMIC CLAIMS:", "").replace("\n", "").split("<SEP>")
    
    # Filter empty or too short
    claims = [c.strip() for c in raw_claims if len(c.strip()) > 5]
    
    # Fallback if splitting failed but response exists
    if not claims and len(response_text) > 5:
        return [response_text.strip()]
    elif not claims:
        return [text]
        
    return claims
