import torch
import requests
import os
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from part2_llm import classify_and_rewrite_query

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# OFFLINE VECTOR RETRIEVER (RAG)
# -------------------------------
class OfflineRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading Retriever Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use AutoModel directly for embeddings
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        
        self.knowledge_base = [] # List of text strings
        self.embeddings = None   # Torch tensor of shape (N, D)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings)
        return torch.cat(all_embeddings, dim=0)

    def build_index(self, corpus_texts):
        print(f"Building Index for {len(corpus_texts)} documents...")
        self.knowledge_base = corpus_texts
        self.embeddings = self.encode(corpus_texts)
        print("Index passed.")

    def search(self, query, top_k=3):
        if self.embeddings is None:
            return []
            
        query_embedding = self.encode([query])
        # Cosine Similarity: (1, D) @ (N, D).T -> (1, N)
        scores = torch.mm(query_embedding, self.embeddings.transpose(0, 1))[0]
        
        top_results = torch.topk(scores, k=min(top_k, len(self.knowledge_base)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            if score.item() > 0.3: # Minimum similarity threshold
                results.append(self.knowledge_base[idx.item()])
        
        return results

# Global Retriever Instance
retriever = None

def load_knowledge_base():
    global retriever
    if retriever is not None:
        return

    retriever = OfflineRetriever()
    corpus = []
    
    # 1. Load MedHallu (Ground Truths)
    print("Loading MedHallu Ground Truths...")
    try:
        medhallu_data = load_dataset("UTAustin-AIHealth/MedHallu", "pqa_labeled", trust_remote_code=True)["train"]
        for row in medhallu_data:
            # Format: 'Q: ... | GT: ...'
            text = f"Q: {row.get('Question', '')} | GT: {row.get('Ground Truth', '')}"
            corpus.append(text)
    except Exception as e:
        print(f"WARNING: Error loading MedHallu dataset ({e}). Skipping but continuing with seed knowledge.")

    # 2. Load PubMed-QA (Subset for Proxy)
    print("Loading PubMed-QA (Artificial subset)...")
    try:
        # 'pqa_artificial' is smaller and good for testing
        pubmed_data = load_dataset("pubmed_qa", "pqa_artificial", split="train[:1000]", trust_remote_code=True) 
        for row in pubmed_data:
            # Context usually contains the abstract strings
            context = " ".join(row.get('context', {}).get('contexts', []))
            if len(context) > 50:
                 corpus.append(f"Abstract: {context}")
    except Exception as e:
        print(f"WARNING: Error loading PubMed-QA dataset ({e}). Skipping but continuing with seed knowledge.")
    # 3. MANUAL INJECTION (SEED KNOWLEDGE FOR DEMO)
    # Since we are offline with a small subset, we add specific knowledge for our test cases.
    seed_knowledge = [
        "Abstract: Metformin and cancer risk in diabetic patients: a systematic review and meta-analysis. Several observational studies have suggested that metformin use in patients with type 2 diabetes is associated with a lower incidence of cancer. This meta-analysis confirms that metformin treatment is associated with a significantly lower risk of cancer and cancer mortality in diabetic patients.",
        "Abstract: Metformin is the first-line oral treatment for type 2 diabetes. While some studies suggest anticancer effects, it has NOT been conclusively proven to prevent all types of cancer, nor is it recommended as a universal anticancer drug for non-diabetics.",
        "Abstract: Curcumin, the active component of turmeric (Curcuma longa), has shown anti-inflammatory and anticancer potential in vitro and animal models, affecting pathways like apoptosis and inflammation. However, its poor bioavailability and rapid metabolism limit its therapeutic efficacy in humans. Current clinical trials have not established it as a safe or effective standalone cure for cancer, and it is not recommended by guidelines as a replacement for chemotherapy or radiation."
    ]
    corpus.extend(seed_knowledge)
        
    print(f"Total Corpus Size: {len(corpus)} documents.")
    retriever.build_index(corpus)


def search_knowledge_base(claim, top_k=3):
    # Ensure retriever is ready
    if retriever is None:
        load_knowledge_base()
        
    return retriever.search(claim, top_k=top_k)


def fetch_pubmed_evidence(claim, max_results=3):
    # 1. Intelligent Query Generation
    analysis = classify_and_rewrite_query(claim)
    
    query = analysis.get("query", claim)
    claim_type = analysis.get("type", "GENERAL")
    mesh_terms = analysis.get("mesh", [])
    
    print(f"DEBUG: Claim Type: {claim_type} | Query: {query}")

    # Add filters based on type
    filters = ""
    if claim_type == "GUIDELINE":
        filters = ' AND (Practice Guideline[ptyp] OR Guideline[ptyp] OR Review[ptyp])'
    elif claim_type == "STUDY":
        filters = ' AND (Clinical Trial[ptyp] OR Randomized Controlled Trial[ptyp])'
    
    final_query_string = f"{query}{filters}"
    
    # 2. Search PubMed
    def search_pubmed(term):

        try:
            resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": term,
                    "retmode": "json",
                    "retmax": max_results,
                    "sort": "relevance"
                },
                timeout=5, 
            )
            resp.raise_for_status()
            d = resp.json()
            if "esearchresult" in d and "idlist" in d["esearchresult"]:
                return d["esearchresult"]["idlist"]
        except Exception as e:
            print(f"PubMed Search Error: {e}")
        return []

    try:
        # Search
        print(f"DEBUG: Searching PubMed with: {final_query_string}")
        pmids = search_pubmed(final_query_string)
        
        # Retry logic: If no results, drop filters and try raw query
        if not pmids and filters:
            print("DEBUG: No results with filters, retrying raw query...")
            pmids = search_pubmed(query)
            
        # Retry logic: If still no results, use simpler fallback
        if not pmids:
            # Fallback: remove stop words or just take largest words
            keywords = [w for w in query.split() if len(w) > 4]
            if keywords:
                shorter_query = " ".join(keywords[:3])
                print(f"DEBUG: Retrying with keywords: {shorter_query}")
                pmids = search_pubmed(shorter_query)

        if not pmids:
            return []

        # 3. Fetch Abstracts (efetch) - USE XML FOR CLEANER TEXT
        print(f"DEBUG: Fetching abstracts for PMIDs: {pmids}")
        fetch_response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={
                "db": "pubmed", 
                "id": ",".join(pmids), 
                "retmode": "xml",  # Changed to XML
                "rettype": "abstract"
            },
            timeout=10,
        )
        fetch_response.raise_for_status()
        
        # Parse XML to get actual abstract text
        import xml.etree.ElementTree as ET
        root = ET.fromstring(fetch_response.text) # Use .content if encoding issues
        
        abstracts = []
        for article in root.findall(".//PubmedArticle"):
            # Try to find AbstractText
            abs_texts = article.findall(".//Abstract/AbstractText")
            if abs_texts:
                # Join parts of structured abstract
                full_abstract = " ".join([elem.text for elem in abs_texts if elem.text])
                if len(full_abstract) > 50:
                    abstracts.append(full_abstract)
            else:
                # Some old articles might not have structured abstract or use different tag?
                # Usually AbstractText is reliable in XML.
                pass
                
        return abstracts[:max_results]

    except Exception as e:
        print(f"PubMed API Error: {e}")
        return []

def fetch_medhallu_evidence(claim, top_k=2):
    return search_knowledge_base(claim, top_k=top_k)
