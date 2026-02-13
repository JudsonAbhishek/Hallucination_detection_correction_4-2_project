
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from rouge_score import rouge_scorer
import torch

def calculate_mrr(relevant_docs, retrieved_docs_list):
    """
    Mean Reciprocal Rank (MRR)
    score = 1/rank of first relevant document
    """
    reciprocal_ranks = []
    for relevant, retrieved in zip(relevant_docs, retrieved_docs_list):
        rank = 0
        found = False
        for i, doc in enumerate(retrieved):
            # Check for partial match or exact match to consider it "relevant"
            # In real world, we check IDs. Here we check text overlap or exact string
            if relevant in doc or doc in relevant: 
                rank = i + 1
                found = True
                break
        
        if found:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            
    return np.mean(reciprocal_ranks)

def calculate_recall_at_k(relevant_docs, retrieved_docs_list, k=3):
    """
    Recall@K: Is the relevant document in the top K results?
    """
    hits = 0
    for relevant, retrieved in zip(relevant_docs, retrieved_docs_list):
        top_k = retrieved[:k]
        found = False
        for doc in top_k:
            if relevant in doc or doc in relevant:
                found = True
                break
        if found:
            hits += 1
    
    return hits / len(relevant_docs) if relevant_docs else 0.0

def calculate_rouge(references, hypotheses):
    """
    Calculate ROUGE-1, ROUGE-2, ROUGE-L
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for ref, hyp in zip(references, hypotheses):
        s = scorer.score(ref, hyp)
        scores['rouge1'].append(s['rouge1'].fmeasure)
        scores['rouge2'].append(s['rouge2'].fmeasure)
        scores['rougeL'].append(s['rougeL'].fmeasure)
        
    return {k: np.mean(v) for k, v in scores.items()}

def calculate_classification_metrics(y_true, y_pred):
    """
    Accuracy, Precision, Recall, F1
    """
    # Ensure binary or multiclass handling
    # Mapping text labels to integers if needed, but sklearn handles strings mostly fine for acc/f1(average)
    
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        "Accuracy": acc,
        "Precision": p,
        "Recall": r,
        "F1 Score": f1
    }
