import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv

# Import the existing LLM pipeline
from part2_llm import verify_claim_with_gemini

load_dotenv()

# DEVICE
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 1. Dataset (the same 14 cases from evaluate_system.py)
ground_truth_data = [
    # Hallucinations
    {"claim": "Metformin causes cancer.", "label": "Hallucinated", "evidence": "Medical research has thoroughly investigated the claim that metformin causes cancer and found no clinical evidence to support it. This is widely recognized as a medical myth."},
    {"claim": "Vitamin C cures all viral infections.", "label": "Hallucinated", "evidence": "While vitamin C supports the immune system, it is not a cure for all viral infections. High doses do not eliminate viruses like HIV or Hepatitis."},
    {"claim": "Vaccines cause autism.", "label": "Hallucinated", "evidence": "Extensive studies involving millions of children have consistently shown no link between vaccines and autism. The original study suggesting a link was fraudulent and retracted."},
    {"claim": "Drinking bleach cures COVID-19.", "label": "Hallucinated", "evidence": "Drinking or injecting bleach is highly dangerous and does not cure COVID-19. It can lead to severe organ damage and death."},
    {"claim": "Homeopathy is proven to cure stage 4 cancer.", "label": "Hallucinated", "evidence": "There is no clinical evidence that homeopathic treatments can cure cancer. Reliance on homeopathy instead of proven treatments can be fatal."},
    {"claim": "5G towers spread COVID-19.", "label": "Hallucinated", "evidence": "COVID-19 is caused by a virus (SARS-CoV-2) and cannot be spread through electromagnetic waves or mobile networks like 5G."},
    {"claim": "Eating garlic prevents malaria.", "label": "Hallucinated", "evidence": "Garlic has antimicrobial properties, but clinical trials show it has no effect on preventing malaria transmission or infection."},
    
    # Verified
    {"claim": "Metformin treats type 2 diabetes.", "label": "Verified", "evidence": "Metformin is the first-line medication for the treatment of type 2 diabetes, particularly in people who are overweight."},
    {"claim": "Hypertension increases risk of stroke.", "label": "Verified", "evidence": "High blood pressure (hypertension) is a major risk factor for stroke, as it damages blood vessels and can lead to clots or bursts."},
    {"claim": "Insulin is used for type 1 diabetes.", "label": "Verified", "evidence": "People with type 1 diabetes produce little or no insulin and require daily insulin injections to maintain blood glucose levels."},
    {"claim": "Smoking is a risk factor for lung cancer.", "label": "Verified", "evidence": "Smoking is the leading cause of lung cancer and is responsible for approximately 85% of all lung cancer cases."},
    {"claim": "Regular exercise improves cardiovascular health.", "label": "Verified", "evidence": "Consistent physical activity strengthens the heart muscle and improves circulation, reducing the risk of heart disease."},
    {"claim": "Antibiotics are ineffective against viruses.", "label": "Verified", "evidence": "Antibiotics are designed to kill bacteria. They do not work on viral infections such as the common cold, flu, or COVID-19."},
    {"claim": "Statins are used to lower cholesterol.", "label": "Verified", "evidence": "Statins are a class of lipid-lowering medications that have been found to reduce cardiovascular disease in those at high risk."}
]

# 2. Evaluation Logic for NLI models
def evaluate_nli_model(model_name, data):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE)
    model.eval()
    
    y_true = []
    y_pred = []
    
    # Mapping for NLI models (usually 0: entailment, 1: neutral, 2: contradiction)
    # Check model config for labels
    labels = model.config.id2label
    entail_idx = None
    contra_idx = None
    
    for idx, label in labels.items():
        if 'entail' in label.lower(): entail_idx = int(idx)
        if 'contra' in label.lower(): contra_idx = int(idx)
        
    for item in data:
        claim = item["claim"]
        evidence = item["evidence"]
        true_label = item["label"]
        
        inputs = tokenizer(evidence, claim, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs).item()
            
        # Decision Logic
        if pred_idx == entail_idx:
            pred = "Verified"
        else:
            pred = "Hallucinated" # Neutral or Contradiction -> Hallucinated
            
        y_true.append(true_label)
        y_pred.append(pred)
        
    return calculate_metrics(y_true, y_pred)

# 3. Evaluation Logic for the LLM Pipeline
def evaluate_llm_pipeline(data):
    print("Evaluating LLM-based Pipeline...")
    y_true = []
    y_pred = []
    
    for item in data:
        claim = item["claim"]
        evidence = [item["evidence"]]
        true_label = item["label"]
        
        status, _, _ = verify_claim_with_gemini(claim, evidence)
        
        if status == "VERIFIED":
            pred = "Verified"
        else:
            pred = "Hallucinated"
            
        y_true.append(true_label)
        y_pred.append(pred)
        
    return calculate_metrics(y_true, y_pred)

def calculate_metrics(y_true, y_pred):
    # Convert string labels to binary for metrics calculation
    true_bin = [1 if l == "Verified" else 0 for l in y_true]
    pred_bin = [1 if l == "Verified" else 0 for l in y_pred]
    
    acc = accuracy_score(true_bin, pred_bin)
    p = precision_score(true_bin, pred_bin, zero_division=0)
    r = recall_score(true_bin, pred_bin, zero_division=0)
    f1 = f1_score(true_bin, pred_bin, zero_division=0)
    
    return {
        "Accuracy": acc * 100,
        "Precision": p * 100,
        "Recall": r * 100,
        "F1-Score": f1 * 100
    }

def main():
    results = {}
    
    # Evaluate Models
    results['BERT'] = evaluate_nli_model("cross-encoder/nli-distilroberta-base", ground_truth_data)
    results['RoBERTa'] = evaluate_nli_model("FacebookAI/roberta-large-mnli", ground_truth_data)
    results['ClinicalBERT'] = evaluate_nli_model("pritamdeka/PubMedBERT-MNLI-MedNLI", ground_truth_data)
    results['LLM-based'] = evaluate_llm_pipeline(ground_truth_data)
    
    print("\nFINAL RESULTS:")
    print(json.dumps(results, indent=4))
    
    # Generate Graph - Grouped Bar Chart for better comparison
    models = list(results.keys())
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    # Use a modern, light color palette
    # colors = ['#A0C4FF', '#B9FBC0', '#FFCFD2', '#F1C0E8'] # Light Blue, Light Green, Light Red/Pink, Light Purple
    colors = ['#1F77B4',  # Blue - Accuracy
          '#FF7F0E',  # Orange - Precision
          '#2CA02C',  # Green - Recall
          '#9467BD']  # Purple - F1-Score
    for i, metric in enumerate(metrics):
        offset = width * multiplier
        values = [results[model][metric] for model in models]
        rects = ax.bar(x + offset, values, width, label=metric, color=colors[i])
        # ax.bar_label(rects, padding=3, fmt='%.1f')
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
    ax.set_title('Detailed Performance Comparison: BERT, RoBERTa, ClinicalBERT, LLM-based', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5, models)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 115) # Room for legend
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Graph saved as performance_comparison.png")

if __name__ == "__main__":
    main()
