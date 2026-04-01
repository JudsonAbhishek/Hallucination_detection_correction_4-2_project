import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# 1. Dataset (Same 14 from the project)
data = [
    {"claim": "Metformin causes cancer.", "label": "Hallucinated", "evidence": "Medical research has thoroughly investigated the claim that metformin causes cancer and found no clinical evidence to support it. This is widely recognized as a medical myth."},
    {"claim": "Vitamin C cures all viral infections.", "label": "Hallucinated", "evidence": "While vitamin C supports the immune system, it is not a cure for all viral infections. High doses do not eliminate viruses like HIV or Hepatitis."},
    {"claim": "Vaccines cause autism.", "label": "Hallucinated", "evidence": "Extensive studies involving millions of children have consistently shown no link between vaccines and autism. The original study suggesting a link was fraudulent and retracted."},
    {"claim": "Drinking bleach cures COVID-19.", "label": "Hallucinated", "evidence": "Drinking or injecting bleach is highly dangerous and does not cure COVID-19. It can lead to severe organ damage and death."},
    {"claim": "Homeopathy is proven to cure stage 4 cancer.", "label": "Hallucinated", "evidence": "There is no clinical evidence that homeopathic treatments can cure cancer. Reliance on homeopathy instead of proven treatments can be fatal."},
    {"claim": "5G towers spread COVID-19.", "label": "Hallucinated", "evidence": "COVID-19 is caused by a virus (SARS-CoV-2) and cannot be spread through electromagnetic waves or mobile networks like 5G."},
    {"claim": "Eating garlic prevents malaria.", "label": "Hallucinated", "evidence": "Garlic has antimicrobial properties, but clinical trials show it has no effect on preventing malaria transmission or infection."},
    
    {"claim": "Metformin treats type 2 diabetes.", "label": "Verified", "evidence": "Metformin is the first-line medication for the treatment of type 2 diabetes, particularly in people who are overweight."},
    {"claim": "Hypertension increases risk of stroke.", "label": "Verified", "evidence": "High blood pressure (hypertension) is a major risk factor for stroke, as it damages blood vessels and can lead to clots or bursts."},
    {"claim": "Insulin is used for type 1 diabetes.", "label": "Verified", "evidence": "People with type 1 diabetes produce little or no insulin and require daily insulin injections to maintain blood glucose levels."},
    {"claim": "Smoking is a risk factor for lung cancer.", "label": "Verified", "evidence": "Smoking is the leading cause of lung cancer and is responsible for approximately 85% of all lung cancer cases."},
    {"claim": "Regular exercise improves cardiovascular health.", "label": "Verified", "evidence": "Consistent physical activity strengthens the heart muscle and improves circulation, reducing the risk of heart disease."},
    {"claim": "Antibiotics are ineffective against viruses.", "label": "Verified", "evidence": "Antibiotics are designed to kill bacteria. They do not work on viral infections such as the common cold, flu, or COVID-19."},
    {"claim": "Statins are used to lower cholesterol.", "label": "Verified", "evidence": "Statins are a class of lipid-lowering medications that have been found to reduce cardiovascular disease in those at high risk."}
]

# Extract text and labels
X_text = [d["claim"] + " " + d["evidence"] for d in data]
y = np.array([1 if d["label"] == "Verified" else 0 for d in data])

# Feature extraction: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(X_text).toarray()

# Train/Val Split (10 train, 4 val)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training on {X_train.shape[0]} samples, Validating on {X_val.shape[0]}")

# Neural Network setup: We'll do partial_fit to capture epoch-by-epoch metrics
mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1, warm_start=True, random_state=42, learning_rate_init=0.01)

history = {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
classes = np.unique(y)
n_epochs = 30

for epoch in range(1, n_epochs + 1):
    mlp.partial_fit(X_train, y_train, classes=classes)
    
    # Train stats
    train_preds = mlp.predict(X_train)
    train_probs = mlp.predict_proba(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    t_loss = log_loss(y_train, train_probs)
    
    # Val stats
    val_preds = mlp.predict(X_val)
    val_probs = mlp.predict_proba(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    v_loss = log_loss(y_val, val_probs)
    
    history["epochs"].append(epoch)
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

print("Training finished! Final Train Acc:", history["train_acc"][-1], "Val Acc:", history["val_acc"][-1])

# Plotting Settings
plt.style.use('seaborn-v0_8-darkgrid')
epochs_range = history["epochs"]

# 1. Loss vs Epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
plt.title('Real Model: Loss vs Epochs (Text Classifier)', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_vs_epochs.png', dpi=150)
plt.close()

# 2. Accuracy vs Epochs
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['train_acc'], 'g-', label='Training Accuracy', linewidth=2)
plt.title('Real Model: Accuracy vs Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_vs_epochs.png', dpi=150)
plt.close()

# 3. Training vs Validation
plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
plt.plot(epochs_range, history['val_loss'], 'r--', label='Validation Loss', linewidth=2)
plt.plot(epochs_range, history['val_acc'], 'orange', label='Validation Acc', linestyle='-.')
plt.title('Real Model: Training vs Validation Curves', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss / Accuracy', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_vs_validation_loss.png', dpi=150)
plt.close()

print("Plots successfully saved utilizing real features and model backpropagation.")
