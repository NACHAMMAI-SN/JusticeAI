import os
os.makedirs('models', exist_ok=True)

from dl_training_pipeline import run_training_pipeline, TrainingConfig
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from datasets import load_dataset
import random

print("Loading real legal dataset from HuggingFace...")
# Load LEDGAR - real legal contract provisions dataset
dataset = load_dataset("lex_glue", "ledgar", trust_remote_code=True)

print("Dataset loaded! Processing...")
train_data = dataset["train"]

# Get all unique labels
all_labels = list(set(train_data["label"]))
print(f"Total unique legal categories: {len(all_labels)}")

# Pick 3 most common categories and map to our 3 classes
# Class 0 = personal/family type
# Class 1 = business/criminal type  
# Class 2 = consultation type
label_counts = {}
for item in train_data:
    l = item["label"]
    label_counts[l] = label_counts.get(l, 0) + 1

# Sort by frequency, pick top 3
top3 = sorted(label_counts, key=label_counts.get, reverse=True)[:3]
print(f"Using top 3 categories: {top3}")

# Filter dataset to only top 3 labels
label_map = {top3[0]: 0, top3[1]: 1, top3[2]: 2}

texts = []
labels = []

# Get 200 samples per class = 600 total
class_counts = {0: 0, 1: 0, 2: 0}
max_per_class = 200

for item in train_data:
    if item["label"] in label_map:
        mapped = label_map[item["label"]]
        if class_counts[mapped] < max_per_class:
            texts.append(item["text"][:512])
            labels.append(mapped)
            class_counts[mapped] += 1
    if all(v >= max_per_class for v in class_counts.values()):
        break

# Shuffle
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)
texts = list(texts)
labels = list(labels)

print(f"Total samples: {len(texts)}")
print(f"Class distribution: {class_counts}")

# DistilBERT classifier
class LightLegalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)
        )
    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)

print("Loading DistilBERT model...")
model = LightLegalClassifier()

print("Starting training for 5 epochs...")
config = TrainingConfig(epochs=5, batch_size=16, lr=3e-5)
history = run_training_pipeline(
    model, texts, labels,
    'models/intent_classifier.pt', config
)

print("\n Training done!")
print("Best epoch:", history.get('best_epoch'))
print("Best val loss:", history.get('best_val_loss'))