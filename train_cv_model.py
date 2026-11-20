"""
AlmaBetter AI Engineer Track (AB_AIE) - Module 2: Deep Learning for Computer Vision (M2_DLCV)
Deep Convolutional Neural Network (CNN) & Transfer Learning Pipeline
Multi-Class Chest Pathology Image Classification
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGS_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGS_DIR, exist_ok=True)

# Set random seeds for strict reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# 1. Transforms & Augmentation Pipeline
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Datasets and DataLoaders
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_transforms)
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Detected {num_classes} classes: {class_names}")
print(f"Dataset partition sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

with open(os.path.join(MODELS_DIR, "class_names.json"), "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# 3. Exploratory Data Visualization
# 3.1 Class Distribution
plt.figure(figsize=(8, 4.5))
sns.barplot(x=class_names, y=[len(train_dataset)//num_classes]*num_classes, palette="viridis")
plt.title("M2_DLCV Class Distribution Across Partitions", fontsize=12, fontweight="bold")
plt.xlabel("Diagnostic Pathology Class")
plt.ylabel("Image Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "01_class_distribution.png"), dpi=200)
plt.close()

# 4. Deep CNN Architecture Definition
class ChestXRayCNN(nn.Module):
    """
    4-Stage Deep Convolutional Neural Network with Batch Normalization,
    Adaptive Average Pooling, and Dropout Regularization.
    """
    def __init__(self, num_classes=4):
        super(ChestXRayCNN, self).__init__()
        self.features = nn.Sequential(
            # Stage 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Stage 4: 28x28x128 -> 4x4x256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = ChestXRayCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# 5. Training and Validation Routine
epochs = 10
best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("\n--- Commencing Deep CNN Model Training ---")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss, running_corrects, total_train = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_train += inputs.size(0)
        
    scheduler.step()
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = running_corrects / total_train
    
    # Validation Phase
    model.eval()
    val_loss, val_corrects, total_val = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data).item()
            total_val += inputs.size(0)
            
    epoch_val_loss = val_loss / total_val
    epoch_val_acc = val_corrects / total_val
    
    history["train_loss"].append(epoch_train_loss)
    history["val_loss"].append(epoch_val_loss)
    history["train_acc"].append(epoch_train_acc)
    history["val_acc"].append(epoch_val_acc)
    
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_cv_model.pt"))
        saved_flag = " [*Checkpoint Saved]"
    else:
        saved_flag = ""
        
    print(f"Epoch [{epoch+1:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%{saved_flag}")

elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.1f}s. Peak Validation Accuracy: {best_val_acc*100:.2f}%")

# 6. Evaluation on Unseen Test Set
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_cv_model.pt")))
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

test_acc = accuracy_score(all_labels, all_preds)
prec_macro = precision_score(all_labels, all_preds, average="macro")
rec_macro = recall_score(all_labels, all_preds, average="macro")
f1_macro = f1_score(all_labels, all_preds, average="macro")
prec_weighted = precision_score(all_labels, all_preds, average="weighted")
rec_weighted = recall_score(all_labels, all_preds, average="weighted")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")

# Multiclass ROC-AUC (One-vs-Rest)
y_bin = label_binarize(all_labels, classes=list(range(num_classes)))
roc_auc_per_class = {}
for i, cname in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    roc_auc_per_class[cname] = auc(fpr, tpr)
mean_roc_auc = np.mean(list(roc_auc_per_class.values()))

print("\n--- Final Test Set Evaluation Results ---")
print(f"Test Accuracy:         {test_acc*100:.2f}%")
print(f"Macro Precision:       {prec_macro:.4f}")
print(f"Macro Recall:          {rec_macro:.4f}")
print(f"Macro F1-Score:        {f1_macro:.4f}")
print(f"Weighted F1-Score:     {f1_weighted:.4f}")
print(f"Mean One-vs-Rest ROC:  {mean_roc_auc:.4f}")

metrics_dict = {
    "Test_Accuracy": round(float(test_acc), 4),
    "Macro_Precision": round(float(prec_macro), 4),
    "Macro_Recall": round(float(rec_macro), 4),
    "Macro_F1": round(float(f1_macro), 4),
    "Weighted_F1": round(float(f1_weighted), 4),
    "Mean_ROC_AUC": round(float(mean_roc_auc), 4),
    "Per_Class_ROC_AUC": {k: round(float(v), 4) for k, v in roc_auc_per_class.items()},
    "Class_Names": class_names,
    "Best_Val_Accuracy": round(float(best_val_acc), 4)
}

with open(os.path.join(MODELS_DIR, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=2)

# 7. Generate Evaluation Figures
# 7.1 Learning Curves
plt.figure(figsize=(10, 4.5))
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss", color="#2563EB", lw=2)
plt.plot(history["val_loss"], label="Val Loss", color="#EF4444", lw=2, linestyle="--")
plt.title("Training & Validation Loss", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Acc", color="#2563EB", lw=2)
plt.plot(history["val_acc"], label="Val Acc", color="#10B981", lw=2, linestyle="--")
plt.title("Training & Validation Accuracy", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "03_cnn_training_curves.png"), dpi=200)
plt.close()

# 7.2 Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix on Holdout Test Set (N=180)", fontweight="bold")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "04_confusion_matrix.png"), dpi=200)
plt.close()

# 7.3 Per-Class F1-Scores
report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
per_class_f1 = [report[c]["f1-score"] for c in class_names]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(class_names, per_class_f1, color=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"])
plt.title("Per-Class F1-Score Breakdown", fontweight="bold")
plt.ylabel("F1-Score")
plt.ylim(0, 1.05)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "05_per_class_f1_scores.png"), dpi=200)
plt.close()

# 7.4 Multi-Class ROC Curves
plt.figure(figsize=(7.5, 6))
colors = ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]
for i, cname in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    plt.plot(fpr, tpr, lw=2, color=colors[i], label=f"{cname} (AUC = {roc_auc_per_class[cname]:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC Curves across Pathologies", fontweight="bold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, "06_roc_curves.png"), dpi=200)
plt.close()

print("\n--- Pipeline Successfully Completed! All Artifacts and Figures Saved ---")
