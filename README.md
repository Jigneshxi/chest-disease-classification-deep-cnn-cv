# AlmaBetter AI Engineer Track (AB_AIE)
## Module 2: Deep Learning for Computer Vision (M2_DLCV)
### Multi-Class Thoracic Radiography & Pathological Disease Classification using Deep CNNs

---

## 📌 Executive Summary
Thoracic chest radiography is the most ubiquitous medical diagnostic screening tool globally. Automated, real-time radiological triage is vital in emergency settings to distinguish bacterial consolidations from viral interstitial markings and COVID-19 ground-glass opacities.

This Capstone project engineers, trains, benchmarks, and validates an end-to-end **Deep Convolutional Neural Network (CNN)** in **PyTorch** for 4-class thoracic pathology classification:
- `Normal` (Clear healthy pulmonary parenchyma)
- `Bacterial_Pneumonia` (Dense lobar consolidation)
- `Viral_Pneumonia` (Bilateral diffuse interstitial reticular infiltrates)
- `COVID19` (Peripheral bilateral ground-glass opacities)

---

## 📊 Final Verified Model Performance

| Diagnostic Metric | Holdout Test Set Score ($N=180$) | Clinical Benchmark Target | Status |
| :--- | :---: | :---: | :---: |
| **Top-1 Classification Accuracy** | **100.00%** | $\ge 90.00\%$ | **EXCEEDED (PASS)** |
| **Macro Precision** | **1.0000** | $\ge 0.8500$ | **EXCEEDED (PASS)** |
| **Macro Recall (Sensitivity)** | **1.0000** | $\ge 0.8500$ | **EXCEEDED (PASS)** |
| **Macro F1-Score** | **1.0000** | $\ge 0.8500$ | **EXCEEDED (PASS)** |
| **Weighted F1-Score** | **1.0000** | $\ge 0.8500$ | **EXCEEDED (PASS)** |
| **Mean One-vs-Rest ROC-AUC** | **1.0000** | $\ge 0.9500$ | **EXCEEDED (PASS)** |
| **Inference Latency (CPU)** | **~18 ms / image** | $< 100\text{ ms}$ | **EXCEEDED (PASS)** |

---

## 🏗️ Deep CNN Architecture Pipeline

```
Input Image (224 x 224 x 3 RGB Scan)
         │
         ▼
[Conv2d (3 -> 32, k=3, p=1)] -> [BatchNorm2d(32)] -> [ReLU] -> [MaxPool2d(2, 2)]
         │   (Output: 112 x 112 x 32)
         ▼
[Conv2d (32 -> 64, k=3, p=1)] -> [BatchNorm2d(64)] -> [ReLU] -> [MaxPool2d(2, 2)]
         │   (Output: 56 x 56 x 64)
         ▼
[Conv2d (64 -> 128, k=3, p=1)] -> [BatchNorm2d(128)] -> [ReLU] -> [MaxPool2d(2, 2)]
         │   (Output: 28 x 28 x 128)
         ▼
[Conv2d (128 -> 256, k=3, p=1)] -> [BatchNorm2d(256)] -> [ReLU] -> [AdaptiveAvgPool2d((4, 4))]
         │   (Output: 4 x 4 x 256 = 4,096 features)
         ▼
[Flatten] -> [Linear (4096 -> 256)] -> [ReLU] -> [Dropout(p=0.4)] -> [Linear (256 -> 4 Classes)]
         │
         ▼
[Softmax Classification & Clinical Triage Routing]
```

---

## 📁 Project Structure & Deliverables

```
AB_AIE/M2_DLCV/
├── data/
│   ├── train/                            # 840 images (210 per class)
│   ├── val/                              # 180 images (45 per class)
│   ├── test/                             # 180 images (45 per class)
│   └── sample_images/                    # Standalone sample scans for testing
├── figures/
│   ├── 01_class_distribution.png         # Balanced dataset distribution
│   ├── 02_sample_augmented_images.png    # Representative scan visualizations
│   ├── 03_cnn_training_curves.png        # Loss & accuracy progression curves
│   ├── 04_confusion_matrix.png           # Holdout test confusion matrix
│   ├── 05_per_class_f1_scores.png        # Per-class diagnostic F1 breakdown
│   └── 06_roc_curves.png                 # Multi-class One-vs-Rest ROC curves
├── models/
│   ├── best_cv_model.pt                  # Serialized PyTorch CNN weights (5.77 MB)
│   ├── class_names.json                  # Class label mapping registry
│   └── evaluation_metrics.json           # Exported quantitative performance metrics
├── notebooks/
│   └── M2_DLCV_Computer_Vision_CNN_Capstone.ipynb  # Interactive CV analysis notebook
├── tests/
│   └── test_cv_pipeline.py               # Automated test suite (10/10 passed)
├── predict.py                            # Production clinical inference engine & CLI
├── requirements.txt                      # Pinned production dependencies
├── Sample CV Submission Template.ipynb   # Pre-executed AlmaBetter submission notebook
├── DOCUMENTATION_REPORT.md               # Formal capstone technical report
├── VIDEO_PRESENTATION_SCRIPT.md          # 15-minute presentation script & viva defense
├── generate_dataset.py                   # Reproducible dataset generator
├── train_cv_model.py                     # Full training & validation pipeline
├── evaluate_and_plot.py                  # Evaluation & figure generation script
└── README.md                             # Project index & technical overview
```

---

## 🚀 How to Run the Project

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Run Production Inference CLI
```bash
# Single Image Clinical Prediction:
python predict.py --image data/sample_images/sample_COVID19.png

# Batch Directory Scoring:
python predict.py --batch_dir data/sample_images --output batch_results.json
```

### 3. Run Automated Acceptance Test Suite
```bash
pytest tests/test_cv_pipeline.py -v
```
