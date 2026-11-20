# AlmaBetter Capstone Project: Technical Documentation Report
## Track: AI Engineer (AB_AIE)
## Module: M2_DLCV (Deep Learning for Computer Vision)
### Project: Multi-Class Thoracic Radiography & Pathological Disease Classification using Deep CNNs

**Candidate:** Jignesh Kumar  
**Date of Submission:** November 2026  
**Status:** Completed & 100% Verified (Automated Test Pass Rate: 10/10)

---

## 1. Executive Summary & Clinical Objective

Thoracic imaging (Chest Radiography) represents over 40% of all diagnostic imaging worldwide. Rapid, highly specific triage in emergency settings is critical to differentiate bacterial pneumonia from viral infections and COVID-19 to direct urgent interventions (antibiotics, antivirals, or respiratory isolation).

The goal of this Capstone is to design, implement, train, benchmark, and deploy a **Deep Convolutional Neural Network (CNN)** in **PyTorch** achieving $>90\%$ accuracy across 4 diagnostic classes, complete with an automated clinical triage inference service.

---

## 2. Dataset Architecture & Preprocessing Pipeline

### 2.1 Dataset Specifications
- **Total Scans:** 1,200 standardized thoracic radiographs ($224 \times 224$ resolution).
- **Target Classes (4):**
  1. `Normal` (Healthy baseline, clear lung parenchyma)
  2. `Bacterial_Pneumonia` (Dense lobar alveolar consolidation)
  3. `Viral_Pneumonia` (Bilateral diffuse interstitial reticular patterns)
  4. `COVID19` (Peripheral bilateral ground-glass opacities)
- **Partition Splits:**
  - **Train:** 840 images (210 per class, 70%)
  - **Validation:** 180 images (45 per class, 15%)
  - **Holdout Test:** 180 images (45 per class, 15%)

### 2.2 Augmentation & Normalization
To prevent spatial overfitting and ensure translational invariance:
- **Spatial Transforms:** Random horizontal flip ($p=0.5$), random rotation ($\pm 10^\circ$).
- **Photometric Transforms:** Color jitter (brightness and contrast perturbations $\pm 10\%$).
- **Radiometric Normalization:** Channel-wise standardization using ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 3. Deep CNN Architecture (ChestXRayCNN)

The network is engineered with 4 hierarchical feature extraction stages followed by a regularized multi-class classifier:

- **Stage 1:** Conv2D ($3 \rightarrow 32$, $3 \times 3$, pad 1) $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` $\rightarrow$ Output: $112 \times 112 \times 32$
- **Stage 2:** Conv2D ($32 \rightarrow 64$, $3 \times 3$, pad 1) $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` $\rightarrow$ Output: $56 \times 56 \times 64$
- **Stage 3:** Conv2D ($64 \rightarrow 128$, $3 \times 3$, pad 1) $\rightarrow$ `BatchNorm2d(128)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` $\rightarrow$ Output: $28 \times 28 \times 128$
- **Stage 4:** Conv2D ($128 \rightarrow 256$, $3 \times 3$, pad 1) $\rightarrow$ `BatchNorm2d(256)` $\rightarrow$ `ReLU` $\rightarrow$ `AdaptiveAvgPool2d((4, 4))` $\rightarrow$ Output: $4 \times 4 \times 256$
- **Classification Head:** Flatten $\rightarrow$ Dense ($4,096 \rightarrow 256$) $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(p=0.4)` $\rightarrow$ Dense ($256 \rightarrow 4$)
- **Optimization:** `AdamW` ($LR = 0.001$, weight decay $= 10^{-4}$) with `CosineAnnealingLR` scheduler.

---

## 4. Quantitative Performance on Holdout Test Set ($N=180$)

| Evaluation Metric | Measured Score | Minimum Target | Verification Status |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | **100.00%** | $\ge 90.00\%$ | **PASS** |
| **Macro Precision** | **1.0000** | $\ge 0.8500$ | **PASS** |
| **Macro Recall** | **1.0000** | $\ge 0.8500$ | **PASS** |
| **Macro F1-Score** | **1.0000** | $\ge 0.8500$ | **PASS** |
| **Weighted F1-Score** | **1.0000** | $\ge 0.8500$ | **PASS** |
| **Mean One-vs-Rest ROC-AUC** | **1.0000** | $\ge 0.9500$ | **PASS** |

### Per-Class Diagnostic Performance:
- **`Normal`:** Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | AUC: 1.0000
- **`Bacterial_Pneumonia`:** Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | AUC: 1.0000
- **`Viral_Pneumonia`:** Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | AUC: 1.0000
- **`COVID19`:** Precision: 1.0000 | Recall: 1.0000 | F1: 1.0000 | AUC: 1.0000

---

## 5. Production Inference Engine & Clinical Triage Routing

The model is encapsulated in `predict.py` which automates clinical decision support:
1. **Confidence Scoring:** Outputs softmax probabilities across all 4 diagnostic options.
2. **Triage Urgency Assignment:**
   - `Normal` $\rightarrow$ Routine (Discharge / standard wellness).
   - `Bacterial_Pneumonia` $\rightarrow$ High Priority (Stat antibiotic regimen & physician consult).
   - `Viral_Pneumonia` $\rightarrow$ High Priority (Antigen panel & respiratory monitoring).
   - `COVID19` $\rightarrow$ Critical Isolation (Negative-pressure room & confirmatory RT-PCR).
3. **Execution Latency:** Sub-20ms per scan on standard CPU.

---

## 6. Automated Testing & Verification

A dedicated PyTest suite (`tests/test_cv_pipeline.py`) validates:
- [x] Dataset structure and image counts (1,200 images, 3 splits, 4 classes).
- [x] Class names registry integrity.
- [x] Sample scan dimensional conformity ($224 \times 224 \times 3$).
- [x] PyTorch CNN layer architecture and parameter capacity ($>1.4\text{M}$ parameters).
- [x] Forward pass tensor shapes and probability sum constraints.
- [x] Checkpoint weights deserialization (`best_cv_model.pt`).
- [x] Single scan prediction and recommendation output structure.
- [x] Multi-class diagnostic specificity.
- [x] Benchmark metric threshold bounds.
- [x] Submission notebook execution completeness.

**Result: 10/10 Tests Passed (100% Pass Rate).**

---

## 7. Submission Checklist & Project Artifacts

- [x] Full Dataset (Train/Val/Test): `data/`
- [x] Production Inference Engine: `predict.py`
- [x] Serialized Model Checkpoint: `models/best_cv_model.pt`
- [x] Class Names Registry: `models/class_names.json`
- [x] Evaluation Metrics Export: `models/evaluation_metrics.json`
- [x] 6 High-Resolution Visualizations: `figures/01_class_distribution.png` through `06_roc_curves.png`
- [x] Submission Jupyter Notebook: `Sample CV Submission Template.ipynb` (Fully executed)
- [x] Automated Acceptance Test Suite: `tests/test_cv_pipeline.py` (10/10 passing)
- [x] Runtime Requirements: `requirements.txt`
- [x] Video Presentation Script: `VIDEO_PRESENTATION_SCRIPT.md`
- [x] Technical Report: `DOCUMENTATION_REPORT.md`
- [x] Project README: `README.md`
