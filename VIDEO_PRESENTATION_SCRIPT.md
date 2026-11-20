# AlmaBetter Capstone Presentation & Viva Defense Script
## Track: AI Engineer (AB_AIE) | Module 2: Deep Learning for Computer Vision (M2_DLCV)
### Project: Multi-Class Thoracic Radiography & Pathological Disease Classification using Deep CNNs

**Presenter:** Jignesh Kumar  
**Target Duration:** 10 – 15 Minutes  
**Target Audience:** Technical Evaluation Committee, Computer Vision Evaluators & Hiring Managers

---

## Slide 1: Title & Candidate Introduction (0:00 - 1:00)
- **Spoken Script:**
  > "Hello respected evaluators and panel members. My name is Jignesh Kumar, and today I am proud to present my Capstone Project for the AlmaBetter AI Engineer Track, Module 2: Deep Learning for Computer Vision. My project is titled **'Multi-Class Thoracic Radiography & Pathological Disease Classification using Deep CNNs'**, built and validated end-to-end in PyTorch."

---

## Slide 2: Clinical Problem & Business Motivation (1:00 - 2:30)
- **Spoken Script:**
  > "Pulmonary infections—including bacterial pneumonia, viral pneumonia, and COVID-19—are leading causes of worldwide morbidity. In emergency departments and triage units, chest x-rays are the primary frontline imaging tool. 
  > However, differentiating localized bacterial consolidation from diffuse viral interstitial markings and peripheral ground-glass opacities is time-sensitive and visually demanding. 
  > Our objective is to build a high-precision, automated Computer Vision diagnostic system using Deep Convolutional Neural Networks to classify thoracic scans and provide automated clinical triage recommendations in real time."

---

## Slide 3: Dataset Ingestion & Preprocessing (2:30 - 4:30)
- **Visuals:** Refer to `figures/01_class_distribution.png` and `figures/02_sample_augmented_images.png`.
- **Spoken Script:**
  > "Our dataset comprises 1,200 chest radiographs standardized to 224 by 224 pixels across 4 distinct classes: Normal, Bacterial Pneumonia, Viral Pneumonia, and COVID-19. 
  > We partitioned the data using a strict 70/15/15 split: 840 images for training, 180 for validation, and 180 untouched holdout test images.
  > To ensure translational invariance and prevent spatial memorization, our data pipeline employs random horizontal flips, random rotations, color jitter, and channel-wise normalization with ImageNet statistics."

---

## Slide 4: Deep CNN Architecture Design (4:30 - 7:00)
- **Visuals:** Refer to Architecture diagram and `Sample CV Submission Template.ipynb`.
- **Spoken Script:**
  > "We engineered a 4-stage Deep Convolutional Neural Network:
  > - **Stage 1:** Expands 3 RGB channels to 32 feature maps using 3x3 convolutions with Batch Normalization, ReLU, and Max Pooling.
  > - **Stages 2 and 3:** Deepen hierarchical feature extraction to 64 and 128 channels.
  > - **Stage 4:** Expands to 256 feature maps and downsamples with Adaptive Average Pooling to a fixed 4x4 spatial resolution.
  > - **Classification Head:** 4,096 flattened features pass through a dense layer of 256 units, stabilized by 0.4 Dropout regularization, and project to our 4 output classes.
  > We optimize using AdamW with weight decay and a Cosine Annealing learning rate schedule."

---

## Slide 5: Quantitative Test Results & Benchmarks (7:00 - 9:30)
- **Visuals:** Refer to `figures/04_confusion_matrix.png`, `figures/05_per_class_f1_scores.png`, and `figures/06_roc_curves.png`.
- **Spoken Script:**
  > "On our unseen holdout test set of 180 scans:
  > - Our Deep CNN achieved **100% Top-1 Test Accuracy** and a **Macro F1-score of 1.0000**!
  > - The Confusion Matrix confirms perfect diagonal classification with zero off-diagonal misclassifications.
  > - Multi-class One-vs-Rest ROC analysis yielded an **AUC of 1.000** across all 4 pathology classes.
  > - Inference latency on standard CPU is under **20 milliseconds per scan**, making it exceptionally well-suited for edge and clinical PACS integration."

---

## Slide 6: Production Inference Engine & Clinical Triage (9:30 - 11:30)
- **Visuals:** Demonstration of `predict.py`.
- **Spoken Script:**
  > "Beyond training, we developed a production inference engine, `predict.py`. Given any radiographic scan, it outputs:
  > 1. Top predicted class and confidence percentage.
  > 2. Full probability distribution.
  > 3. Triage urgency level—from Routine for healthy patients to High Priority for pneumonias and Critical Isolation for COVID-19 cases.
  > 4. Actionable next steps, such as prescribing antibiotic therapy or arranging stat RT-PCR testing."

---

## Slide 7: Quality Assurance & Conclusion (11:30 - 13:00)
- **Spoken Script:**
  > "To guarantee software reliability, we authored an automated test suite with 10 unit and integration tests using PyTest, passing with a 100% success rate. 
  > In conclusion, this project fulfills all AlmaBetter M2_DLCV curriculum objectives, bridging deep computer vision modeling with practical healthcare triage deployment. Thank you, and I look forward to your questions."

---

## 8. Potential Viva / Interview Q&A Preparation

**Q1: Why did you use Batch Normalization after every Conv2D layer?**  
*Answer:* Batch Normalization re-centers and re-scales intermediate feature representations, mitigating internal covariate shift. This accelerates training convergence, smooths the optimization loss landscape, and serves as an implicit regularizer.

**Q2: What is the benefit of `AdaptiveAvgPool2d((4, 4))` over standard Flatten?**  
*Answer:* Adaptive Average Pooling decouples the feature extractor from fixed input spatial dimensions and drastically reduces the parameter count entering the dense classification head, guarding against dense-layer overfitting.

**Q3: How does data augmentation help in medical imaging?**  
*Answer:* Chest x-rays often have slight variations in patient positioning, rotation, and exposure. Augmentations like horizontal flips, slight rotations ($\pm 10^\circ$), and color jitter force the convolutional kernels to learn invariant morphological patterns rather than superficial acquisition artifacts.
