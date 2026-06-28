# 🧠 NeuroSense: Stacked Multimodal Parkinson's Disease Detection

An AI-powered multimodal machine learning system for the early detection of Parkinson's Disease using **handwriting** and **speech biomarkers**. The project combines independent machine learning models through a **stacked ensemble architecture** to improve diagnostic reliability while providing a severity-based risk assessment.

---

## 📖 Overview

Parkinson's Disease (PD) is a progressive neurodegenerative disorder that primarily affects movement, speech, and motor coordination. Traditional diagnosis relies heavily on clinical expertise and may delay early intervention.

NeuroSense addresses this challenge by combining two complementary biomarkers:

- ✍️ Handwriting analysis (motor impairment)
- 🎤 Speech analysis (voice impairment)

Each modality is processed independently using Support Vector Machine (SVM) classifiers. Their probability outputs are then combined using a **Stacked SVM Meta-Classifier**, allowing the system to learn how much importance should be given to each modality instead of relying on simple averaging.

To improve robustness, the training pipeline incorporates:

- SMOTE for handling class imbalance
- Stratified 5-Fold Cross Validation
- Feature-Level Fusion
- Probability-Based Severity Scoring

This multimodal approach provides more reliable predictions than using handwriting or speech independently.

---

# ✨ Features

- 🧠 **Multimodal Analysis**
  - Combines handwriting and speech biomarkers for improved Parkinson's Disease detection.

- ✍️ **Handwriting Analysis**
  - Uses spiral and meander handwriting datasets to identify motor impairments associated with Parkinson's Disease.

- 🎤 **Speech Analysis**
  - Uses acoustic voice features to capture speech abnormalities caused by Parkinson's Disease.

- ⚙️ **Independent Machine Learning Models**
  - Trains separate Support Vector Machine (SVM) models for handwriting and speech data.

- 🔗 **Stacked Ensemble Learning**
  - Combines probability outputs from individual SVM models using a Stacked SVM Meta-Classifier.

- ⚖️ **Class Imbalance Handling**
  - Uses SMOTE (Synthetic Minority Oversampling Technique) during training to improve minority class learning.

- 📊 **Robust Model Evaluation**
  - Uses Stratified 5-Fold Cross Validation for reliable performance estimation.

- 📈 **Performance Metrics**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - Confusion Matrix
  - Precision-Recall Curve

- 🚦 **Severity Prediction**
  - Provides probability-based Parkinson's Disease severity levels:
    - Healthy
    - Mild
    - Moderate
    - Severe

    ---

# 📂 Project Structure

```
NeuroSense-Parkinsons-Detection
│
├── graphs/                 # Performance graphs and evaluation plots
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code
│
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT License
└── .gitignore              # Files ignored by Git
```

---

# 🏗️ System Architecture

```text
                        Input Datasets
                              │
         ┌────────────────────┴────────────────────┐
         │                                         │
         │                                         │
 Speech Dataset                          Handwriting Dataset
         │                                         │
         ▼                                         ▼
Speech Preprocessing               Handwriting Preprocessing
         │                                         │
         ▼                                         ▼
 Speech Feature Extraction          Handwriting Feature Extraction
         │                                         │
         ▼                                         ▼
     Speech SVM                         Handwriting SVM
         │                                         │
         └──────────────┬──────────────────────────┘
                        │
                        ▼
           Stacked SVM Meta Classifier
                        │
                        ▼
          Probability-Based Prediction
                        │
                        ▼
          Severity Risk Assessment
      Healthy • Mild • Moderate • Severe
```
---

# 🛠️ Tech Stack

### Programming Language
- Python

### Machine Learning
- Support Vector Machine (SVM)
- Stacked Ensemble Learning
- Scikit-Learn

### Data Processing
- NumPy
- Pandas

### Data Balancing
- SMOTE (Synthetic Minority Oversampling Technique)

### Model Evaluation
- Stratified 5-Fold Cross Validation
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Precision-Recall Curve
- Confusion Matrix

### Visualization
- Matplotlib
- Seaborn

### Development Environment
- Jupyter Notebook
- VS Code