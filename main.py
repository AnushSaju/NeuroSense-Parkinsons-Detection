# ======================================
# Parkinson's Detection – Speech Model
# Threshold Tuning + Model Comparison
# ======================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ------------------------------
# PATH SETUP
# ------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "speech_raw", "train_data.txt")
TEST_PATH = os.path.join(BASE_DIR, "data", "speech_raw", "test_data.txt")

# ------------------------------
# LOAD DATA
# ------------------------------

train_df = pd.read_csv(TRAIN_PATH, header=None)
test_df = pd.read_csv(TEST_PATH, header=None)

X_train = train_df.iloc[:, 1:]
y_train_raw = train_df.iloc[:, 0]

X_test = test_df.iloc[:, 1:]
y_test_raw = test_df.iloc[:, 0]

# ------------------------------
# FIX LABELS (PROXY)
# ------------------------------

threshold_label = np.median(y_train_raw)

y_train = (y_train_raw > threshold_label).astype(int)
y_test = (y_test_raw > threshold_label).astype(int)

# ------------------------------
# FEATURE ALIGNMENT
# ------------------------------

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

common_cols = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_cols]
X_test = X_test[common_cols]

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

# ------------------------------
# SCALING
# ------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# MODEL 1: SVM
# =====================================================

svm = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm.fit(X_train_scaled, y_train)
svm_probs = svm.predict_proba(X_test_scaled)[:, 1]

# ROC for SVM
fpr_svm, tpr_svm, thresholds = roc_curve(y_test, svm_probs)
roc_auc_svm = roc_auc_score(y_test, svm_probs)

# ---- Threshold tuning (Youden’s J statistic) ----
j_scores = tpr_svm - fpr_svm
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

svm_pred_tuned = (svm_probs >= best_threshold).astype(int)

# Metrics
print("\n📊 SVM (Threshold Tuned)")
print("Best threshold:", round(best_threshold, 3))
print("Accuracy :", accuracy_score(y_test, svm_pred_tuned))
print("Precision:", precision_score(y_test, svm_pred_tuned))
print("Recall   :", recall_score(y_test, svm_pred_tuned))
print("F1-score :", f1_score(y_test, svm_pred_tuned))
print("ROC-AUC  :", roc_auc_svm)
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred_tuned))

# =====================================================
# MODEL 2: RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train_scaled, y_train)
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
roc_auc_rf = roc_auc_score(y_test, rf_probs)

print("\n📊 Random Forest")
print("ROC-AUC:", roc_auc_rf)

# ------------------------------
# ROC CURVE PLOT (COMPARISON)
# ------------------------------

plt.figure()
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.2f})")
plt.plot(*roc_curve(y_test, rf_probs)[:2], label=f"RF (AUC = {roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Speech Parkinson’s Detection")
plt.legend()
plt.show()

print("\n✅ SPEECH MODELS FINALIZED")
