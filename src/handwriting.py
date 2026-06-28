# ======================================
# Parkinson's Detection – Handwriting
# Spiral Drawing Model
# ======================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIRAL_PATH = os.path.join(
    BASE_DIR, "data", "handwriting_raw", "Spiral_HandPD.csv"
)

# ------------------------------
# LOAD DATA
# ------------------------------

df = pd.read_csv(SPIRAL_PATH)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------------
# CLEAN DATA
# ------------------------------

TARGET_COL = "CLASS_TYPE"

# Remove non-numeric / non-useful columns
drop_cols = [
    "_ID_EXAM",
    "IMAGE_NAME",
    "ID_PATIENT",
    "GENDER",
    "RIGH/LEFT-HANDED"
]

drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)

# Split features & labels
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].map({1: 0, 2: 1})


# Ensure numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

print("Feature shape:", X.shape)

# ------------------------------
# TRAIN–TEST SPLIT
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

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
svm_pred = svm.predict(X_test_scaled)

roc_auc_svm = roc_auc_score(y_test, svm_probs)

print("\n📊 SPIRAL – SVM RESULTS")
print("Accuracy :", accuracy_score(y_test, svm_pred))
print("Precision:", precision_score(y_test, svm_pred))
print("Recall   :", recall_score(y_test, svm_pred))
print("F1-score :", f1_score(y_test, svm_pred))
print("ROC-AUC  :", roc_auc_svm)
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))

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
rf_pred = rf.predict(X_test_scaled)

roc_auc_rf = roc_auc_score(y_test, rf_probs)

print("\n📊 SPIRAL – RANDOM FOREST RESULTS")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall   :", recall_score(y_test, rf_pred))
print("F1-score :", f1_score(y_test, rf_pred))
print("ROC-AUC  :", roc_auc_rf)
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# ------------------------------
# ROC CURVE
# ------------------------------

fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

plt.figure()
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Spiral Handwriting")
plt.legend()
plt.show()

print("\n✅ SPIRAL HANDWRITING MODEL COMPLETE")

# =====================================================
# MEANDER HANDWRITING MODEL
# =====================================================

print("\n==============================")
print("MEANDER HANDWRITING MODEL")
print("==============================")

MEANDER_PATH = os.path.join(
    BASE_DIR, "data", "handwriting_raw", "Meander_HandPD.csv"
)

# ------------------------------
# LOAD MEANDER DATA
# ------------------------------

df_m = pd.read_csv(MEANDER_PATH)

print("Original shape:", df_m.shape)
print("Columns:", df_m.columns.tolist())

TARGET_COL = "CLASS_TYPE"

drop_cols = [
    "_ID_EXAM",
    "IMAGE_NAME",
    "ID_PATIENT",
    "GENDER",
    "RIGH/LEFT-HANDED"
]
drop_cols = [c for c in drop_cols if c in df_m.columns]
df_m = df_m.drop(columns=drop_cols)

# Labels: map {1,2} -> {0,1}
y_m = df_m[TARGET_COL].map({1: 0, 2: 1})
X_m = df_m.drop(columns=[TARGET_COL])

# Ensure numeric
X_m = X_m.apply(pd.to_numeric, errors="coerce")
X_m = X_m.fillna(X_m.mean())

print("Meander feature shape:", X_m.shape)

# ------------------------------
# TRAIN–TEST SPLIT
# ------------------------------

X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_m,
    y_m,
    test_size=0.2,
    random_state=42,
    stratify=y_m
)

# ------------------------------
# SCALING
# ------------------------------

scaler_m = StandardScaler()
X_m_train_scaled = scaler_m.fit_transform(X_m_train)
X_m_test_scaled = scaler_m.transform(X_m_test)

# =====================================================
# MODEL 1: SVM (MEANDER)
# =====================================================

svm_m = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)

svm_m.fit(X_m_train_scaled, y_m_train)

svm_m_probs = svm_m.predict_proba(X_m_test_scaled)[:, 1]
svm_m_pred = svm_m.predict(X_m_test_scaled)

roc_auc_svm_m = roc_auc_score(y_m_test, svm_m_probs)

print("\n📊 MEANDER – SVM RESULTS")
print("Accuracy :", accuracy_score(y_m_test, svm_m_pred))
print("Precision:", precision_score(y_m_test, svm_m_pred))
print("Recall   :", recall_score(y_m_test, svm_m_pred))
print("F1-score :", f1_score(y_m_test, svm_m_pred))
print("ROC-AUC  :", roc_auc_svm_m)
print("Confusion Matrix:\n", confusion_matrix(y_m_test, svm_m_pred))

# =====================================================
# MODEL 2: RANDOM FOREST (MEANDER)
# =====================================================

rf_m = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf_m.fit(X_m_train_scaled, y_m_train)

rf_m_probs = rf_m.predict_proba(X_m_test_scaled)[:, 1]
rf_m_pred = rf_m.predict(X_m_test_scaled)

roc_auc_rf_m = roc_auc_score(y_m_test, rf_m_probs)

print("\n📊 MEANDER – RANDOM FOREST RESULTS")
print("Accuracy :", accuracy_score(y_m_test, rf_m_pred))
print("Precision:", precision_score(y_m_test, rf_m_pred))
print("Recall   :", recall_score(y_m_test, rf_m_pred))
print("F1-score :", f1_score(y_m_test, rf_m_pred))
print("ROC-AUC  :", roc_auc_rf_m)
print("Confusion Matrix:\n", confusion_matrix(y_m_test, rf_m_pred))

# ------------------------------
# ROC CURVE (MEANDER)
# ------------------------------

fpr_svm_m, tpr_svm_m, _ = roc_curve(y_m_test, svm_m_probs)
fpr_rf_m, tpr_rf_m, _ = roc_curve(y_m_test, rf_m_probs)

plt.figure()
plt.plot(fpr_svm_m, tpr_svm_m, label=f"SVM (AUC = {roc_auc_svm_m:.2f})")
plt.plot(fpr_rf_m, tpr_rf_m, label=f"RF (AUC = {roc_auc_rf_m:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Meander Handwriting")
plt.legend()
plt.show()

print("\n✅ MEANDER HANDWRITING MODEL COMPLETE")

