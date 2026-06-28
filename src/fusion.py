# =====================================================
# Multimodal Parkinson's Detection
# Feature-Level Fusion (Spiral + Meander) + Speech
# Advanced Stacked SVM
# =====================================================

import os
from imblearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict



# =====================================================
# PATH SETUP
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEECH_TRAIN = os.path.join(BASE_DIR, "data", "speech_raw", "train_data.txt")
SPEECH_TEST  = os.path.join(BASE_DIR, "data", "speech_raw", "test_data.txt")

SPIRAL_PATH  = os.path.join(BASE_DIR, "data", "handwriting_raw", "Spiral_HandPD.csv")
MEANDER_PATH = os.path.join(BASE_DIR, "data", "handwriting_raw", "Meander_HandPD.csv")


# =====================================================
# 1️⃣ SPEECH MODEL (Cross-Validated Predictions)
# =====================================================

speech_df = pd.read_csv(SPEECH_TRAIN, header=None)

X_s = speech_df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
y_s_raw = speech_df.iloc[:, 0]

thr = np.median(y_s_raw)
y_s = (y_s_raw > thr).astype(int)

X_s = X_s.fillna(X_s.mean())


svm_speech = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("svm", SVC(kernel="rbf",
                probability=True,
                random_state=42))
])

# 5-fold cross validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

P_speech = cross_val_predict(
    svm_speech,
    X_s,
    y_s,
    cv=kf,
    method="predict_proba"
)[:,1]


# =====================================================
# 2️⃣ HANDWRITING FEATURE-LEVEL FUSION
# =====================================================

spiral_df  = pd.read_csv(SPIRAL_PATH)
meander_df = pd.read_csv(MEANDER_PATH)

drop_cols = ["_ID_EXAM","IMAGE_NAME","ID_PATIENT",
             "GENDER","RIGH/LEFT-HANDED"]

spiral_df  = spiral_df.drop(columns=[c for c in drop_cols if c in spiral_df.columns])
meander_df = meander_df.drop(columns=[c for c in drop_cols if c in meander_df.columns])

# Use Meander labels (same subjects)
y_hand = meander_df["CLASS_TYPE"].map({1:0, 2:1})

print("\n🔍 ORIGINAL HANDWRITING DATA")
print("Total samples:", len(y_hand))
print("Class distribution:\n", y_hand.value_counts())

X_spiral  = spiral_df.drop(columns=["CLASS_TYPE"]).apply(pd.to_numeric, errors="coerce")
X_meander = meander_df.drop(columns=["CLASS_TYPE"]).apply(pd.to_numeric, errors="coerce")

# Combine features horizontally
X_hand_combined = pd.concat([X_spiral, X_meander], axis=1)

# Handle missing values
X_hand_combined = X_hand_combined.fillna(X_hand_combined.mean())


# =====================================================
# HANDWRITING MODEL (Cross-Validated Predictions)
# =====================================================

# Scale all handwriting features

svm_hand = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("svm", SVC(kernel="rbf",
                probability=True,
                random_state=42))
])

# Use same 5-fold strategy as speech
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n🔍 CHECKING ONE FOLD BEFORE SMOTE")

for train_idx, test_idx in kf.split(X_hand_combined, y_hand):
    X_train, y_train = X_hand_combined.iloc[train_idx], y_hand.iloc[train_idx]
    
    print("Training size:", len(y_train))
    print("Before SMOTE:\n", y_train.value_counts())
    break  # only check first fold

print("\n🔍 SMOTE INSIDE ONE FOLD (DETAILED CHECK)")

for train_idx, test_idx in kf.split(X_hand_combined, y_hand):
    X_train, y_train = X_hand_combined.iloc[train_idx], y_hand.iloc[train_idx]

    print("\n--- BEFORE SMOTE ---")
    print("Training size:", len(y_train))
    print(y_train.value_counts())

    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)

    X_res, y_res = sm.fit_resample(X_train, y_train)

    print("\n--- AFTER SMOTE ---")
    print(pd.Series(y_res).value_counts())
    print("After SMOTE total:", len(y_res))

    break  # only check first fold

# Generate out-of-fold predictions
P_hand = cross_val_predict(
    svm_hand,
    X_hand_combined,
    y_hand,
    cv=kf,
    method="predict_proba"
)[:,1]

print("\n🔍 HANDWRITING PREDICTIONS CHECK")
print("Total handwriting predictions:", len(P_hand))
# =====================================================
# 3️⃣ ALIGN PROBABILITIES
# =====================================================

N = min(len(P_speech), len(P_hand))

P_s = P_speech[:N]
P_h = P_hand[:N]

# Use handwriting labels for aligned samples
y_final = y_hand.values[:N]


# =====================================================
# 4️⃣ STACKED META-SVM (Cross-Validated)
# =====================================================

X_meta = np.column_stack([P_s, P_h])

meta_model = SVC(kernel="rbf", probability=True,
                 class_weight="balanced", random_state=42)

# Generate cross-validated predictions

P_final = cross_val_predict(
    meta_model,
    X_meta,
    y_final,
    cv=kf,
    method="predict_proba"
)[:,1]

# =====================================================
# 4.1️⃣ THRESHOLD OPTIMIZATION
# =====================================================

import numpy as np
from sklearn.metrics import confusion_matrix

thresholds = np.linspace(0.3, 0.8, 50)

best_threshold = 0.5
best_fp = float('inf')

for t in thresholds:
    y_temp = (P_final > t).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_final, y_temp).ravel()
    recall = tp / (tp + fn)
    
    # Keep recall high
    if recall >= 0.90:
        if fp < best_fp:
            best_fp = fp
            best_threshold = t
            best_values = (tn, fp, fn, tp, recall)

# Apply best threshold
tn, fp, fn, tp, rec_opt = best_values
y_pred = (P_final > best_threshold).astype(int)

print("\n✅ OPTIMAL THRESHOLD FOUND:", round(best_threshold, 2))
print(f"Optimized Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Optimized Recall: {rec_opt:.3f}")


# =====================================================
# 5️⃣ METRICS
# =====================================================

acc  = accuracy_score(y_final, y_pred)
prec = precision_score(y_final, y_pred)
rec  = recall_score(y_final, y_pred)
f1   = f1_score(y_final, y_pred)
auc  = roc_auc_score(y_final, P_final)

print("\n📊 FINAL MULTIMODAL RESULTS (Unified Handwriting + Speech)")
print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("ROC-AUC  :", auc)
print("Confusion Matrix:\n", confusion_matrix(y_final, y_pred))





# =====================================================
# 7️⃣ SEVERITY SCORE
# =====================================================

def severity_band(p):
    if p < 0.3: return "Healthy"
    if p < 0.5: return "Mild"
    if p < 0.7: return "Moderate"
    return "Severe"

print("\nSample Severity Predictions:")
for i in range(min(10, N)):
    print(f"Subject {i+1}: {P_final[i]:.2f} → {severity_band(P_final[i])}")

print("\n✅ FINAL UNIFIED MULTIMODAL SYSTEM COMPLETE")


# ESSENTIAL VISUALIZATIONS (Mentor-Ready)
# =====================================================

print("\n📊 GENERATING ESSENTIAL VISUAL OUTPUTS...")

from sklearn.metrics import precision_recall_curve
import seaborn as sns

# ------------------------------
# 1️⃣ Confusion Matrix
# ------------------------------

cm = confusion_matrix(y_final, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Healthy","PD"],
            yticklabels=["Healthy","PD"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Multimodal Model")
plt.tight_layout()
plt.show()


# ------------------------------
# 2️⃣ ROC Curve
# ------------------------------

fpr, tpr, _ = roc_curve(y_final, P_final)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Multimodal Fusion")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------
# 3️⃣ Precision-Recall Curve
# ------------------------------

precision_curve, recall_curve, _ = precision_recall_curve(y_final, P_final)

plt.figure()
plt.plot(recall_curve, precision_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.tight_layout()
plt.show()
