"""
BankMind - Track B (ML Engineer)
Predicts whether a bank customer will subscribe to a term deposit (UCI Bank
Marketing dataset, bank-full.csv) using Logistic Regression (baseline) vs
XGBoost (main model).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: LOAD & INSPECT DATA")
print("=" * 70)

df = pd.read_csv("bank-full.csv", sep=";")
df.columns = [c.strip() for c in df.columns]

print(f"\nShape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")

print(f"\nMissing values (literal NaN) per column:\n{df.isnull().sum().sum()} total NaNs")

# This dataset doesn't use NaN for missing data - categorical columns use the
# literal string 'unknown' instead. Worth flagging explicitly since a naive
# df.isnull().sum() would say "no missing data" which is misleading.
print("\n'unknown' value counts in categorical columns (this dataset's real missing-data signal):")
for col in ["job", "education", "contact", "poutcome"]:
    n_unknown = (df[col] == "unknown").sum()
    print(f"  {col}: {n_unknown} ({n_unknown/len(df)*100:.1f}%)")

print(f"\nTarget class distribution:")
print(df["y"].value_counts())
print(df["y"].value_counts(normalize=True) * 100)

# ---------------------------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: PREPROCESSING")
print("=" * 70)

# IMPORTANT: 'duration' (last call duration in seconds) is a leakage feature.
# The dataset's own documentation flags this: duration=0 almost guarantees
# y='no', and duration is only known AFTER the call ends - so it can't be
# used by a real RM-facing model that predicts BEFORE the call happens.
# Dropping it for a model meant to actually guide who to call.
LEAKY_COLS = ["duration"]
df_model = df.drop(columns=LEAKY_COLS)
print(f"Dropped leakage column(s): {LEAKY_COLS} (not known before a call is made)")

target_map = {"no": 0, "yes": 1}
y = df_model["y"].map(target_map)
X = df_model.drop(columns=["y"])

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train class balance: {y_train.mean()*100:.2f}% yes")
print(f"Test class balance:  {y_test.mean()*100:.2f}% yes")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

# ---------------------------------------------------------------------------
# 3. BASELINE MODEL: LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: BASELINE - LOGISTIC REGRESSION")
print("=" * 70)

logreg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
logreg_pipeline.fit(X_train, y_train)
logreg_pred = logreg_pipeline.predict(X_test)
logreg_proba = logreg_pipeline.predict_proba(X_test)[:, 1]

print("\nClassification report (Logistic Regression):")
print(classification_report(y_test, logreg_pred, target_names=["no", "yes"]))
print("Confusion matrix:\n", confusion_matrix(y_test, logreg_pred))

logreg_metrics = {
    "accuracy": accuracy_score(y_test, logreg_pred),
    "precision": precision_score(y_test, logreg_pred),
    "recall": recall_score(y_test, logreg_pred),
    "f1": f1_score(y_test, logreg_pred),
    "roc_auc": roc_auc_score(y_test, logreg_proba),
}
print("Summary metrics:", {k: round(v, 4) for k, v in logreg_metrics.items()})

# ---------------------------------------------------------------------------
# 4. MAIN MODEL: XGBOOST
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: MAIN MODEL - XGBOOST")
print("=" * 70)

# class imbalance ratio for XGBoost's scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight (neg/pos ratio): {scale_pos_weight:.2f}")

xgb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )),
])
xgb_pipeline.fit(X_train, y_train)
xgb_pred = xgb_pipeline.predict(X_test)
xgb_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

print("\nClassification report (XGBoost):")
print(classification_report(y_test, xgb_pred, target_names=["no", "yes"]))
print("Confusion matrix:\n", confusion_matrix(y_test, xgb_pred))

xgb_metrics = {
    "accuracy": accuracy_score(y_test, xgb_pred),
    "precision": precision_score(y_test, xgb_pred),
    "recall": recall_score(y_test, xgb_pred),
    "f1": f1_score(y_test, xgb_pred),
    "roc_auc": roc_auc_score(y_test, xgb_proba),
}
print("Summary metrics:", {k: round(v, 4) for k, v in xgb_metrics.items()})

print("\n--- COMPARISON TABLE ---")
comp = pd.DataFrame([logreg_metrics, xgb_metrics], index=["LogisticRegression", "XGBoost"])
print(comp.round(4))

# ---------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (XGBoost)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: FEATURE IMPORTANCE")
print("=" * 70)

feature_names = xgb_pipeline.named_steps["preprocess"].get_feature_names_out()
importances = xgb_pipeline.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).head(15)
print(imp_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. SAMPLE PREDICTIONS - 5 CUSTOMERS (>=2 yes, >=2 no)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: SAMPLE CUSTOMER PREDICTIONS")
print("=" * 70)

results_df = X_test.copy()
results_df["actual"] = y_test.values
results_df["predicted"] = xgb_pred
results_df["probability_yes"] = xgb_proba.round(3)

yes_samples = results_df[results_df["predicted"] == 1].sample(2, random_state=7)
no_samples = results_df[results_df["predicted"] == 0].sample(3, random_state=7)
sample_set = pd.concat([yes_samples, no_samples])

display_cols = ["age", "job", "marital", "education", "balance",
                 "housing", "loan", "predicted", "probability_yes", "actual"]
sample_display = sample_set[display_cols].copy()
sample_display["predicted"] = sample_display["predicted"].map({1: "yes", 0: "no"})
sample_display["actual"] = sample_display["actual"].map({1: "yes", 0: "no"})

print("\n5 sample customers from the test set:\n")
print(sample_display.to_string())

sample_display.to_csv("sample_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# 7. SAVE MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: SAVE MODEL")
print("=" * 70)

joblib.dump(xgb_pipeline, "bankmind_xgb_model.pkl")
print("Saved full pipeline (preprocessing + model) to bankmind_xgb_model.pkl")

print("\nDONE.")
