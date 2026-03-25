import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
df = pd.read_csv("../data/raw/creditcard.csv")
df.head()
print("Shape of dataset:", df.shape)
print("\nColumns:")
print(df.columns)
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))
df["Class"].value_counts().plot(kind="bar")
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
df.describe()
df.groupby("Class")["Amount"].describe()
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_preds))

print("Random Forest ROC-AUC:", roc_auc_score(y_test, rf_probs))
print("Random Forest PR-AUC:", average_precision_score(y_test, rf_probs))
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))

print("XGBoost ROC-AUC:", roc_auc_score(y_test, xgb_probs))
print("XGBoost PR-AUC:", average_precision_score(y_test, xgb_probs))
results = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "ROC-AUC": [
        roc_auc_score(y_test, rf_probs),
        roc_auc_score(y_test, xgb_probs)
    ],
    "PR-AUC": [
        average_precision_score(y_test, rf_probs),
        average_precision_score(y_test, xgb_probs)
    ]
})
results