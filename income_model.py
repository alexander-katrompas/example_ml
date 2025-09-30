#########################################################################
# Change parameters in income_config.py to experiment with
# different settings.
# This file is used to train and evaluate a Logistic Regression model
# on the preprocessed Adult Census dataset.
#########################################################################

import income_config as cfg
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

# 1. Load Data
train = pd.read_csv("train_" + cfg.DATAFILE)
test = pd.read_csv("test_" + cfg.DATAFILE)

# 2. Split features and target (label)
X_train = train.drop("income", axis=1)
y_train = train["income"]

X_test = test.drop("income", axis=1)
y_test = test["income"]

# 3. Initialize Logistic Regression
# solver='liblinear' works well for smaller datasets and binary classification
log_reg = LogisticRegression(max_iter=cfg.MAXITER, solver="liblinear", class_weight=cfg.CLASSWEIGHT)

# 4. Fit model
log_reg.fit(X_train, y_train)

# 5. Predictions
y_pred = log_reg.predict(X_test)
y_pred_proba = log_reg.predict_proba(X_test)[:, 1]

# 6. Evaluation
print("=== Logistic Regression Results ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 7. (Optional) Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values(by="Coefficient", ascending=False)

print("\n=== Top Features by Coefficient ===")
print(feature_importance.head(10))
print("\n=== Bottom Features by Coefficient ===")
print(feature_importance.tail(10))
