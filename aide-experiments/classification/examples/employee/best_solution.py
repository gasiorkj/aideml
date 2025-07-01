import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv("./input/employee.csv")

# Preprocess data
categorical_cols = ["City", "Education", "EverBenched", "Gender", "LeaveOrNot"]
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define features and target
X = df.drop("LeaveOrNot", axis=1)
y = df["LeaveOrNot"]

# Define TabNet classifier
clf = TabNetClassifier()

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train TabNet model
    clf.fit(
        X_train.values,
        y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        max_epochs=50,
        patience=10,
    )

    # Predict on validation set
    y_pred = clf.predict(X_val.values)

    # Compute accuracy
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

# Print average accuracy across folds
print(f"Average Accuracy: {sum(accuracies)/len(accuracies):.4f}")
