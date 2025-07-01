import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import numpy as np

# Load data
df = pd.read_csv("./input/customer_satisfaction_in_airline.csv")

# Preprocess data
le = LabelEncoder()
for col in ["Class", "CustomerType", "TypeofTravel", "satisfaction"]:
    df[col] = le.fit_transform(df[col])
df["ArrivalDelayinMinutes"] = df["ArrivalDelayinMinutes"].astype(float)

# Define features and target
X = df.drop(["satisfaction"], axis=1)
y = df["satisfaction"]

# Initialize 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train a LightGBM model
    model = lgb.LGBMClassifier(objective="binary", random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)
    print(f"Fold {fold+1} Accuracy: {accuracy:.4f}")

# Print average accuracy across folds
print(f"Average Accuracy: {np.mean(accuracies):.4f}")

# Train on the full dataset and save predictions for test data (if any)
# Since there's no test data provided in the task description, we'll assume the task is to evaluate on the given data.
# If test data is available and needs prediction, it should be loaded, preprocessed similarly, and used for prediction here.
