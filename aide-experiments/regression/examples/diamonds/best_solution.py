import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load data
data = pd.read_csv("./input/diamonds.csv")

# Preprocess data
le = LabelEncoder()
for col in ["clarity", "color", "cut"]:
    data[col] = le.fit_transform(data[col])

# Define features and target
X = data.drop(["carat"], axis=1)
y = data["carat"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model and perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_fold, y_train_fold)
    y_pred = model.predict(X_val_fold)
    mae_scores.append(mean_absolute_error(y_val_fold, y_pred))

# Train model on full training set and evaluate on validation set
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val, y_pred)

# Print evaluation metric
print(f"Validation MAE: {val_mae:.4f}")
print(f"Average 5-fold CV MAE: {np.mean(mae_scores):.4f}")

# Make predictions on test set if available
try:
    test_data = pd.read_csv("./input/test.csv")
    for col in ["clarity", "color", "cut"]:
        test_data[col] = le.transform(test_data[col])
    test_pred = model.predict(test_data)
    submission = pd.DataFrame({"id": test_data.index, "carat": test_pred})
    submission.to_csv("./working/submission.csv", index=False)
except FileNotFoundError:
    pass
