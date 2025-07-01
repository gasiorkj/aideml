import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load the data
df = pd.read_csv("./input/airfoil_self_noise.csv")

# Define features and target
X = df.drop("scaled-sound-pressure", axis=1)
y = df["scaled-sound-pressure"]

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store MAE scores
mae_scores = []

# Perform 5-fold cross-validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Initialize and train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = model.predict(X_val)

    # Calculate MAE
    mae = mean_absolute_error(y_val, y_pred)
    mae_scores.append(mae)

# Print the average MAE across 5 folds
print(np.mean(mae_scores))

# Train a model on the entire dataset and make predictions on the test set (if it existed)
# Since there's no test set provided, we'll just use the trained model for evaluation
# If there was a test set, we'd do something like this:
# test_df = pd.read_csv('./input/test.csv')
# test_pred = model.predict(test_df)
# submission_df = pd.DataFrame({'scaled-sound-pressure': test_pred})
# submission_df.to_csv('./working/submission.csv', index=False)
