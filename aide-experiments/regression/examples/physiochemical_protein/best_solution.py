import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# Load the dataset
data = pd.read_csv("./input/physiochemical_protein.csv")

# Split the data into features (X) and target (y)
X = data.drop(["ResidualSize"], axis=1)
y = data["ResidualSize"]

# Define the number of folds for cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Lists to store the MAE for each fold
mae_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize and train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the validation set
    y_pred = model.predict(X_val)

    # Calculate the MAE for this fold
    mae = mean_absolute_error(y_val, y_pred)
    mae_scores.append(mae)

    # For the first fold, also predict on the test set if it exists
    # However, the task does not explicitly mention a test set, so we'll focus on the validation MAE

# Calculate the average MAE across all folds
average_mae = sum(mae_scores) / n_folds
print(average_mae)
