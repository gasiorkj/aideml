import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Load the data
df = pd.read_csv("./input/solar_flare.csv")

# Preprocess categorical variables
categorical_cols = ["LargestSpotSize", "ModZurichClass", "SpotDist"]
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split data into features and target
X = df.drop("common_flares", axis=1)
y = df["common_flares"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
}
model = lgb.train(params, train_data, num_boost_round=100)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model using MAE
mae = mean_absolute_error(y_val, y_pred)
print(mae)

# Make predictions on the test set (assuming there's a test set, but it's not provided in the task description)
# For the sake of completeness, let's assume we have a test set and we need to save the predictions
# test_df = pd.read_csv('./input/test.csv')  # Uncomment and adjust as necessary
# test_df = test_df.drop('common_flares', axis=1) if 'common_flares' in test_df.columns else test_df
# for col in categorical_cols:
#     test_df[col] = le.transform(test_df[col])
# test_pred = model.predict(test_df)
# submission_df = pd.DataFrame({'common_flares': test_pred})
# submission_df.to_csv('./working/submission.csv', index=False)
