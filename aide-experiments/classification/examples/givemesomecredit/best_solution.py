import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# Load data
data = pd.read_csv("./input/GiveMeSomeCredit.csv")

# Preprocess data
le = LabelEncoder()
data["FinancialDistressNextTwoYears"] = le.fit_transform(
    data["FinancialDistressNextTwoYears"]
)

# Handle missing values
data["MonthlyIncome"] = data["MonthlyIncome"].replace("?", np.nan).astype(float)
data["NumberOfDependents"] = (
    data["NumberOfDependents"].replace("?", np.nan).astype(float)
)

imputer = SimpleImputer(strategy="mean")
data[["MonthlyIncome", "NumberOfDependents"]] = imputer.fit_transform(
    data[["MonthlyIncome", "NumberOfDependents"]]
)

# Split data into features and target
X = data.drop("FinancialDistressNextTwoYears", axis=1)
y = data["FinancialDistressNextTwoYears"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict_proba(X_val)[:, 1]

# Evaluate the model using AUC-ROC
auc = roc_auc_score(y_val, y_pred)
print(auc)

# Make predictions on the test set (assuming it's the same as the input data for simplicity)
# In a real competition, you would load the test data here
test_data = data.drop("FinancialDistressNextTwoYears", axis=1)
test_pred = clf.predict_proba(test_data)[:, 1]
submission_df = pd.DataFrame({"Id": range(len(test_pred)), "Probability": test_pred})
submission_df.to_csv("./working/submission.csv", index=False)
