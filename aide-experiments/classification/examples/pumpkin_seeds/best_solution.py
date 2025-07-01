import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("./input/pumpkin_seeds.csv")

# Encode target variable
le = LabelEncoder()
data["SeedType"] = le.fit_transform(data["SeedType"])

# Split data into features and target
X = data.drop("SeedType", axis=1)
y = data["SeedType"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Make predictions on the test set (assuming there's no test set, this step is skipped)
# Since there's no test data provided, we won't create a submission.csv file
