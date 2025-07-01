import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load the data
data = pd.read_csv("./input/cars.csv")

# Encode categorical variables
le = LabelEncoder()
data["Make"] = le.fit_transform(data["Make"])
data["Model"] = le.fit_transform(data["Model"])
data["Trim"] = le.fit_transform(data["Trim"])
data["Type"] = le.fit_transform(data["Type"])

# Split the data into features and target
X = data.drop(["Price"], axis=1)
y = data["Price"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model using MAE
mae = mean_absolute_error(y_val, y_pred)
print(mae)
