import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load data
df = pd.read_csv("./input/Is-this-a-good-customer.csv")

# Preprocess target variable
le = LabelEncoder()
df["bad_client_target"] = le.fit_transform(df["bad_client_target"])

# Define preprocessing steps for numerical and categorical columns
numerical_cols = df.select_dtypes(include=["int64"]).columns.tolist()
numerical_cols.remove("bad_client_target")
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

numerical_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Define the model pipeline
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ]
)

# Define cross-validation strategy
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and evaluate the model
f1_scores = []
for train_index, val_index in kf.split(
    df.drop("bad_client_target", axis=1), df["bad_client_target"]
):
    X_train, X_val = (
        df.drop("bad_client_target", axis=1).iloc[train_index],
        df.drop("bad_client_target", axis=1).iloc[val_index],
    )
    y_train, y_val = (
        df["bad_client_target"].iloc[train_index],
        df["bad_client_target"].iloc[val_index],
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1_scores.append(f1_score(y_val, y_pred))

print(sum(f1_scores) / len(f1_scores))
