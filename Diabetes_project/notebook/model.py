import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# -------------------------------
# 1. Load dataset
df = pd.read_csv("../data/diabetes.csv")

# -------------------------------
# 2. Handle missing/zero values
cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

# -------------------------------
# 3. Split data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 5. XGBoost Model (Better)
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss'
)

# -------------------------------
# 6. Train
model.fit(X_train, y_train)

# -------------------------------
# 7. Evaluate
pred = model.predict(X_test)

print("\nXGBoost Model")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))

# -------------------------------
# 8. Save model & scaler
pickle.dump(model, open("../model/model.pkl", "wb"))
pickle.dump(scaler, open("../model/scaler.pkl", "wb"))

print("\nModel saved successfully!")