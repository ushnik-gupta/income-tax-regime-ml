import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/tax_ml_data.csv")

# Encode target column
le = LabelEncoder()
data["Best_Regime"] = le.fit_transform(data["Best_Regime"])  # Old=0, New=1

# Features and target
X = data[["Income", "Age", "Deductions", "Old_Tax", "New_Tax"]]
y = data["Best_Regime"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Example prediction
sample_input = [[900000, 35, 150000, 85000, 75000]]
prediction = model.predict(sample_input)

result = "Old Regime" if prediction[0] == 0 else "New Regime"
print("Recommended Tax Regime for sample input:", result)
