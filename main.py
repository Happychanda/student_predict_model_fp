# ================================
# Student Success Prediction Model
# ================================

# ---- Import required libraries ----
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ---- Load Dataset ----
df = pd.read_csv("student_success_dataset.csv")

print("Sample Data:")
print(df.head())


# ---- Dataset Info ----
print("\nDataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nDataset Info:")
print(df.info())


# ---- Check Missing Values ----
print("\nMissing values in each column:")
print(df.isnull().sum())


# ---- Handle Missing Values (if any) ----
# Fill numeric missing values with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical missing values with mode
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)


# ---- Encode Categorical Columns ----
le = LabelEncoder()

# Example categorical columns
df["Internet"] = le.fit_transform(df["Internet"])   # yes=1, no=0
df["Passed"] = le.fit_transform(df["Passed"])       # yes=1, no=0

print("\nData after Encoding:")
print(df.head())


# ---- Feature Selection ----
features = ["StudyHours", "Attendance", "PastScore", "SleepHours"]
target = "Passed"

X = df[features]
y = df[target]


# ---- Feature Scaling ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ---- Train Logistic Regression Model ----
model = LogisticRegression()
model.fit(X_train, y_train)


# ---- Model Prediction ----
y_pred = model.predict(X_test)


# ---- Model Evaluation ----
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ---- Confusion Matrix ----
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ================================
# User Input Prediction Section
# ================================
try:
    # ---- Take User Input ----
    study_hours = float(input("Enter Study Hours: "))
    attendance = float(input("Enter Attendance (%): "))
    past_score = float(input("Enter Past Score: "))
    sleep_hours = float(input("Enter Sleep Hours: "))

    # ---- Convert Input to DataFrame ----
    user_input_df = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PastScore": past_score,
        "SleepHours": sleep_hours
    }])

    # ---- Scale User Input ----
    user_input_scaled = scaler.transform(user_input_df)

    # ---- Predict Result ----
    prediction = model.predict(user_input_scaled)[0]

    result = "PASS ✅" if prediction == 1 else "FAIL ❌"
    print(f"\nPrediction Based on Input: {result}")

except Exception as e:
    print("An error occurred:", e)
