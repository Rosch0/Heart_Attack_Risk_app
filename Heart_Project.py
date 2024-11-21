import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load the CSV file to inspect its structure
file_path = r"C:\Users\Blaze\Desktop\Heart_Attack_Risk_app\heart.csv"
data = pd.read_csv(file_path)

data.head(), data.info()
X = data.drop(columns='output')
y = data['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report


# Function to predict heart attack risk
def predict_risk():
    try:
        inputs = [
            float(entry_age.get()),
            int(entry_sex.get()),
            int(entry_cp.get()),
            float(entry_trtbps.get()),
            float(entry_chol.get()),
            int(entry_fbs.get()),
            int(entry_restecg.get()),
            float(entry_thalachh.get()),
            int(entry_exng.get()),
            float(entry_oldpeak.get()),
            int(entry_slp.get()),
            int(entry_caa.get()),
            int(entry_thall.get())
        ]
        prediction = rf_model.predict_proba([inputs])[0][1] * 100
        messagebox.showinfo("Prediction Result", f"Estimated heart attack risk: {prediction:.2f}%")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid data in all fields.")

#GUI window
root = tk.Tk()
root.title("Heart Attack Risk Predictor")

# Labels and entry fields for each feature
fields = [
    ("Age", "entry_age"),
    ("Sex (1=Male, 0=Female)", "entry_sex"),
    ("Chest Pain Type (1-4)", "entry_cp"),
    ("Resting Blood Pressure", "entry_trtbps"),
    ("Cholesterol Level", "entry_chol"),
    ("Fasting Blood Sugar (>120 mg/dl, 1=True, 0=False)", "entry_fbs"),
    ("Resting ECG Results (0-2)", "entry_restecg"),
    ("Max Heart Rate Achieved", "entry_thalachh"),
    ("Exercise Induced Angina (1=Yes, 0=No)", "entry_exng"),
    ("ST Depression", "entry_oldpeak"),
    ("Slope of Peak Exercise ST Segment (0-2)", "entry_slp"),
    ("Number of Major Vessels (0-3)", "entry_caa"),
    ("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", "entry_thall")
]

entries = {}
for i, (label_text, var_name) in enumerate(fields):
    tk.Label(root, text=label_text).grid(row=i, column=0, pady=5, sticky="w")
    entries[var_name] = tk.Entry(root)
    entries[var_name].grid(row=i, column=1, pady=5)

# Assign entries to variables for easy access
entry_age = entries["entry_age"]
entry_sex = entries["entry_sex"]
entry_cp = entries["entry_cp"]
entry_trtbps = entries["entry_trtbps"]
entry_chol = entries["entry_chol"]
entry_fbs = entries["entry_fbs"]
entry_restecg = entries["entry_restecg"]
entry_thalachh = entries["entry_thalachh"]
entry_exng = entries["entry_exng"]
entry_oldpeak = entries["entry_oldpeak"]
entry_slp = entries["entry_slp"]
entry_caa = entries["entry_caa"]
entry_thall = entries["entry_thall"]

# Predict button
tk.Button(root, text="Predict Risk", command=predict_risk).grid(row=len(fields), column=0, columnspan=2, pady=20)

root.mainloop()
