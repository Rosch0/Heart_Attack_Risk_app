import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Wczytaj dane
@st.cache_data
def load_data():
    file_path = "heart.csv"  # Upewnij się, że plik `heart.csv` znajduje się w tym samym folderze
    data = pd.read_csv(file_path)
    return data

data = load_data()

# Przygotowanie danych
X = data.drop(columns='output')
y = data['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trening modelu
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Interfejs użytkownika
st.title("Heart Attack Risk Predictor")
st.markdown("Wprowadź dane pacjenta, aby oszacować ryzyko zawału serca.")

# Pola wejściowe
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex (1=Male, 0=Female)", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type (1-4)", options=[1, 2, 3, 4])
trtbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Cholesterol Level", value=200)
fbs = st.selectbox("Fasting Blood Sugar (>120 mg/dl, 1=True, 0=False)", options=[1, 0], format_func=lambda x: "True" if x == 1 else "False")
restecg = st.selectbox("Resting ECG Results (0-2)", options=[0, 1, 2])
thalachh = st.number_input("Max Heart Rate Achieved", value=150)
exng = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.number_input("ST Depression", value=1.0)
slp = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", options=[0, 1, 2])
caa = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
thall = st.selectbox("Thalassemia (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", options=[1, 2, 3])

# Przewidywanie
if st.button("Predict Risk"):
    try:
        inputs = [[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]]
        prediction = rf_model.predict_proba(inputs)[0][1] * 100
        st.success(f"Estimated heart attack risk: {prediction:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
