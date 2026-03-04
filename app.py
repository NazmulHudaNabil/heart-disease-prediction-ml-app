import streamlit as st
import pandas as pd
import joblib

# Load model only
model = joblib.load("/Users/mdnazmulhudanabil/Desktop/Heart Disease Advance Projects/model/svm_heart_model.pkl")

st.title("🫀 Heart Disease Prediction")

# ======================
# USER INPUT
# ======================
age = st.slider("Age", 20, 100, 40)
sex = st.selectbox("Sex", [0,1])
chest_pain = st.selectbox("Chest Pain Type", [0,1,2,3])
resting_bp = st.number_input("Resting BP", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 400, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [0,1])
rest_ecg = st.selectbox("Resting ECG", [0,1,2])
max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", [0,1,2])

# Raw DataFrame
input_df = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "chest pain type": [chest_pain],
    "resting bp s": [resting_bp],
    "cholesterol": [cholesterol],
    "fasting blood sugar": [fasting_bs],
    "resting ecg": [rest_ecg],
    "max heart rate": [max_hr],
    "exercise angina": [exercise_angina],
    "oldpeak": [oldpeak],
    "ST slope": [st_slope],
})

# 🔥 Apply get_dummies
input_df = pd.get_dummies(input_df)

# 🔥 Match exact training columns
input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction (NO SCALER)
if st.button("Predict"):

    st.write("Final Input Given To Model:")
    st.write(input_df)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.write("Raw Probability:", probability)

    if prediction == 1:
        st.error("⚠️ High Risk")
    else:
        st.success("✅ Low Risk")
