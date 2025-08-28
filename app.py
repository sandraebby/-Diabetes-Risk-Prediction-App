import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load("diabetes_logistic_model_pipeline.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Prediction App")
st.write("This app predicts the risk of diabetes based on health information. "
         "Please note this is **not a medical diagnosis**. Consult a doctor for professional advice.")

# --- Input form ---
with st.form("user_input_form"):
    st.subheader("📋 Enter your details:")

    # Age slider
    age = st.slider("Age", min_value=1, max_value=120, value=30, step=1)

    # Gender dropdown
    gender = st.radio("Gender", ["male", "female", "other"], horizontal=True)

    # Hypertension & Heart Disease
    hypertension = st.selectbox("Hypertension (High Blood Pressure)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Smoking history
    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "former", "current", "ever", "not current", "unknown"]
    )

    # BMI slider
    bmi = st.slider("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.0, step=0.1)

    # HbA1c slider
    HbA1c_level = st.slider("HbA1c Level", min_value=3.0, max_value=20.0, value=5.5, step=0.1)

    # Blood glucose level slider
    blood_glucose_level = st.slider("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=100, step=1)

    # Submit button
    submit = st.form_submit_button("🔮 Predict Risk")

# -----------------------------
# Prediction
# -----------------------------
if submit:
    input_data = pd.DataFrame({
        "age": [int(age)],  
        "gender": [str(gender)],  
        "hypertension": [float(hypertension)],  # now float
        "heart_disease": [float(heart_disease)],  # now float
        "smoking_history": [str(smoking_history)],  
        "bmi": [float(bmi)],
        "HbA1c_level": [float(HbA1c_level)],
        "blood_glucose_level": [float(blood_glucose_level)]
    })

    # Predict
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    # Results
    st.subheader("🔍 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n\n**Probability: {probability:.2f}**")
        st.info("👉 Please consult a healthcare professional for further evaluation.")
    else:
        st.success(f"✅ Low Risk of Diabetes\n\n**Probability: {probability:.2f}**")
        st.balloons()
