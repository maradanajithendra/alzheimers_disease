import streamlit as st
import numpy as np
import joblib
import plotly.express as px

# -------------------------------
# Load Models and Scaler
# -------------------------------
log_model = joblib.load("logistic_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Streamlit Page Settings
# -------------------------------
st.set_page_config(page_title="Alzheimer's Prediction", layout="wide")
st.title("🧠 Alzheimer’s Disease Prediction System")
st.write("Fill the patient details in the sidebar to predict Alzheimer's risk.")

# -------------------------------
# Sidebar Input Sections
# -------------------------------
st.sidebar.header("Patient Input Form")
with st.sidebar.expander("Patient Identity"):
    PatientID = st.number_input("Patient ID", min_value=1, max_value=99999, value=4751)

with st.sidebar.expander("Basic Info"):
    Age = st.number_input("Age", 50, 100, 70)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3])
    EducationLevel = st.selectbox("Education Level", [0, 1, 2, 3])
    BMI = st.number_input("BMI", 10.0, 50.0, 25.0)

with st.sidebar.expander("Lifestyle"):
    Smoking = st.selectbox("Smoking", [0, 1])
    Alcohol = st.number_input("Alcohol Consumption", 0.0, 20.0, 5.0)
    PhysicalActivity = st.number_input("Physical Activity", 0.0, 10.0, 5.0)
    DietQuality = st.number_input("Diet Quality", 0.0, 10.0, 5.0)
    SleepQuality = st.number_input("Sleep Quality", 0.0, 10.0, 7.0)

with st.sidebar.expander("Medical History"):
    FamilyHistory = st.selectbox("Family History of Alzheimer's", [0, 1])
    CardiovascularDisease = st.selectbox("Cardiovascular Disease", [0, 1])
    Diabetes = st.selectbox("Diabetes", [0, 1])
    Depression = st.selectbox("Depression", [0, 1])
    HeadInjury = st.selectbox("Head Injury", [0, 1])
    Hypertension = st.selectbox("Hypertension", [0, 1])

with st.sidebar.expander("Vitals & Labs"):
    SystolicBP = st.number_input("Systolic BP", 80, 200, 120)
    DiastolicBP = st.number_input("Diastolic BP", 50, 120, 80)
    CholTotal = st.number_input("Total Cholesterol", 100, 400, 200)
    CholLDL = st.number_input("Cholesterol LDL", 40, 200, 120)
    CholHDL = st.number_input("Cholesterol HDL", 20, 100, 50)
    CholTrig = st.number_input("Cholesterol Triglycerides", 50, 500, 150)

with st.sidebar.expander("Cognitive & Functional"):
    MMSE = st.number_input("MMSE Score", 0, 30, 20)
    Functional = st.number_input("Functional Assessment", 0.0, 15.0, 5.0)
    MemoryComplaints = st.selectbox("Memory Complaints", [0, 1])
    BehavioralProblems = st.selectbox("Behavioral Problems", [0, 1])
    ADL = st.number_input("ADL (Daily Activity Score)", 0.0, 20.0, 5.0)
    Confusion = st.selectbox("Confusion", [0, 1])
    Disorientation = st.selectbox("Disorientation", [0, 1])
    PersonalityChanges = st.selectbox("Personality Changes", [0, 1])
    DifficultyTasks = st.selectbox("Difficulty Completing Tasks", [0, 1])
    Forgetfulness = st.selectbox("Forgetfulness", [0, 1])

# -------------------------------
# Preprocessing Input
# -------------------------------
Gender = 1 if Gender == "Female" else 0

input_data = np.array([[Age, Gender, Ethnicity, EducationLevel, BMI, Smoking, Alcohol,
                        PhysicalActivity, DietQuality, SleepQuality, FamilyHistory,
                        CardiovascularDisease, Diabetes, Depression, HeadInjury,
                        Hypertension, SystolicBP, DiastolicBP, CholTotal, CholLDL,
                        CholHDL, CholTrig, MMSE, Functional, MemoryComplaints,
                        BehavioralProblems, ADL, Confusion, Disorientation,
                        PersonalityChanges, DifficultyTasks, Forgetfulness]])

input_data_scaled = scaler.transform(input_data)

# -------------------------------
# Model Selection
# -------------------------------
model_choice = st.sidebar.selectbox("Choose Model", ["Logistic Regression", "Decision Tree"])
model = log_model if model_choice == "Logistic Regression" else dt_model

# -------------------------------
# Prediction
# -------------------------------
st.subheader("Prediction Results")

if st.button("Predict Alzheimer’s Risk"):
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]  # risk probability

    if prediction == 1:
        st.error(f"⚠ **High Risk of Alzheimer's Disease**")
    else:
        st.success(f"✔ **Low Risk of Alzheimer's Disease**")

    st.write(f"**Risk Probability:** {probability*100:.2f}%")

    # Visual Pie Chart
    fig = px.pie(values=[probability, 1-probability], 
                 names=["Risk", "No Risk"], 
                 color_discrete_sequence=["red", "green"])
    st.plotly_chart(fig)

    st.info("This prediction is based on your trained ML model.")

# -------------------------------
# Footer
# -------------------------------
st.caption("Developed as part of the Alzheimer's Disease ML Classification Project.")


