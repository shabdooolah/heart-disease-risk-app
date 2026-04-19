import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained pipeline
model = joblib.load("heart_pipeline.pkl")

st.title("Heart Disease Prediction Dashboard")
st.write("This system predicts the risk of heart disease using a trained machine learning model.")

st.header("Enter Patient Details")

# -------- Numeric Inputs --------
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
physical_health = st.slider("Physical Health (days)", 0, 30, 0)
mental_health = st.slider("Mental Health (days)", 0, 30, 0)
sleep_time = st.slider("Sleep Time (hours)", 0, 24, 7)

# -------- Yes/No Inputs --------
smoking = st.selectbox("Smoking", ["Yes", "No"])
smoking = 1 if smoking == "Yes" else 0

alcohol = st.selectbox("Alcohol Drinking", ["Yes", "No"])
alcohol = 1 if alcohol == "Yes" else 0

stroke = st.selectbox("Stroke", ["Yes", "No"])
stroke = 1 if stroke == "Yes" else 0

diff_walking = st.selectbox("Difficulty Walking", ["Yes", "No"])
diff_walking = 1 if diff_walking == "Yes" else 0

physical_activity = st.selectbox("Physical Activity", ["Yes", "No"])
physical_activity = 1 if physical_activity == "Yes" else 0

asthma = st.selectbox("Asthma", ["Yes", "No"])
asthma = 1 if asthma == "Yes" else 0

kidney_disease = st.selectbox("Kidney Disease", ["Yes", "No"])
kidney_disease = 1 if kidney_disease == "Yes" else 0

skin_cancer = st.selectbox("Skin Cancer", ["Yes", "No"])
skin_cancer = 1 if skin_cancer == "Yes" else 0

# -------- Categorical Inputs --------
sex = st.selectbox("Sex", ["Male", "Female"])

age_category = st.selectbox(
    "Age Category",
    ["18-24","25-29","30-34","35-39","40-44","45-49",
     "50-54","55-59","60-64","65-69","70-74","75-79","80 or older"]
)

race = st.selectbox(
    "Race",
    ["White","Black","Asian","American Indian/Alaska Native","Hispanic","Other"]
)

diabetic = st.selectbox("Diabetic", ["Yes","No"])

gen_health = st.selectbox(
    "General Health",
    ["Poor","Fair","Good","Very good","Excellent"]
)

# -------- Prediction Button --------
if st.button("Predict Heart Disease Risk"):

    input_data = pd.DataFrame({
        "BMI": [bmi],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol],
        "Stroke": [stroke],
        "PhysicalHealth": [physical_health],
        "MentalHealth": [mental_health],
        "DiffWalking": [diff_walking],
        "Sex": [sex],
        "AgeCategory": [age_category],
        "Race": [race],
        "Diabetic": [diabetic],
        "PhysicalActivity": [physical_activity],
        "GenHealth": [gen_health],
        "SleepTime": [sleep_time],
        "Asthma": [asthma],
        "KidneyDisease": [kidney_disease],
        "SkinCancer": [skin_cancer]
    })

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    risk = probability[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"Heart Disease Probability: **{risk*100:.2f}%**")
    st.subheader("Risk Meter")

    st.progress(int(risk * 100))

    st.write(f"Heart Disease Risk Level: {risk*100:.2f}%")
    

    # -------- Visualization --------
    st.subheader("Risk Visualization")

    fig, ax = plt.subplots()
    ax.bar(["Low Risk", "High Risk"], [1-risk, risk])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    ax.set_title("Heart Disease Risk")

    st.pyplot(fig)
    st.subheader("Patient Data Used For Prediction")
    st.dataframe(input_data)

# -------- Project Info --------
st.sidebar.title("Project Info")

st.sidebar.write("""
**Machine Learning Model:** Logistic Regression  
**Technique:** Pipeline + OneHotEncoding  
**Goal:** Predict heart disease risk based on lifestyle and health data.
""")