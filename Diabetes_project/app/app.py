import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Diabetes AI", layout="wide")

# ------------------ SESSION STATE ------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "input"

if "result" not in st.session_state:
    st.session_state.result = None

if "input_data" not in st.session_state:
    st.session_state.input_data = None

# ------------------ LOAD MODEL ------------------
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# ------------------ SIDEBAR ------------------
st.sidebar.title("🧠 Diabetes AI")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Prediction", "ℹ️ About"])

# ------------------ HOME ------------------
if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction System")
    st.subheader("AI-powered health risk detection")

    st.markdown("""
    ### 🚀 Features
    - Machine Learning based prediction  
    - Real-time analysis  
    - Interactive UI  

    ### 🧠 Model Used
    - Random Forest Classifier  

    ### 📊 Input Parameters
    - Glucose, BMI, Age, Blood Pressure, etc.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "86%")
    col2.metric("Model", "Random Forest")
    col3.metric("Dataset Size", "768")

    st.success("👉 Go to Prediction tab to try the model")

# ------------------ PREDICTION ------------------
elif page == "📊 Prediction":

    st.title("📊 Enter Patient Details")

    # TAB SWITCH LOGIC
    if st.session_state.active_tab == "input":
        tab1, tab2 = st.tabs(["📝 Input Data", "📈 Result"])
    else:
        tab2, tab1 = st.tabs(["📈 Result", "📝 Input Data"])

    # ---------------- INPUT TAB ----------------
    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            glucose = st.number_input("Glucose", 0, 200, value=100)
            bp = st.number_input("Blood Pressure", 0, 150, value=80)
            bmi = st.number_input("BMI", 0.0, 70.0, value=25.0)

        with col2:
            if gender == "Male":
                preg = 0
            else:
                preg = st.number_input("Pregnancies", 0, 20, value=1)

            skin = st.number_input("Skin Thickness", 0, 100, value=20)
            insulin = st.number_input("Insulin", 0, 900, value=80)
            age = st.number_input("Age", 1, 120, value=30)

        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, value=0.5)

        if st.button("🚀 Predict", use_container_width=True):

            data = pd.DataFrame(
                [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                columns=[
                    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
                ]
            )

            data = scaler.transform(data)
            result = model.predict(data)

            # STORE RESULT
            st.session_state.result = result[0]
            st.session_state.input_data = (glucose, bmi, bp)

            # SWITCH TAB
            st.session_state.active_tab = "result"
            st.rerun()

    # ---------------- RESULT TAB ----------------
    with tab2:

        if st.session_state.result is not None:

            st.subheader("📊 Prediction Result")

            col1, col2 = st.columns(2)

            if st.session_state.result == 1:
                col1.error("⚠️ High Risk")
                col2.metric("Prediction", "Diabetic")
            else:
                col1.success("✅ Low Risk")
                col2.metric("Prediction", "Not Diabetic")

            st.progress(90 if st.session_state.result == 1 else 30)

            # Chart
            glucose, bmi, bp = st.session_state.input_data

            chart_data = pd.DataFrame({
                "Feature": ["Glucose", "BMI", "BP"],
                "Value": [glucose, bmi, bp]
            })

            st.bar_chart(chart_data.set_index("Feature"))

        else:
            st.info("👉 Enter data and click Predict to see results")

# ------------------ ABOUT ------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About Project")

    st.write("""
    This project predicts diabetes using Machine Learning.

    ### 🧠 Technologies Used:
    - Python  
    - Scikit-learn  
    - Streamlit  

    ### 👨‍💻 Developed By:
    Punit Kumar Singh
    """)

# ------------------ FOOTER ------------------
st.sidebar.markdown("---")
st.sidebar.caption("🚀 Developed by Punit")

