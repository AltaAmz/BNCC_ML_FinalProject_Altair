import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Obesity Level Classifier",
    page_icon="🏥",
    layout="centered"
)

MODEL_PATH = "best_obesity_pipeline.joblib"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "Model file 'best_obesity_pipeline.joblib' was not found. "
        "Please place the model file in the same folder as app.py."
    )
    st.stop()

LABELS = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

st.title("🏥 Obesity Level Classifier")

st.write(
    """
    This application predicts an individual's obesity level
    using anthropometric measurements and lifestyle-related
    factors through a trained machine learning pipeline.
    """
)

st.info(
    """
    **Academic Project Disclaimer**

    This application is developed for machine-learning project
    demonstration purposes. The prediction is not a medical
    diagnosis and should not replace professional medical advice.
    """
)


with st.form("obesity_prediction_form"):

    st.header("👤 Personal Information")

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=1.0,
        max_value=100.0,
        value=21.0,
        step=1.0
    )

    height = st.number_input(
        "Height (meters)",
        min_value=1.0,
        max_value=2.5,
        value=1.65,
        step=0.01
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=20.0,
        max_value=250.0,
        value=60.0,
        step=0.5
    )

    st.header("🍽️ Eating Habits")

    family_history = st.selectbox(
        "Family history with overweight",
        ["yes", "no"]
    )

    favc = st.selectbox(
        "Frequent consumption of high-caloric food (FAVC)",
        ["yes", "no"]
    )

    fcvc = st.number_input(
        "Frequency of vegetable consumption (FCVC)",
        min_value=0.0,
        max_value=3.0,
        value=2.0,
        step=0.1
    )

    ncp = st.number_input(
        "Number of main meals per day (NCP)",
        min_value=0.0,
        max_value=6.0,
        value=3.0,
        step=0.1
    )

    caec = st.selectbox(
        "Consumption of food between meals (CAEC)",
        ["no", "Sometimes", "Frequently", "Always"]
    )

    calc = st.selectbox(
        "Alcohol consumption (CALC)",
        ["no", "Sometimes", "Frequently", "Always"]
    )

    st.header("🏃 Lifestyle")

    smoke = st.selectbox(
        "Smoking (SMOKE)",
        ["yes", "no"]
    )

    ch2o = st.number_input(
        "Daily water consumption (CH2O)",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.1
    )

    scc = st.selectbox(
        "Calories consumption monitoring (SCC)",
        ["yes", "no"]
    )

    faf = st.number_input(
        "Physical activity frequency (FAF)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    tue = st.number_input(
        "Time using technology devices (TUE)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

    mtrans = st.selectbox(
        "Main transportation mode (MTRANS)",
        [
            "Automobile",
            "Bike",
            "Motorbike",
            "Public_Transportation",
            "Walking"
        ]
    )

    submitted = st.form_submit_button(
        "🔮 Predict Obesity Level"
    )

if submitted:

    input_data = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history_with_overweight": family_history,
        "FAVC": favc,
        "FCVC": fcvc,
        "NCP": ncp,
        "CAEC": caec,
        "SMOKE": smoke,
        "CH2O": ch2o,
        "SCC": scc,
        "FAF": faf,
        "TUE": tue,
        "CALC": calc,
        "MTRANS": mtrans
    }])

    try:
        prediction = model.predict(input_data)[0]
        prediction = int(prediction)

        predicted_label = LABELS.get(
            prediction,
            "Unknown Class"
        )

        st.divider()
        st.header("📊 Prediction Result")
        st.success(f"### Predicted Obesity Level: {predicted_label}")

        bmi = weight / (height ** 2)

        st.metric(label="BMI", value=f"{bmi:.2f}")

        st.caption(
            "BMI is shown as additional information. "
            "It is not added as a separate model feature."
        )

        with st.expander("🔎 View Input Data"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error("An error occurred while making the prediction.")
        st.exception(error)

st.divider()

st.caption(
    "Machine Learning Final Project — Obesity Level Classification"
)