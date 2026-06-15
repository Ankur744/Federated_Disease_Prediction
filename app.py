from __future__ import annotations

import numpy as np
import streamlit as st

from src.data import FEATURE_NAMES, fit_standard_scaler, generate_healthcare_dataset, train_test_split
from src.federated import train_federated


@st.cache_resource
def build_model():
    x, y = generate_healthcare_dataset(samples=2500, seed=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, seed=42)
    scaler = fit_standard_scaler(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    clients = []
    for chunk in np.array_split(np.arange(len(y_train)), 5):
        clients.append((x_train_scaled[chunk], y_train[chunk]))

    model, _ = train_federated(
        clients=clients,
        x_test=x_test_scaled,
        y_test=y_test,
        rounds=20,
        local_epochs=3,
        dp_noise=0.01,
        seed=42,
    )
    return model, scaler


st.set_page_config(page_title="Federated Disease Prediction", layout="centered")
st.title("Federated Disease Prediction")
st.caption("Privacy-preserving disease risk prediction using simulated federated learning")

model, scaler = build_model()

age = st.slider("Age", 18, 85, 48)
bmi = st.slider("BMI", 16.0, 45.0, 27.0)
blood_pressure = st.slider("Blood Pressure", 80, 190, 125)
glucose = st.slider("Glucose", 60, 260, 115)
cholesterol = st.slider("Cholesterol", 110, 340, 205)
insulin = st.slider("Insulin", 10, 260, 85)
family_history = st.selectbox("Family History", [0, 1], format_func=lambda value: "Yes" if value else "No")
physical_activity = st.slider("Physical Activity Hours/Week", 0.0, 10.0, 4.2)

patient = np.array(
    [[age, bmi, blood_pressure, glucose, cholesterol, insulin, family_history, physical_activity]],
    dtype=float,
)
patient_scaled = scaler.transform(patient)
risk = float(model.predict_proba(patient_scaled)[0])

st.metric("Predicted Disease Risk", f"{risk * 100:.2f}%")
st.progress(min(max(risk, 0.0), 1.0))

if risk >= 0.5:
    st.warning("High-risk prediction. Clinical confirmation is required.")
else:
    st.success("Low-risk prediction. Continue preventive monitoring.")

st.subheader("Input Features")
st.write(dict(zip(FEATURE_NAMES, patient[0].tolist())))
