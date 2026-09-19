# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 00:32:40 2026

@author: 10191
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 18:36:29 2026

@author: 10191
"""

import streamlit as st
import numpy as np
import xgboost as xgb
import os

# =========================
# Page settings
# =========================
st.set_page_config(
    page_title="TNBC Liver Metastasis Risk Calculator",
    layout="centered"
)

st.title("TNBC Liver Metastasis Risk Calculator")

st.write(
    "This web-based calculator estimates the probability of Liver metastasis "
    "in patients with TNBC using an XGBoost model."
)

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Patient Variables")

# Age
Age = st.sidebar.selectbox(
    "Age",
    ["<50", "≥50"]
)

Age_map = {
    "<50": 0,
    "≥50": 1
}

Rural_Urban = st.sidebar.selectbox(
    "Rural_Urban",
    ["<100", "≥100"]
)

Rural_Urban_map = {
    "<100": 0,
    "≥100": 1
}



# Surgery
Radiation = st.sidebar.selectbox(
    "Radiation",
    ["No", "Yes"]
)

Radiation_map = {
    "No": 0,
    "Yes": 1
}

# Tumor size
Chemotherapy = st.sidebar.selectbox(
    "Chemotherapy ",
    ["No", "Yes"]
)

Chemotherapy_map = {
    "No": 0,
    "Yes": 1
}

# T stage
PR= st.sidebar.selectbox(
    "PR",
    ["Negative", "Positive"]
)

PR_map = {
    "Negative": 0,
    "Positive": 1
}

HER2= st.sidebar.selectbox(
    "HER2",
    ["Negative", "Positive"]
)

HER2_map = {
    "Negative": 0,
    "Positive": 1
}


# N stage
Grade = st.sidebar.selectbox(
    "Grade",
    ["Grade I", "Grade II", "Grade III", "Grade IV"]
)

Grade_map = {
    "Grade I": 0,
    "Grade II": 1,
    "Grade III": 2,
    "Grade IV": 3
}


T_stage = st.sidebar.selectbox(
    "T_stage",
    ["T1", "T2", "T3", "T4"]
)

T_stage_map = {
    "T1": 0,
    "T2": 1,
    "T3": 2,
    "T4": 3
}


N_stage = st.sidebar.selectbox(
    "N_stage",
    ["N0", "N1", "N2", "N3"]
)

N_stage_map = {
    "N0": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3
}

Surgery = st.sidebar.selectbox(
    "Surgery",
    ["No", "Yes"]
)

Surgery_map = {
    "No": 0,
    "Yes": 1
}



# =========================
# Construct model input
# =========================
x = np.array([
    Age_map[Age],
    Rural_Urban_map[Rural_Urban],
    Radiation_map[Radiation],
    Chemotherapy_map[Chemotherapy],
    PR_map[PR],
    HER2_map[HER2],
    Grade_map[Grade],
    T_stage_map[T_stage],
    N_stage_map[N_stage],
    Surgery_map[Surgery],
]).reshape(1, 9)

# =========================
# Load model
# =========================
MODEL_FILE = "modelTNBC.json"

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    return model

# =========================
# Prediction
# =========================
if st.button("Predict"):

    if not os.path.exists(MODEL_FILE):
        st.error("Model file 'modelTNBC.json' was not found.")

    else:
        try:
            modelXGB = load_model()

            y_pred = modelXGB.predict_proba(x)

            probability = float(y_pred[0, 1]) * 100

            st.subheader("Prediction Result")

            st.metric(
                label="Probability of Liver Metastasis",
                value=f"{probability:.2f}%"
            )

            if probability >= 50:
                st.warning(
                    "The predicted probability of Liver metastasis is relatively high."
                )
            else:
                st.success(
                    "The predicted probability of Liver metastasis is relatively low."
                )

        except Exception as e:
            st.error("An error occurred while loading the model or making the prediction.")
            st.exception(e)

# =========================
# Disclaimer
# =========================
st.markdown("---")

st.caption(
    "Disclaimer: This calculator is intended for research purposes only "
    "and should not replace clinical judgment."
)