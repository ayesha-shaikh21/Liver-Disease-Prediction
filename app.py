import streamlit as st
import pandas as pd
import joblib

# ------------------------
# Load model, feature columns, and scaler
# ------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("liver_disease_risk_model.pkl")
    feature_cols = joblib.load("liver_feature_columns.pkl")
    scaler = joblib.load("liver_scaler.pkl")  # we saved this in notebook
    return model, feature_cols, scaler

model, feature_cols, scaler = load_artifacts()

# ------------------------
# Streamlit UI
# ------------------------
st.title("🩺 Liver Disease Risk Prediction")
st.write("Enter patient information to predict liver disease risk (educational use only).")

# Inputs based on your actual columns:
# ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
#  'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
#  'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
#  'Albumin_and_Globulin_Ratio']

age = st.number_input("Age", min_value=1, max_value=120, value=45)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0

total_bilirubin = st.number_input("Total_Bilirubin", min_value=0.0, value=1.0, step=0.1)
direct_bilirubin = st.number_input("Direct_Bilirubin", min_value=0.0, value=0.5, step=0.1)
alk_phos = st.number_input("Alkaline_Phosphotase", min_value=0, value=200, step=10)
alt = st.number_input("Alamine_Aminotransferase", min_value=0, value=30, step=1)
ast = st.number_input("Aspartate_Aminotransferase", min_value=0, value=35, step=1)
total_protiens = st.number_input("Total_Protiens", min_value=0.0, value=6.5, step=0.1)
albumin = st.number_input("Albumin", min_value=0.0, value=3.5, step=0.1)
ag_ratio = st.number_input("Albumin_and_Globulin_Ratio", min_value=0.0, value=1.0, step=0.1)

# Build a single-row DataFrame
input_dict = {
    "Age": age,
    "Gender": gender_val,
    "Total_Bilirubin": total_bilirubin,
    "Direct_Bilirubin": direct_bilirubin,
    "Alkaline_Phosphotase": alk_phos,
    "Alamine_Aminotransferase": alt,
    "Aspartate_Aminotransferase": ast,
    "Total_Protiens": total_protiens,
    "Albumin": albumin,
    "Albumin_and_Globulin_Ratio": ag_ratio
}

input_df = pd.DataFrame([input_dict])

# Ensure same column order as during training
input_df = input_df[feature_cols]

st.subheader("Input Summary")
st.write(input_df)

# ------------------------
# Prediction
# ------------------------
if st.button("Predict Liver Disease Risk"):
    # Scale input (because you trained on scaled data)
    X_scaled = scaler.transform(input_df)

    # Predict probability (risk) and class
    proba = model.predict_proba(X_scaled)[0, 1]  # probability of class 1 (disease)
    pred_class = model.predict(X_scaled)[0]

    risk_percent = proba * 100

    st.subheader("Prediction Result")

    if pred_class == 1:
        st.error(f"⚠ Model prediction: **Liver Disease**\n\nRisk Score: **{risk_percent:.2f}%**")
    else:
        st.success(f"✅ Model prediction: **No Liver Disease**\n\nRisk Score: **{risk_percent:.2f}%**")

    st.caption("This is a machine learning demo and not a medical diagnosis.")
