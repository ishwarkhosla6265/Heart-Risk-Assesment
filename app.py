import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. SETTING UP THE PAGE ---
st.set_page_config(page_title="Heart Risk AI", page_icon="❤️")
st.title("❤️ Heart Disease Clinical Risk Assessment")
st.write("This tool uses a Logistic Regression model to calculate the statistical probability of heart disease based on patient vitals.")

# --- 2. LOADING THE ASSETS ---
# These must be in the same folder as this .py file!
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('heart_scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

# --- 3. CREATING THE USER INTERFACE ---
st.sidebar.header("Patient Vitals")

def get_user_input():
    st.sidebar.header("Patient Vitals")

    # 1. Numerical Inputs (Keep as is)
    age = st.sidebar.slider("Age", 20, 80, 50)
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 6.2, 1.0)
    
    # 2. Categorical Inputs with Layman Labels
    
    # Sex Mapping
    sex_map = {"Male": 1, "Female": 0}
    sex_label = st.sidebar.selectbox("Sex", options=list(sex_map.keys()))
    sex = sex_map[sex_label]

    # Chest Pain Mapping
    cp_map = {
        "Typical Angina (Heart Related)": 0,
        "Atypical Angina (Non-Heart Related)": 1,
        "Non-anginal Pain (Chest Pain)": 2,
        "Asymptomatic (No Pain)": 3
    }
    cp_label = st.sidebar.selectbox("Chest Pain Type", options=list(cp_map.keys()))
    cp = cp_map[cp_label]

    # Fasting Blood Sugar Mapping
    fbs_map = {"Yes (Above 120 mg/dl)": 1, "No (Below 120 mg/dl)": 0}
    fbs_label = st.sidebar.selectbox("Fasting Blood Sugar High?", options=list(fbs_map.keys()))
    fbs = fbs_map[fbs_label]

    # Resting ECG Mapping
    ecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    ecg_label = st.sidebar.selectbox("Resting ECG Results", options=list(ecg_map.keys()))
    restecg = ecg_map[ecg_label]

    # Exercise Induced Angina
    exang_map = {"Yes": 1, "No": 0}
    exang_label = st.sidebar.selectbox("Exercise Induced Chest Pain?", options=list(exang_map.keys()))
    exang = exang_map[exang_label]

    # Slope Mapping
    slope_map = {"Upsloping (Better)": 0, "Flat": 1, "Downsloping (Worse)": 2}
    slope_label = st.sidebar.selectbox("Slope of Peak Exercise ST", options=list(slope_map.keys()))
    slope = slope_map[slope_label]

    # Major Vessels
    ca = st.sidebar.selectbox("Number of Major Vessels Visible (0-3)", options=[0, 1, 2, 3])

    # Thalassemia Mapping
    thal_map = {"Fixed Defect": 1, "Normal": 2, "Reversible Defect": 3}
    thal_label = st.sidebar.selectbox("Thalassemia Status", options=list(thal_map.keys()))
    thal = thal_map[thal_label]

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

df_input = get_user_input()

# --- 4. PREPROCESSING (The "Cleaning" Step) ---
if st.button("Generate Risk Report"):
    # A. Capping Outliers (Using your 99th percentile logic)
    df_input['chol'] = np.where(df_input['chol'] > 406, 406, df_input['chol'])
    df_input['trestbps'] = np.where(df_input['trestbps'] > 180, 180, df_input['trestbps'])
    
    # B. Log Transformation for Oldpeak
    df_input['oldpeak_log'] = np.log1p(df_input['oldpeak'])
    df_input.drop('oldpeak', axis=1, inplace=True)
    
    # C. One-Hot Encoding (Creating the _1, _2 columns)
    df_encoded = pd.get_dummies(df_input, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
    
    # D. Aligning Columns (Matches the encoded patient to the training columns)
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # E. Scaling
    scaled_data = scaler.transform(df_final)
    
    # --- 5. PREDICTION & DISPLAY ---
    risk_prob = model.predict_proba(scaled_data)[:, 1][0] * 100
    
    st.subheader("Results")
    if risk_prob >= 30:
        st.error(f"High Risk Detected: {risk_prob:.2f}%")
        st.write("📢 **Recommendation:** Consult a specialist immediately.")
    else:
        st.success(f"Low/Moderate Risk: {risk_prob:.2f}%")
        st.write("✅ **Recommendation:** Routine check-up and healthy lifestyle.")