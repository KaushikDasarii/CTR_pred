import streamlit as st
import pandas as pd
import joblib

# 🎬 Netflix-style title
st.markdown("<h1 style='color:red;'>🎬 Netflix Ads: Click Prediction System</h1>", unsafe_allow_html=True)
st.write("Predict if a user will click on an ad based on device, time, and site info.")

# 🔍 Load model and encoders
model = joblib.load("model/xgb_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

# ✍️ Form Input
with st.form("input_form"):
    site_id = st.selectbox("📍 Site ID", ['1fbe01fe', '85f751fd', 'e151e245', 'fe6b92e5'])
    device_model = st.selectbox("📱 Device Model", ['dlcecap1', 'dlcecap2', 'rmx1001', 'iphone12'])

    hour = st.slider("🕓 Hour of the Day", 0, 23, 13)
    banner_pos = st.selectbox("🪧 Banner Position", [0, 1, 2, 3])
    device_type = st.selectbox("💻 Device Type", [1, 2, 4])
    device_conn_type = st.selectbox("🌐 Connection Type", [0, 2, 3])
    C1 = st.number_input("🔐 C1 Code", min_value=1000, max_value=9999, value=1010)

    submitted = st.form_submit_button("🚀 Predict Click Probability")

# 🛠️ Format Input Data
if submitted:
    input_data = {
        "site_id": site_id,
        "device_model": device_model,
        "hour": hour,
        "banner_pos": banner_pos,
        "device_type": device_type,
        "device_conn_type": device_conn_type,
        "C1": C1
    }

    input_df = pd.DataFrame([input_data])

    # 🔁 Apply label encoders
    try:
        for col in ["site_id", "device_model"]:
            input_df[col] = label_encoders[col].transform(input_df[col])
    except Exception as e:
        st.error(f"Encoding error: {e}")

    # 📈 Predict
    try:
        prob = model.predict_proba(input_df)[0][1]
        st.success(f"🎯 Predicted Click Probability: **{prob:.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
