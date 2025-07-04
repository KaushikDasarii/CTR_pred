import streamlit as st
import pandas as pd
import joblib

#  Netflix-style title
st.markdown("<h1 style='color:red;'> Netflix Ads: Click Prediction System</h1>", unsafe_allow_html=True)
st.write("Predict if a user will click on an ad based on device, time, and site info.")

#  Load model and encoders
model = joblib.load("model/xgb_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

#  User Input Form
with st.form("input_form"):
    valid_site_ids = label_encoders["site_id"].classes_.tolist()
    site_id = st.selectbox(" Site ID", valid_site_ids)

    valid_device_models = label_encoders["device_model"].classes_.tolist()
    device_model = st.selectbox("Device Model", valid_device_models)
    hour = st.slider("Hour of the Day", 0, 23, 13)
    banner_pos = st.selectbox(" Banner Position", [0, 1, 2, 3])
    device_type = st.selectbox("Device Type", [1, 2, 4])
    device_conn_type = st.selectbox(" Connection Type", [0, 2, 3])
    C1 = st.number_input(" C1 Code", min_value=1000, max_value=9999, value=1010)

    submitted = st.form_submit_button(" Predict Click Probability")

if submitted:
    # Create input DataFrame
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

    #  Apply label encoders (make sure dtype becomes int)
    try:
        for col in ["site_id", "device_model"]:
            le = label_encoders[col]
            input_df[col] = le.transform([input_df[col][0]]).astype(int)
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    #  Check DataFrame dtypes (debugging)
    # st.write("Debug dtypes:", input_df.dtypes)

    #  Predict
    try:
        prob = model.predict_proba(input_df)[0][1]
        st.success(f" Predicted Click Probability: **{prob:.2f}**")
    except Exception as e:
        st.error(f"Prediction error: {e}")
