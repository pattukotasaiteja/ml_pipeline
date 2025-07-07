import os
import streamlit as st
import base64
import json
import tensorflow as tf
from google.cloud import aiplatform

# --------------------------------------
# 1. Load credentials from secrets
# --------------------------------------
with open("service_account.json", "w") as f:
    json.dump(st.secrets["gcp_service_account"], f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_account.json"

# --------------------------------------
# 2. Init Vertex AI and Endpoint
# --------------------------------------
PROJECT = "analog-arbor-464806-n7"
LOCATION = "us-central1"
ENDPOINT_ID = "4218004780590563328"

aiplatform.init(project=PROJECT, location=LOCATION)
endpoint = aiplatform.Endpoint(f"projects/129249538815/locations/{LOCATION}/endpoints/{ENDPOINT_ID}")

# --------------------------------------
# 3. Streamlit UI
# --------------------------------------
st.title("📊 Loan Approval Predictor")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", value=25)
    income = st.number_input("Income", value=50000.0)
    experience = st.number_input("Employment Experience (years)", value=1)
    loan_amount = st.number_input("Loan Amount", value=15000.0)

with col2:
    int_rate = st.number_input("Interest Rate (%)", value=12.5)
    cred_hist = st.slider("Credit History Length (Years)", 0, 20, 5)
    credit_score = st.slider("Credit Score", 100, 850, 650)

loan_percent_income = round(loan_amount / income, 2) if income > 0 else 0.0

gender = st.selectbox("Gender", ["male", "female"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master"])
home = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE"])
intent = st.selectbox("Loan Intent", ["EDUCATION", "PERSONAL", "VENTURE"])
defaults = st.selectbox("Previous Defaults", ["Yes", "No"])

# --------------------------------------
# 4. Predict Button
# --------------------------------------
if st.button("🔍 Predict Approval Probability"):
    try:
        # Build example
        example = tf.train.Example(features=tf.train.Features(feature={
            "person_age": tf.train.Feature(float_list=tf.train.FloatList(value=[age])),
            "person_income": tf.train.Feature(float_list=tf.train.FloatList(value=[income])),
            "person_emp_exp": tf.train.Feature(int64_list=tf.train.Int64List(value=[experience])),
            "loan_amnt": tf.train.Feature(float_list=tf.train.FloatList(value=[loan_amount])),
            "loan_int_rate": tf.train.Feature(float_list=tf.train.FloatList(value=[int_rate])),
            "loan_percent_income": tf.train.Feature(float_list=tf.train.FloatList(value=[loan_percent_income])),
            "cb_person_cred_hist_length": tf.train.Feature(float_list=tf.train.FloatList(value=[cred_hist])),
            "credit_score": tf.train.Feature(int64_list=tf.train.Int64List(value=[credit_score])),
            "person_gender": tf.train.Feature(bytes_list=tf.train.BytesList(value=[gender.encode()])),
            "person_education": tf.train.Feature(bytes_list=tf.train.BytesList(value=[education.encode()])),
            "person_home_ownership": tf.train.Feature(bytes_list=tf.train.BytesList(value=[home.encode()])),
            "loan_intent": tf.train.Feature(bytes_list=tf.train.BytesList(value=[intent.encode()])),
            "previous_loan_defaults_on_file": tf.train.Feature(bytes_list=tf.train.BytesList(value=[defaults.encode()]))
        }))

        # Serialize and encode
        serialized = example.SerializeToString()
        encoded = base64.b64encode(serialized).decode("utf-8")

        # Predict
        response = endpoint.predict(instances=[{"b64": encoded}])
        probability = round(response.predictions[0][0] * 100, 2)
        st.success(f"✅ Loan Approval Probability: **{probability}%**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
