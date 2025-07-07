import streamlit as st
import tensorflow as tf
import base64
from google.cloud import aiplatform

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="centered")
st.title("💰 Loan Approval Probability")
st.markdown("Predict your loan approval chance based on your profile and finances.")

age = st.slider("Age", 18, 100, 30)
income = st.number_input("Annual Income ($)", min_value=1000.0, value=50000.0)
emp_exp = st.slider("Years of Employment", 0, 40, 5)
loan_amt = st.number_input("Loan Amount ($)", min_value=100.0, value=10000.0)
int_rate = st.slider("Loan Interest Rate (%)", 0.0, 50.0, 10.5)
loan_percent_income = loan_amt / (income + 1e-5)
cred_len = st.slider("Credit History Length (Years)", 0, 30, 5)
credit_score = st.slider("Credit Score", 300, 850, 600)
gender = st.selectbox("Gender", ["male", "female"])
education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"])
loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "PERSONAL", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
defaults = st.selectbox("Previous Defaults on File", ["Yes", "No"])

if st.button("🔍 Predict Approval Probability"):
    example = tf.train.Example(features=tf.train.Features(feature={
        'person_age': tf.train.Feature(float_list=tf.train.FloatList(value=[age])),
        'person_income': tf.train.Feature(float_list=tf.train.FloatList(value=[income])),
        'person_emp_exp': tf.train.Feature(int64_list=tf.train.Int64List(value=[emp_exp])),
        'loan_amnt': tf.train.Feature(float_list=tf.train.FloatList(value=[loan_amt])),
        'loan_int_rate': tf.train.Feature(float_list=tf.train.FloatList(value=[int_rate])),
        'loan_percent_income': tf.train.Feature(float_list=tf.train.FloatList(value=[loan_percent_income])),
        'cb_person_cred_hist_length': tf.train.Feature(float_list=tf.train.FloatList(value=[cred_len])),
        'credit_score': tf.train.Feature(int64_list=tf.train.Int64List(value=[credit_score])),
        'person_gender': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gender.encode()])),
        'person_education': tf.train.Feature(bytes_list=tf.train.BytesList(value=[education.encode()])),
        'person_home_ownership': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ownership.encode()])),
        'loan_intent': tf.train.Feature(bytes_list=tf.train.BytesList(value=[loan_intent.encode()])),
        'previous_loan_defaults_on_file': tf.train.Feature(bytes_list=tf.train.BytesList(value=[defaults.encode()]))
    }))
    
    serialized = example.SerializeToString()
    encoded = base64.b64encode(serialized).decode("utf-8")

    aiplatform.init(project="analog-arbor-464806-n7", location="us-central1")
    endpoint = aiplatform.Endpoint("projects/129249538815/locations/us-central1/endpoints/7588949101677379584")

    try:
        prediction = endpoint.predict(instances=[{"b64": encoded}])
        prob = prediction.predictions[0][0]
        st.success(f"💡 Loan Approval Probability: **{prob * 100:.2f}%**")
        if prob > 0.7:
            st.markdown("✅ Likely to be **approved**.")
        else:
            st.markdown("⚠️ Might be **rejected**, consider improving your score or income.")
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
