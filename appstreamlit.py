import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("updatedfraud_model.pkl")
label_encoders = joblib.load("label_encoders.joblib")

# Page setup
st.set_page_config(page_title="Insurance Fraud Detection", page_icon="🚗", layout="centered")

 #Custom Dark Theme CSS - White labels and input text
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #000000;
        color: white !important;
    }

    .stApp {
        background-color: #000000;
        color: white;
    }

    h1, h2, h3, h4, h5, h6, p, label, .css-17eq0hr {
        color: white !important;
    }

    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox div div div,
    .stSelectbox label,
    .stNumberInput label,
    .stRadio label,
    .stSelectbox > label {
        color: white !important;
        background-color: #1e1e1e;
    }

    .stSelectbox div[data-baseweb="select"] {
        background-color: #1e1e1e;
    }

    .stButton > button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.6em 2em;
    }

    .stButton > button:hover {
        background-color: #135d96;
    }
    </style>
""", unsafe_allow_html=True)


# App Title
st.title("🚗 Insurance Fraud Detection")
st.markdown("### 🔍 Predict potential fraud in insurance claims")

# Input layout
st.markdown("#### 📋 Policy & Incident Details")
col1, col2 = st.columns(2)

with col1:
    months_as_customer = st.number_input("Months as Customer", min_value=0)
    policy_csl = st.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
    policy_deductable = st.number_input("Policy Deductible", min_value=0)
    policy_annual_premium = st.number_input("Annual Premium", min_value=0.0)
    umbrella_limit = st.number_input("Umbrella Limit")
    insured_sex = st.selectbox("Gender", ["MALE", "FEMALE"])
    capital_gains = st.number_input("Capital Gains", min_value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0)
    incident_type = st.selectbox("Incident Type", ["Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"])
    collision_type = st.selectbox("Collision Type", ["Front Collision", "NA", "Rear Collision", "Side Collision"])

with col2:
    incident_severity = st.selectbox("Incident Severity", ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"])
    authorities_contacted = st.selectbox("Authorities Contacted", ["Ambulance", "Fire", "NA", "Other", "Police"])
    incident_hour_of_the_day = st.number_input("Incident Hour", min_value=0, max_value=23)
    number_of_vehicles_involved = st.number_input("Vehicles Involved", min_value=1)
    property_damage = st.selectbox("Property Damage", ["NA", "NO", "YES"])
    bodily_injuries = st.number_input("Bodily Injuries", min_value=0)
    witnesses = st.number_input("Witnesses", min_value=0)
    police_report_available = st.selectbox("Police Report", ["NA", "NO", "YES"])
    total_claim_amount = st.number_input("Total Claim", min_value=0.0)
    injury_claim = st.number_input("Injury Claim", min_value=0.0)
    property_claim = st.number_input("Property Claim", min_value=0.0)
    vehicle_claim = st.number_input("Vehicle Claim", min_value=0.0)

# Raw input dict
raw_input = {
    'months_as_customer': months_as_customer,
    'policy_csl': policy_csl,
    'policy_deductable': policy_deductable,
    'policy_annual_premium': policy_annual_premium,
    'umbrella_limit': umbrella_limit,
    'insured_sex': insured_sex,
    'capital-gains': capital_gains,
    'capital-loss': capital_loss,
    'incident_type': incident_type,
    'collision_type': collision_type,
    'incident_severity': incident_severity,
    'authorities_contacted': authorities_contacted,
    'incident_hour_of_the_day': incident_hour_of_the_day,
    'number_of_vehicles_involved': number_of_vehicles_involved,
    'property_damage': property_damage,
    'bodily_injuries': bodily_injuries,
    'witnesses': witnesses,
    'police_report_available': police_report_available,
    'total_claim_amount': total_claim_amount,
    'injury_claim': injury_claim,
    'property_claim': property_claim,
    'vehicle_claim': vehicle_claim
}

# Encode categorical
input_encoded = raw_input.copy()
for col in label_encoders:
    if col in raw_input:
        input_encoded[col] = label_encoders[col].transform([raw_input[col]])[0]

input_df = pd.DataFrame([input_encoded])

# Predict button
if st.button("🔎 Predict Fraud"):
    with st.spinner("Analyzing..."):
        prediction = model.predict(input_df)[0]
        output = label_encoders['fraud_reported'].inverse_transform([prediction])[0] if 'fraud_reported' in label_encoders else prediction

    if output == "Y" or output == 1:
        st.error("🚨 Potential Fraud Detected!")
    else:
        st.success("✅ Claim Seems Legitimate.")


