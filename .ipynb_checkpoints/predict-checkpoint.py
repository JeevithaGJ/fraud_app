import joblib
import pandas as pd
import numpy as np

# Load saved model
model = joblib.load('fraud_model.pkl')

# Adjust features list below to match your model training features
def preprocess_input(data):
    df = pd.DataFrame([data])
    df['umbrella_limit_log'] = np.log1p(df['umbrella_limit'])
    df['claim_per_vehicle'] = df['total_claim_amount'] / df['vehicle_count']
    features = ['umbrella_limit_log', 'claim_per_vehicle']  # Add all real feature names here
    return df[features]

def predict_fraud(data):
    df = preprocess_input(data)
    prediction = model.predict(df)
    return int(prediction[0])
