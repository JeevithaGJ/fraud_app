import joblib
import pandas as pd
import numpy as np

# Load saved model
model = joblib.load('fraud_model.pkl')

def preprocess_input(data):
    df = pd.DataFrame([data])

    # Drop unused columns (safe removal)
    drop_cols = ['policy_number','policy_bind_date','policy_state','incident_type',
                 'insured_zip','insured_hobbies','incident_state','incident_city',
                 'incident_location','auto_make','auto_model','incident_date','_c39']
    for col in drop_cols:
        df.pop(col, None)

    # Handle missing values
    df['collision_type'] = df['collision_type'].fillna('NA')
    df['property_damage'] = df['property_damage'].fillna('NA')
    df['police_report_available'] = df['police_report_available'].fillna('NA')
    df['authorities_contacted'] = df['authorities_contacted'].fillna('NA')

    # Label encoding with predefined mappings
    label_maps = {
        'insured_sex': {'MALE': 1, 'FEMALE': 0},
        'insured_education_level': {'High School': 0, 'Associate': 1, 'College': 2, 'Masters': 3, 'PhD': 4},
        'insured_occupation': {'clerical': 0, 'doctor': 1, 'manager': 2, 'lawyer': 3, 'blue-collar': 4, 'executive': 5, 'entrepreneur': 6, 'student': 7},
        'insured_relationship': {'husband': 0, 'wife': 1, 'own-child': 2, 'other-relative': 3, 'unmarried': 4, 'not-in-family': 5},
        'collision_type': {'Rear Collision': 0, 'Side Collision': 1, 'Front Collision': 2, 'NA': 3},
        'incident_severity': {'Minor Damage': 0, 'Major Damage': 1, 'Total Loss': 2, 'Trivial Damage': 3},
        'authorities_contacted': {'Police': 0, 'Fire': 1, 'Ambulance': 2, 'Other': 3, 'NA': 4},
        'property_damage': {'YES': 1, 'NO': 0, 'NA': 2},
        'police_report_available': {'YES': 1, 'NO': 0, 'NA': 2}
    }
    for col, mapping in label_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Feature engineering
    df['umbrella_limit_log'] = np.log1p(df['umbrella_limit'])
    df['claim_per_vehicle'] = df['total_claim_amount'] / df['vehicle_count']

    # Select final features
    features = ['umbrella_limit_log', 'claim_per_vehicle','months_as_customer','age','policy_csl',
                'policy_deductable','policy_annual_premium','insured_sex','insured_education_level',
                'insured_occupation','insured_relationship','capital-gains','capital-loss',
                'collision_type','incident_severity','authorities_contacted','incident_hour_of_the_day',
                'number_of_vehicles_involved','property_damage','bodily_injuries','witnesses',
                'police_report_available','total_claim_amount','injury_claim','property_claim',
                'vehicle_claim','auto_year']
    return df[features]

def predict_fraud(data):
    df = preprocess_input(data)
    prediction = model.predict(df)
    return int(prediction[0])
