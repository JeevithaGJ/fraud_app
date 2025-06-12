from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load("updatedfraud_model.pkl")
label_encoders = joblib.load("label_encoders.joblib")


@app.route('/', methods=['GET', 'POST'])
def index2():
    prediction_result = None

    if request.method == 'POST':
        # Collect input values from form
        form = request.form
        raw_input = {
            'months_as_customer': int(form['months_as_customer']),
            'policy_csl': form['policy_csl'],
            'policy_deductable': int(form['policy_deductable']),
            'policy_annual_premium': float(form['policy_annual_premium']),
            'umbrella_limit': int(form['umbrella_limit']),
            'insured_sex': form['insured_sex'],
            'capital-gains': int(form['capital_gains']),
            'capital-loss': int(form['capital_loss']),
            'incident_type': form['incident_type'],
            'collision_type': form['collision_type'],
            'incident_severity': form['incident_severity'],
            'authorities_contacted': form['authorities_contacted'],
            'incident_hour_of_the_day': int(form['incident_hour_of_the_day']),
            'number_of_vehicles_involved': int(form['number_of_vehicles_involved']),
            'property_damage': form['property_damage'],
            'bodily_injuries': int(form['bodily_injuries']),
            'witnesses': int(form['witnesses']),
            'police_report_available': form['police_report_available'],
            'total_claim_amount': float(form['total_claim_amount']),
            'injury_claim': float(form['injury_claim']),
            'property_claim': float(form['property_claim']),
            'vehicle_claim': float(form['vehicle_claim'])
        }

        # Encode categorical variables
        input_encoded = raw_input.copy()
        for col in label_encoders:
            if col in input_encoded:
                input_encoded[col] = label_encoders[col].transform([input_encoded[col]])[0]

        input_df = pd.DataFrame([input_encoded])

        # Prediction
        prediction = model.predict(input_df)[0]
        output = label_encoders['fraud_reported'].inverse_transform([prediction])[0] if 'fraud_reported' in label_encoders else prediction

        prediction_result = "Fraud Detected!" if output == "Y" or output == 1 else "Claim Seems Legitimate."

    return render_template("index2.html", result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
