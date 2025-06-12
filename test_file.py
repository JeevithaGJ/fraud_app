import requests

url = "http://127.0.0.1:5000/"
data = {
    "months_as_customer": 52,
    "age": 41,
    "policy_csl": 250000,
    "policy_deductable": 1000,
    "policy_annual_premium": 1000.50,
    "insured_sex": "MALE",
    "insured_education_level": "College",
    "insured_occupation": "manager",
    "insured_relationship": "husband",
    "capital-gains": 0,
    "capital-loss": 0,
    "collision_type": "Side Collision",
    "incident_severity": "Minor Damage",
    "authorities_contacted": "Police",
    "incident_hour_of_the_day": 15,
    "number_of_vehicles_involved": 2,
    "property_damage": "YES",
    "bodily_injuries": 0,
    "witnesses": 2,
    "police_report_available": "YES",
    "total_claim_amount": 7000.0,
    "injury_claim": 2000.0,
    "property_claim": 1500.0,
    "vehicle_claim": 3500.0,
    "auto_year": 2015,
    "umbrella_limit": 1000000,
    "vehicle_count": 1
}

response = requests.post(url, json=data)
print("Prediction:", response.json())
