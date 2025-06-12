from flask import Flask, request, jsonify
from predict import predict_fraud

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = predict_fraud(data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
