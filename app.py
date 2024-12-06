from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
import pickle
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load model and vectorizer with error handling
try:
    Classifier = pickle.load(open('model.pkl', 'rb'))
    Vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError as e:
    logging.error(f"Error: {e}. Ensure 'model.pkl' and 'vectorizer.pkl' exist.")
    exit(1)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            # Get user input
            message = request.form['message']
            logging.info(f"Received message: {message}")
            vectorized_message = Vectorizer.transform([message])
            predict = Classifier.predict(vectorized_message)[0]
            predict_proba = Classifier.predict_proba(vectorized_message).tolist()
            return render_template('index.html', message=message, predict=predict, predict_proba=predict_proba)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return render_template('index.html', error=str(e))
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid input. Please provide a "message" key.'}), 400

    try:
        message = data['message']
        vectorized_message = Vectorizer.transform([message])
        predict = Classifier.predict(vectorized_message)[0]
        predict_proba = Classifier.predict_proba(vectorized_message).tolist()
        return jsonify({'message': message, 'predict': predict, 'predict_proba': predict_proba})
    except Exception as e:
        logging.error(f"Error during API prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
