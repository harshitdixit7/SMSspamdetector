import os
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd

app = Flask(__name__)

# Load data
data = pd.read_csv('spam.csv', encoding='latin-1')

# Check if the column names match what you're expecting
print(data.columns)  # For debugging, ensure 'v1' is the label and 'v2' is the text

# Splitting the dataset
train_data = data[:4400]  # 4400 items for training
test_data = data[4400:]  # 1172 items for testing

# Train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()

# Vectorizing text data
vectorize_text = Vectorizer.fit_transform(train_data['v2'])  # Assuming 'v2' is the message column
Classifier.fit(vectorize_text, train_data['v1'])  # Assuming 'v1' is the label column ('ham' or 'spam')

@app.route('/', methods=['GET', 'POST'])
def index():
    error = ''
    predict_proba = ''
    predict = ''
    message = ''

    global Classifier
    global Vectorizer

    try:
        # For POST requests, use form data
        if request.method == 'POST':
            message = request.form.get('message', '')

        # For GET requests, use URL query parameter
        elif request.method == 'GET':
            message = request.args.get('message', '')

        # If there's a message, predict spam or ham
        if len(message) > 0:
            vectorize_message = Vectorizer.transform([message])
            predict = Classifier.predict(vectorize_message)[0]
            predict_proba = Classifier.predict_proba(vectorize_message).tolist()

    except Exception as inst:
        error = f"Error: {type(inst).__name__}, {str(inst)}"

    return jsonify(
        message=message,
        predict=predict,
        predict_proba=predict_proba,
        error=error
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
