from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input message from the user
        message = request.form['message']
        
        # Vectorize the input message
        vectorized_message = vectorizer.transform([message])
        
        # Predict the class (ham or spam)
        predict = model.predict(vectorized_message)[0]
        
        # Get the prediction probabilities
        predict_proba = model.predict_proba(vectorized_message).tolist()
        
        return render_template('index.html', message=message, predict=predict, predict_proba=predict_proba)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
