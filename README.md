# SMS Spam Detection

This project detects whether an SMS message is **spam** or **ham** (non-spam) using a machine learning model. The model is trained using a **Support Vector Classifier (SVC)** in a **One-vs-Rest** setup. The user can input a message, and the system will predict whether it's spam or not.

## Project Setup

### 1. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/harshitdixit7/SMSspamdetector.git

Install Python and Virtual Environment
python --version

Create a virtual environment
python -m venv sklearn-env

Activate the virtual environment:
sklearn-env\Scripts\activate

Install Dependencies
pip install -r requirements.txt

Train the Model
python scikit.py

Run the Flask Application
python app.py



scikit.py: Contains code to train the machine learning model using the spam.csv dataset.
classifier.py: Defines the classifier and handles model training and saving.
app.py: Flask application to serve the trained model and allow users to input SMS messages for prediction.
index.html: The HTML form for user input and displaying the results.
requirements.txt: Lists the necessary Python libraries.
spam.csv: The dataset containing SMS messages labeled as spam or ham.



Here’s a simple README.md file that provides a clear guide on how to set up and run your SMS Spam Detection project.

README.md
markdown
Copy code
# SMS Spam Detection

This project detects whether an SMS message is **spam** or **ham** (non-spam) using a machine learning model. The model is trained using a **Support Vector Classifier (SVC)** in a **One-vs-Rest** setup. The user can input a message, and the system will predict whether it's spam or not.

## Project Setup

### 1. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/harshitdixit7/SMSspamdetector.git
2. Install Python and Virtual Environment
Ensure that you have Python 3.x installed. You can check this by running:

bash
Copy code
python --version
Create a virtual environment to keep your dependencies isolated:

bash
Copy code
python -m venv sklearn-env
Activate the virtual environment:

For Windows:
bash
Copy code
sklearn-env\Scripts\activate
For macOS/Linux:
bash
Copy code
source sklearn-env/bin/activate
3. Install Dependencies
Install the required Python packages by running:

bash
Copy code
pip install -r requirements.txt
This will install all necessary dependencies, including Flask, scikit-learn, pandas, and others.

4. Train the Model (if not already done)
If the model (model.pkl) and vectorizer.pkl files don't already exist, you will need to train the model first.

To do this, run:

bash
Copy code
python scikit.py
This script will:

Load and preprocess the spam.csv dataset.
Train the model.
Save the trained model (model.pkl) and vectorizer (vectorizer.pkl) for future use.
5. Run the Flask Application
Start the Flask web application:

bash
Copy code
python app.py
This will start the Flask server, and you should see output like:

csharp
Copy code
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
6. Access the Web Application
Open a web browser and go to http://127.0.0.1:5000/. You should see a form where you can input a message. The application will predict whether the message is spam or ham.

Project Structure
scikit.py: Contains code to train the machine learning model using the spam.csv dataset.
classifier.py: Defines the classifier and handles model training and saving.
app.py: Flask application to serve the trained model and allow users to input SMS messages for prediction.
index.html: The HTML form for user input and displaying the results.
requirements.txt: Lists the necessary Python libraries.
spam.csv: The dataset containing SMS messages labeled as spam or ham.
Model Details
Classifier: OneVsRestClassifier with a linear SVC (Support Vector Classifier).
Vectorizer: TfidfVectorizer to convert SMS messages into numerical features.
Accuracy: After training, the model achieves an accuracy of around 98% on the test data.