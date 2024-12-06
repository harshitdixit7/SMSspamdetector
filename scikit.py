from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load dataset
data = pandas.read_csv('spam.csv', encoding='latin-1')

# Validate dataset columns
if 'v1' not in data.columns or 'v2' not in data.columns:
    raise ValueError("Dataset must have 'v1' as labels and 'v2' as text columns.")

# Drop rows with missing labels or messages
data = data.dropna(subset=['v1', 'v2'])

if data.empty:
    raise ValueError("Dataset is empty or all rows have missing values.")

# Split data into training and testing sets
train_data = data[:4400]  # 4400 items for training
test_data = data[4400:]   # 1172 items for testing

# Define classifier and vectorizer
classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

# Train model
vectorize_text = vectorizer.fit_transform(train_data.v2)
classifier.fit(vectorize_text, train_data.v1)

# Evaluate model
vectorize_text = vectorizer.transform(test_data.v2)
score = classifier.score(vectorize_text, test_data.v1)

# Log the result
logging.info(f"Model trained with accuracy: {score}")

# Save model and vectorizer
with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

logging.info("Model and vectorizer saved successfully!")
