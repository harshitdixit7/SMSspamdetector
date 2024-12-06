from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas as pd
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def perform_and_save(classifiers, vectorizers, train_data, test_data):
    best_score = 0
    best_classifier = None
    best_vectorizer = None
    
    for classifier in classifiers:
        for vectorizer in vectorizers:
            try:
                # Train
                vectorize_text = vectorizer.fit_transform(train_data.v2)
                classifier.fit(vectorize_text, train_data.v1)

                # Test and evaluate
                vectorize_text_test = vectorizer.transform(test_data.v2)
                score = classifier.score(vectorize_text_test, test_data.v1)

                logging.info(f"{classifier.__class__.__name__} with {vectorizer.__class__.__name__}. Has score: {score}")
                
                # Save the best-performing model
                if score > best_score:
                    best_score = score
                    best_classifier = classifier
                    best_vectorizer = vectorizer
            except Exception as e:
                logging.error(f"Error with {classifier.__class__.__name__} and {vectorizer.__class__.__name__}: {e}")

    # Save the best classifier and vectorizer only if valid
    if best_classifier and best_vectorizer:
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(best_classifier, model_file)
        with open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(best_vectorizer, vectorizer_file)
        logging.info(f"Best Model: {best_classifier.__class__.__name__} with {best_vectorizer.__class__.__name__}, Score: {best_score}")
        logging.info("Best model and vectorizer saved successfully!")
    else:
        logging.warning("No valid model and vectorizer were found.")

# Load dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Validate dataset columns
if 'v1' not in data.columns or 'v2' not in data.columns:
    raise ValueError("Dataset must have 'v1' as labels and 'v2' as text columns.")

# Drop rows with missing values
data = data.dropna(subset=['v1', 'v2'])

if data.empty:
    raise ValueError("Dataset is empty or all rows have missing values.")

learn = data[:4400]  # 4400 items for training
test = data[4400:]   # 1172 items for testing

perform_and_save(
    [
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        ExtraTreesClassifier(),
        OneVsRestClassifier(SVC(kernel='linear', probability=True)),
        KNeighborsClassifier()
    ],
    [
        TfidfVectorizer(),
        CountVectorizer()
    ],
    learn,
    test
)
