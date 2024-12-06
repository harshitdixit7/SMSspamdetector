import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from sklearn.metrics import classification_report
import pickle

# Preprocess the text: clean the message
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Load the dataset (using 'spam.csv' with latin-1 encoding)
data = pd.read_csv('spam.csv', encoding='latin-1')

# Ensure the dataset has the required columns ('v1' for labels and 'v2' for messages)
if 'v1' not in data.columns or 'v2' not in data.columns:
    raise ValueError("Dataset must have 'v1' as labels and 'v2' as text columns.")

# Split data into training (4400 items) and testing (1172 items)
train_data = data[:4400]
test_data = data[4400:]

# Apply preprocessing to the 'v2' column (messages)
train_data['v2'] = train_data['v2'].apply(preprocess_text)
test_data['v2'] = test_data['v2'].apply(preprocess_text)

# Handle class imbalance by oversampling the minority class (spam)
# Separate majority and minority classes
ham = train_data[train_data['v1'] == 'ham']
spam = train_data[train_data['v1'] == 'spam']

# Oversample the minority class (spam)
spam_upsampled = resample(spam, 
                          replace=True,     # Sample with replacement
                          n_samples=len(ham),  # Match number of samples in ham
                          random_state=42)   # For reproducibility

# Combine the upsampled minority class with the majority class
train_data_balanced = pd.concat([ham, spam_upsampled])

# Optionally, print the class distribution after balancing
print("Class distribution after balancing:")
print(train_data_balanced['v1'].value_counts())

# Initialize the vectorizer and classifier with adjusted parameters
vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2))  # Adjust ngram_range as needed
classifier = OneVsRestClassifier(SVC(kernel='linear'))

# Train the model on the balanced data
vectorize_text = vectorizer.fit_transform(train_data_balanced['v2'])  # Vectorize training text
classifier.fit(vectorize_text, train_data_balanced['v1'])  # Train on the vectorized text

# Evaluate the model using the test data
vectorize_text_test = vectorizer.transform(test_data['v2'])  # Vectorize test text
predictions = classifier.predict(vectorize_text_test)  # Get predictions

# Print classification report (precision, recall, F1-score, accuracy)
print("Model evaluation on test data:")
print(classification_report(test_data['v1'], predictions))

# Calculate and print the accuracy
score = classifier.score(vectorize_text_test, test_data['v1'])  # Get accuracy score
print(f"Model accuracy: {score * 100:.2f}%")

# Save the trained model and vectorizer to disk using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully!")
