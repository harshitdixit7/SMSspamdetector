from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import csv

# Load dataset
data = pandas.read_csv('spam.csv', encoding='latin-1')

# Split data into train and test sets
train_data = data[:4400]  # 4400 items for training
test_data = data[4400:]   # 1172 items for testing

# Initialize classifier and vectorizer
classifier = OneVsRestClassifier(SVC(kernel='linear'))
vectorizer = TfidfVectorizer()

# Train the model
vectorize_text = vectorizer.fit_transform(train_data['v2'])  # 'v2' is the text column
classifier.fit(vectorize_text, train_data['v1'])  # 'v1' is the label column

# Prepare list to store results
csv_arr = []

# Loop through test data to get predictions and compare with actual values
for index, row in test_data.iterrows():
    answer = row['v1']  # True label
    text = row['v2']    # Message text
    vectorized_text = vectorizer.transform([text])
    predict = classifier.predict(vectorized_text)[0]  # Model prediction

    result = 'right' if predict == answer else 'wrong'

    # Append to the results array
    csv_arr.append([len(csv_arr), text, answer, predict, result])

# Write results to CSV with utf-8 encoding
with open('test_score.csv', 'w', newline='', encoding='utf-8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write header
    spamwriter.writerow(['#', 'text', 'answer', 'predict', 'result'])

    # Write each row of results
    for row in csv_arr:
        spamwriter.writerow(row)

print("Test results saved to 'test_score.csv'.")
