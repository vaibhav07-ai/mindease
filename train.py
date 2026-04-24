import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

lemmatizer = WordNetLemmatizer()

# Load your training data
with open('intents.json') as f:
    data = json.load(f)

# Separate sentences and labels
sentences = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern.lower())
        labels.append(intent['tag'])

# Clean each sentence (lemmatize = reduce words to root form)
# Example: "worrying" becomes "worry", "feeling" becomes "feel"
def clean(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

clean_sentences = [clean(s) for s in sentences]

# Convert labels to numbers
encoder = LabelEncoder()
label_numbers = encoder.fit_transform(labels)

# Build the ML model
# TF-IDF converts text to numbers, Naive Bayes classifies them
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Train it!
model.fit(clean_sentences, label_numbers)

# Save the model and encoder to files
joblib.dump(model, 'model.pkl')
joblib.dump(encoder, 'encoder.pkl')

print("Training done! model.pkl and encoder.pkl created.")