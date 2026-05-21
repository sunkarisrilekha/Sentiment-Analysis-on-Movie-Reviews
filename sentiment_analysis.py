import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample Movie Review Dataset
data = {
    'Review': [
        'Amazing movie with great acting',
        'Worst movie I have ever seen',
        'Excellent storyline and visuals',
        'Terrible experience and boring plot',
        'Fantastic performance by actors',
        'Not worth watching',
        'Loved the movie so much',
        'Waste of time'
    ],
    'Sentiment': [
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative',
        'Positive',
        'Negative'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Labels
X = df['Review']
y = df['Sentiment']

# Convert text into numerical format
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test Custom Review
review = ["The movie was absolutely fantastic"]
review_vector = vectorizer.transform(review)

prediction = model.predict(review_vector)

print("\nReview Prediction:", prediction[0])
