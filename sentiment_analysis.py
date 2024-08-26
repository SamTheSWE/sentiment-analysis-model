

### 2. `sentiment_analysis.py`

```python
# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a sample dataset of movie reviews and their corresponding sentiments.
# In a real-world scenario, you would replace this with a larger, more comprehensive dataset.
data = {
    'review': [
        "I loved the movie, it was fantastic!",
        "The film was boring and too long.",
        "What an amazing experience, truly enjoyed it.",
        "I didn't like the movie at all.",
        "It was okay, but not great.",
        "Terrible movie, I wasted my time.",
        "The best movie I've seen in a long time.",
        "Not my type of film, I didn't enjoy it.",
        "Absolutely brilliant, a must-watch!",
        "Awful, just awful."
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 
        'negative', 'neutral', 'negative', 
        'positive', 'negative', 'positive', 'negative'
    ]
}

# Convert the dataset into a Pandas DataFrame.
df = pd.DataFrame(data)

# Step 2: Split the data into training and testing sets.
# 80% of the data will be used for training, and 20% will be used for testing.
X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Convert the text data into numerical data using CountVectorizer.
# This creates a bag-of-words model where each word is assigned a unique integer ID.
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train a Naive Bayes classifier on the training data.
# Naive Bayes is a simple but effective algorithm for text classification tasks.
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Evaluate the model's performance on the test set.
# We'll check the accuracy and generate a detailed classification report.
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 6: Test the model with new movie reviews.
# You can modify the 'new_reviews' list to test the model with your own inputs.
new_reviews = ["The movie was incredible!", "I didn't enjoy the film.", "It was just okay."]
new_reviews_vec = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_vec)

# Print the predictions for the new reviews.
for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: '{review}' -> Sentiment: {sentiment}")
