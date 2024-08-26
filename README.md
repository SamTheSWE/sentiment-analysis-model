# Sentiment Analysis Model

This project is a simple sentiment analysis tool that classifies movie reviews as positive, negative, or neutral. The model is built using Python and the `scikit-learn` library, and it utilizes the Naive Bayes algorithm for classification.

## Features

- **Data Preprocessing:** Converts text data into numerical features using `CountVectorizer`.
- **Model Training:** Trains a Naive Bayes classifier on a sample dataset of movie reviews.
- **Evaluation:** Provides accuracy and classification report metrics.
- **Prediction:** Classifies new movie reviews as positive, negative, or neutral.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SamTheSWE/sentiment-analysis-model.git
    cd sentiment-analysis-model
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the script:**
    ```bash
    python sentiment_analysis.py
    ```

## Usage

- The script is self-contained and easy to run.
- Modify the `new_reviews` list at the bottom of the script to classify your own reviews.

## Example Output

```bash
Accuracy: 0.75

Classification Report:
              precision    recall  f1-score   support

    negative       1.00      0.50      0.67         2
     neutral       0.00      0.00      0.00         0
    positive       0.67      1.00      0.80         2

    accuracy                           0.75         4
   macro avg       0.56      0.50      0.49         4
weighted avg       0.83      0.75      0.73         4

Review: 'The movie was incredible!' -> Sentiment: positive
Review: 'I didn't enjoy the film.' -> Sentiment: negative
Review: 'It was just okay.' -> Sentiment: neutral
