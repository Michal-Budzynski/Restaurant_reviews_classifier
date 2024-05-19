# Restaurant Reviews Sentiment Analysis

This project focuses on building a sentiment analysis model to classify restaurant reviews as positive or negative. The model uses Natural Language Processing (NLP) techniques and machine learning algorithms to analyze and predict the sentiment of the reviews.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Try it Yourself](#try-it-yourself)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to develop a classification model that can predict the sentiment of restaurant reviews. The model processes the reviews using NLP techniques, such as tokenization and stemming, and then applies a machine learning algorithm for classification.

## Dataset
The dataset used in this project consists of restaurant reviews labeled as positive or negative. The reviews are preprocessed to remove noise and convert text into a format suitable for machine learning models.

## Requirements
- Python 3.x
- Jupyter Notebook
- NumPy
- pandas
- matplotlib
- scikit-learn
- NLTK

## Installation
1. Clone this repository:
    ```sh
    git clone https://github.com/yourusername/restaurant-reviews-sentiment-analysis.git
    cd restaurant-reviews-sentiment-analysis
    ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
1. Open the Jupyter Notebook:
    ```sh
    jupyter notebook Restaurant_reviews_model.ipynb
    ```
2. Run the notebook cells to preprocess the data, train the model, and evaluate its performance.

## Model Evaluation
The model's performance is evaluated using a confusion matrix, accuracy, precision, and recall metrics. The confusion matrix provides insights into the number of correct and incorrect predictions, while precision and recall help understand the model's ability to classify positive and negative reviews correctly.

### Results
- **Confusion Matrix:**
  - 167 correct predictions of negative reviews
  - 184 correct predictions of positive reviews
  - 13 incorrect predictions of negative reviews
  - 36 incorrect predictions of positive reviews
- **Type I Error:** 8.25%
- **Precision:** 89.33%

## Try it Yourself
You can test the model with your own review by modifying the `new_review` string variable in the following code cell and running it:

```python
new_review = 'I am a big fan of this restaurant' # your own review
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier_ksvm.predict(new_X_test)
if new_y_pred == 1:
    print('Positive review')
else:
    print('Negative review')
