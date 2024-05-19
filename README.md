# Restaurant Reviews Sentiment Analysis

This project focuses on building a sentiment analysis model to classify restaurant reviews as positive or negative. The model uses Natural Language Processing (NLP) techniques and machine learning algorithms to analyze and predict the sentiment of the reviews.


## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Try it Yourself](#try-it-yourself)
- [License](#license)

## Project Overview
The goal of this project is to develop a classification model that can predict the sentiment of restaurant reviews. The model processes the reviews using NLP techniques, such as tokenization and stemming, and then applies a machine learning algorithms for classification.

## Dataset
The dataset used in this project consists of restaurant reviews labeled as positive (1) or negative(0). The reviews are preprocessed to remove noise and convert text into a format suitable for machine learning models.

## Requirements
- Python 3.x
- Jupyter Notebook or JupyterLab

## Usage
1. Clone or download all contents of this repository.
   
2. Open the Jupyter Notebook or JupyterLab Notebook.

3. Run the notebook cells to preprocess the data, train the model, and evaluate its performance.

4. Try it Yourself! You can test the model with your own review in the "Try me!" chapter.
   
## Model Evaluation
The model's performance is evaluated using a confusion matrix, accuracy, precision, and recall metrics. The confusion matrix provides insights into the number of correct and incorrect predictions, while precision and recall help understand the model's ability to classify positive and negative reviews correctly.

### Results
- **Accuracy:** 78%
- **Confusion Matrix:**
  - 89 correct predictions of negative reviews
  - 67 correct predictions of positive reviews
  - 8 incorrect predictions of negative reviews
  - 36 incorrect predictions of positive reviews
- **Type I Error:** 8.25%
- **Precision:** 89.33%
- Model is **pretty good** in general, but **prone to misclassifying positive reviews as negative.**


### License
This project is licensed under the **MIT License**. See the LICENSE file for more details.
