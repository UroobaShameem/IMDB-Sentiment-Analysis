# IMDB Movie Reviews Sentiment Analysis

This project is a sentiment analysis of IMDB movie reviews using various text preprocessing techniques and machine learning models. The primary goal is to classify movie reviews as either positive or negative by applying logistic regression on two different feature extraction methods: Bag of Words (BOW) and TF-IDF.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Text Preprocessing](#text-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)

## Introduction
Sentiment analysis is a common text classification task where the goal is to predict the sentiment (positive or negative) of textual data. In this project, we perform sentiment analysis on the IMDB movie reviews dataset, using logistic regression for classification. The reviews are processed using two different methods: Bag of Words (BOW) and TF-IDF.

## Dataset
The dataset used in this project is the [IMDB Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The dataset contains two columns:
- `review`: The text of the movie review.
- `sentiment`: The sentiment of the review, either "positive" or "negative".

## Text Preprocessing
Before feeding the data into machine learning models, several preprocessing steps are performed:
1. **Noise Removal**: HTML tags and unnecessary characters are removed using the BeautifulSoup library.
2. **Tokenization**: Splitting the text into tokens.
3. **Stopwords Removal**: Common words (like "the", "is", etc.) are removed to reduce noise.
4. **Stemming**: Reducing words to their root form using the Porter Stemmer.

## Feature Extraction
Two methods of feature extraction are used to convert the text data into a numerical form suitable for machine learning:
1. **Bag of Words (BOW)**: Converts the text into a matrix of token counts, considering unigrams, bigrams, and trigrams.
2. **TF-IDF**: A matrix that reflects both the term frequency and inverse document frequency, downweighting common words.

## Modeling
A logistic regression model is used for binary classification. The model is trained using both BOW and TF-IDF features.

### Models:
- **Bag of Words (BOW)**: Logistic regression is applied to the BOW features.
- **TF-IDF**: Logistic regression is also applied to the TF-IDF features.

## Evaluation
The models are evaluated based on accuracy scores. Both the BOW-based and TF-IDF-based models are compared to determine which method performs better.

## Dependencies
To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `nltk`
- `seaborn`
- `matplotlib`
- `sklearn`
- `wordcloud`
- `beautifulsoup4`
- `textblob`
- `spacy`

You can install the required packages using:
```bash
pip install numpy pandas nltk seaborn matplotlib scikit-learn wordcloud beautifulsoup4 textblob spacy
```

## How to Run
1. Download the dataset from Kaggle and place it in the `dataset` folder.
2. Run the Python script for sentiment analysis using the command:
   ```bash
   python sentiment_analysis.py
   ```
3. The script will preprocess the data, extract features, train the model, and display the accuracy of the model.
