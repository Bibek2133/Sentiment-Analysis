# 🎯 Twitter Sentiment Analysis
<br>
Overview
<br>
Analyze Twitter data to classify tweets into Positive, Negative, or Neutral sentiments. This project leverages machine learning techniques to preprocess text data, train a model, and make real-time predictions via a Flask web application.
<br>
🚀 Features
<br>
Data Preprocessing: Clean and lemmatize tweets for effective analysis.
Sentiment Classification: Predict tweet sentiment using a trained SVM model.
Real-Time Analysis: Input text in a user-friendly web interface and get instant sentiment feedback.
Visualization: Explore sentiment distribution and word clouds of common words in the dataset.
<br>
🗂 Project Structure
<br>
twitter-sentiment-analysis/
│
├── data/
│   └── Twitter_Data.csv           # Dataset for model training
│
├── notebooks/
│   ├── data_preprocessing.ipynb   # Data cleaning and preprocessing steps
│   ├── model_training.ipynb       # Model training and evaluation
│   └── evaluation.ipynb           # Performance evaluation and metrics
│
├── src/
│   ├── app.py                     # Flask app for real-time sentiment analysis
│   ├── main.py                    # Script for training the sentiment analysis model
│   └── templates/
│       └── index.html             # Web interface for user interaction
│
├── sentiment_analysis_model.pkl   # Trained sentiment analysis model
├── tfidf_vectorizer.pkl           # Trained TF-IDF vectorizer
├── README.md                      # Project documentation
└── requirements.txt               # Required Python packages

