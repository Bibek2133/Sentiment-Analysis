
# 🎯 Twitter Sentiment Analysis

## Overview
Analyze Twitter data to classify tweets into **Positive**, **Negative**, or **Neutral** sentiments. This project leverages machine learning techniques to preprocess text data, train a model, and make real-time predictions via a Flask web application.

## 🚀 Features
- **Data Preprocessing:** Clean and lemmatize tweets for effective analysis.
- **Sentiment Classification:** Predict tweet sentiment using a trained SVM model.
- **Real-Time Analysis:** Input text in a user-friendly web interface and get instant sentiment feedback.
- **Visualization:** Explore sentiment distribution and word clouds of common words in the dataset.

## 🗂 Project Structure
```plaintext
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
```

## 🛠️ Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/twitter-sentiment-analysis.git
    ```
2. Navigate to the project directory:
    ```bash
    cd twitter-sentiment-analysis
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## 🎯 Usage
1. **Train the Model**: 
   - Run `main.py` to preprocess the data, train the SVM model, and save the model and vectorizer.
   
2. **Start the Flask App**:
    ```bash
    python src/app.py
    ```
3. **Interact via Web Interface**:
   - Open your browser and go to `http://127.0.0.1:5000/`.
   - Enter text to analyze sentiment in real-time.

## 📊 Dataset
- **Twitter_Data.csv**: A CSV file containing tweets labeled as positive, negative, or neutral.

## 👨‍💻 Authors
- **Bibek Ranjan Sahoo**

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
