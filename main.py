# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
df = pd.read_csv("Twitter_Data.csv")
df.head()

df['category'].value_counts()

df['category'].value_counts().plot(kind='bar')

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
colors = ['green', 'orange', 'blue']
df['category'].value_counts().plot(kind='pie', autopct='%.1f%%',shadow = True,colors = colors)
plt.title('label distribution')

# Preprocessing
lemmatizer = WordNetLemmatizer()


import numpy as np

def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]
        # Lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    else:
        return ''  # Replace NaN values with an empty string or any other placeholder

# Apply preprocessing to clean_text column
df['clean_text'] = df['clean_text'].apply(preprocess_text)


df['clean_text']

# Drop rows with NaN values from the DataFrame
df.dropna(subset=['clean_text', 'category'], inplace=True)

# Split the data into features (X) and target variable (y)
X = df['clean_text']
y = df['category']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data 
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the model
model = SVC(kernel='linear')
model.fit(X_train_vectors, y_train)

# Evaluate the model
predictions = model.predict(X_test_vectors)
print(classification_report(y_test, predictions))


import joblib

# Save the trained model
joblib.dump(model, 'sentiment_analysis_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

from wordcloud import WordCloud

# Concatenate all the comments into a single string
all_comments = ' '.join(df['clean_text'])

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_comments)

# Plot the word cloud
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud')
plt.axis('off')
plt.show()

from nltk.probability import FreqDist
import string

# Concatenate all the comments into a single string
all_comments = ' '.join(df['clean_text'])

# Tokenize the string into words
tokens = word_tokenize(all_comments)

# Convert tokens to lowercase
tokens = [word.lower() for word in tokens]

# Remove stopwords and punctuation
stop_words = set(stopwords.words('english') + list(string.punctuation))
filtered_tokens = [word for word in tokens if word not in stop_words]

# Calculate frequency distribution of words
fdist = FreqDist(filtered_tokens)

# Get the most common words and their frequencies
most_common_words = fdist.most_common(30)
words, frequencies = zip(*most_common_words)

# Plot the most common words as a bar graph
plt.figure(figsize=(12, 6))
plt.bar(words, frequencies, color='skyblue')
plt.title('Most Common Words')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


