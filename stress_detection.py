# MINI PROJECT: MENTAL STRESS DETECTION IN STUDENTS USING MACHINE LEARNING

# 1. Importing required libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# ✅ Custom logic import (no pop-up)
from Logics.skipper import stopword, stemmer

# 2. Reading the CSV file
data = pd.read_csv("stress.csv")

# 3. Text cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = " ".join([word for word in text.split() if word not in stopword])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

# 4. Apply cleaning to the text column
data["text"] = data["text"].apply(clean)

# 6. Label mapping
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]

# 7. Text Vectorization and Data Split
x = np.array(data["text"])
y = np.array(data["label"])
cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# 8. Model Training
model = BernoulliNB()
model.fit(xtrain, ytrain)

# 9. User Input Prediction
user_input = input("Enter your text: ")
user_clean = clean(user_input)
user_vector = cv.transform([user_clean])
output = model.predict(user_vector)
print("Prediction:", output[0])
