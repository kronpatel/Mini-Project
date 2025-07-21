# MINI PROJECT: MENTAL STRESS DETECTION IN STUDENTS USING MACHINE LEARNING

# 1. Importing required libraries
import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# 2. Reading the CSV file
data = pd.read_csv("stress.csv")

# 3. Downloading NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stopword = set(stopwords.words('english'))
stemmer = nltk.SnowballStemmer("english")

# 4. Text cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = " ".join([word for word in text.split() if word not in stopword])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

# 5. Apply cleaning to the text column
data["text"] = data["text"].apply(clean)

# 6. Word Cloud Visualization
text = " ".join(i for i in data.text)
stopwords_wc = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords_wc, background_color="white").generate(text)
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# 7. Label mapping
data["label"] = data["label"].map({0: "No Stress", 1: "Stress"})
data = data[["text", "label"]]

# 8. Text Vectorization and Data Split
x = np.array(data["text"])
y = np.array(data["label"])
cv = CountVectorizer()
X = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# 9. Model Training
model = BernoulliNB()
model.fit(xtrain, ytrain)

# 10. User Input Prediction
user_input = input("Enter your text: ")
user_clean = clean(user_input)
user_vector = cv.transform([user_clean])
output = model.predict(user_vector)
print("Prediction:", output[0])