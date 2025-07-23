import nltk
from nltk.corpus import stopwords

# Try to load stopwords silently
try:
    stopword = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stopword = set(stopwords.words('english'))

# Create and expose stemmer object
stemmer = nltk.SnowballStemmer("english")
