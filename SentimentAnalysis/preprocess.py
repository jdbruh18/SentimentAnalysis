import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords (punkt not needed anymore)
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Optional: Reduce size to avoid memory issues
df = df.sample(10000, random_state=42)

# Optional: Binary sentiment
df['sentiment'] = df['sentiment'].replace(4, 1)

# Use regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# Clean function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Show cleaned sample
print(df[['text', 'cleaned_text']].head())

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text']).toarray()
y = df['sentiment']

print("TF-IDF Matrix Shape:", X.shape)
print("Labels Shape:", y.shape)
