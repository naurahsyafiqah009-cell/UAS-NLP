import pandas as pd
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("dataset.csv")

data['text'] = data['text'].str.lower()
data['tokens'] = data['text'].apply(word_tokenize)

stop_words = set(stopwords.words('indonesian'))
data['tokens'] = data['tokens'].apply(lambda x: [w for w in x if w not in stop_words])

factory = StemmerFactory()
stemmer = factory.create_stemmer()
data['tokens'] = data['tokens'].apply(lambda x: [stemmer.stem(w) for w in x])

data['clean_text'] = data['tokens'].apply(lambda x: " ".join(x))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
