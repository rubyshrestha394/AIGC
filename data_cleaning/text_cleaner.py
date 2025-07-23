import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def decode_unicode(text):
    try:
        return text.encode().decode('unicode_escape')
    except:
        return text

def clean_text(text):
    text = decode_unicode(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_and_clean(text):
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(t) for t in tokens if t not in string.punctuation]
