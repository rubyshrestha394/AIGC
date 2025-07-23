from nltk.tokenize import sent_tokenize
import string
import textstat

def compute_length(text):
    return len(text.split())

def compute_sentence_stats(text):
    sentences = sent_tokenize(text)
    count = len(sentences)
    avg = len(text.split()) / count if count > 0 else 0
    return count, avg

def readability_score(text):
    return textstat.flesch_reading_ease(text)

def punctuation_count(text):
    return sum(1 for char in text if char in string.punctuation)

def type_token_ratio(tokens):
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)
