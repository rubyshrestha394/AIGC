import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from string import punctuation

# Download required NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# === UTILITY FUNCTIONS ===

def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return pd.DataFrame([json.loads(line) for line in f])

def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)      # remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # normalize whitespace
    return text

def tokenize_and_clean(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def compute_length(text):
    return len(text.split())

def compute_sentence_stats(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return 0, 0
    sentence_count = len(sentences)
    avg_len = sum(len(s.split()) for s in sentences) / sentence_count
    return sentence_count, avg_len

def readability_score(text):
    words = text.split()
    sentences = sent_tokenize(text)
    syllable_count = sum(sum(1 for char in word if char in "aeiou") for word in words)
    if len(words) == 0 or len(sentences) == 0:
        return 0
    score = 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllable_count / len(words))
    return round(score, 2)

def punctuation_count(text):
    return sum(1 for char in text if char in punctuation)

def type_token_ratio(tokens):
    if not tokens:
        return 0
    return round(len(set(tokens)) / len(tokens), 2)

# === MAIN SCRIPT ===

INPUT_FILE = "train_dataset.jsonl"  
OUTPUT_FILE = "train_dataset_with_features.jsonl"
TEXT_FIELD = "text"

df = load_jsonl(INPUT_FILE)
print(f"Loaded {len(df)} records.")

# df = df.head(100)

output_data = []
for _, row in df.iterrows():
    raw_text = row["text"]
    cleaned = clean_text(raw_text)
    tokens = tokenize_and_clean(cleaned)

    sentence_count, avg_sentence_len = compute_sentence_stats(cleaned)
    entry = {
        "id": row["id"],
        "source": row["source"],
        "sub_source": row.get("sub_source", None),
        "lang": row.get("lang", None),
        "model": row["model"],
        "label": row["label"],
        "text": raw_text,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "text_length": compute_length(cleaned),
        "sentence_count": sentence_count,
        "avg_words_per_sentence": avg_sentence_len,
        "readability": readability_score(cleaned),
        "punctuation_count": punctuation_count(raw_text),
        "type_token_ratio": type_token_ratio(tokens)
    }
    output_data.append(entry)

save_jsonl(output_data, OUTPUT_FILE)
print(f"Saved {len(output_data)} cleaned records to {OUTPUT_FILE}")
