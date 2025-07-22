from data_loader import download_jsonl_from_drive, load_jsonl
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Google Drive File ID and Path
FILE_ID = "1QfOx_Wgm6enL_0yXo9mntOU4Aoc92VVY"
DATA_PATH = "cleaned_aigc_data_selected_features.jsonl"

# Step 1: Download if not already present
download_jsonl_from_drive(FILE_ID, DATA_PATH)

# Step 2: Load data
df = load_jsonl(DATA_PATH)

# Step 3: Preprocessing
print("Data Shape:", df.shape)
print("Label Distribution:\n", df['label'].value_counts())

vectorizer = TfidfVectorizer(max_features=5005, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df['cleaned_text'])
X_extra = df[['text_length', 'avg_words_per_sentence', 'readability', 'punctuation_count', 'type_token_ratio']].values
X = hstack([X_tfidf, X_extra])
y = df['label'].values

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Train SVM example (rest of models can follow similar)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model

svm = SVC(probability=True, random_state=42)
param_grid = {'C': [1], 'kernel': ['linear'], 'class_weight': ['balanced']}
grid = GridSearchCV(svm, param_grid, scoring='f1', cv=5)
grid.fit(X_train, y_train)

evaluate_model(grid.best_estimator_, "SVM", X_val, y_val, X_test, y_test)

# Save
joblib.dump(grid.best_estimator_, 'svm_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
