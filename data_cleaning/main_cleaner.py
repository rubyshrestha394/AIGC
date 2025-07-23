import nltk
import json
import os
import gdown

from loader import load_jsonl
from text_cleaner import clean_text, tokenize_and_clean
from features import *

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Google Drive File Download
INPUT_FILE = "en_devtest_all.jsonl"
FILE_ID = "15S14Y-tbrTa0wWeaSiwn-YYbrxjR7MTJ"
DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(INPUT_FILE):
    print(f"Downloading file from Google Drive...")
    gdown.download(DRIVE_URL, INPUT_FILE, quiet=False)

OUTPUT_FILE = "cleaned_aigc_data_selected_features.jsonl"

df = load_jsonl(INPUT_FILE)
print("Initial shape:", df.shape)
