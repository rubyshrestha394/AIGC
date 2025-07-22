import gdown
import os
import json
import pandas as pd

def download_jsonl_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading dataset from Google Drive to {output_path}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"File already exists at {output_path}")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")
    return pd.DataFrame(data)
