import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import pandas as pd
from collections import Counter

from config import DATA_ROOT


# LOAD CSV
CSV_PATH = f"{DATA_ROOT}/madverse_preprocessed_data.csv"
OUTPUT_CSV_PATH = f"{DATA_ROOT}/ocr_ads_cleaned.csv"

TEXT_COL = "slogan_text"
LABEL_COL = "label"

df = pd.read_csv(CSV_PATH)
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)


# CHARACTER-LEVEL CLEANING
def is_valid_word(word):
    if len(word) < 3:
        return False

    if re.search(r"\d", word):  # remove digits in words
        return False

    vowel_ratio = sum(c in "aeiou" for c in word.lower()) / len(word)
    return vowel_ratio > 0.25


def clean_ocr_text(text):
    words = re.findall(r"[a-zA-Z]+", text)
    words = [w.lower() for w in words if is_valid_word(w)]
    return " ".join(words)


df["clean_text"] = df[TEXT_COL].apply(clean_ocr_text)

# CORPUS-LEVEL FREQUENCY FILTERING
all_words = []

for text in df["clean_text"]:
    all_words.extend(text.split())

# Compute global word frequencies
freq = Counter(all_words)

# keep words appearing at least N times in dataset
MIN_FREQ = 5
COMMON_WORDS = {w for w, c in freq.items() if c >= MIN_FREQ}


def filter_common_words(text):
    return " ".join(w for w in text.split() if w in COMMON_WORDS)


df["filtered_text"] = df["clean_text"].apply(filter_common_words)


df_out = df.copy()
df_out.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Saved cleaned CSV -> {OUTPUT_CSV_PATH}")
