import csv
import re
from tqdm import tqdm
from config import DATA_ROOT

INPUT_CSV = f"{DATA_ROOT}/madverse_preprocessed_data.csv"
OUTPUT_CSV = f"{DATA_ROOT}/madverse_final_clean.csv"


def clean_ocr_text(text):
    if not text:
        return ""

    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove ad codes & numbers
    text = re.sub(r"\b[a-z]{0,3}\d{2,}\b", "", text)

    # remove special chars
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove very short tokens
    words = [w for w in text.split() if len(w) > 2]

    return " ".join(words)


def extract_slogan(clean_text):
    if not clean_text or len(clean_text.split()) < 3:
        return None

    phrases = re.split(r"[.,]", clean_text)

    candidates = [p.strip() for p in phrases if 3 <= len(p.split()) <= 15]

    if not candidates:
        return None

    return max(candidates, key=len)


with open(INPUT_CSV, "r", encoding="utf-8") as infile, open(
    OUTPUT_CSV, "w", newline="", encoding="utf-8"
) as outfile:

    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    writer.writerow(["image_path", "ocr_text_clean", "slogan", "label"])

    for row in tqdm(reader):
        raw_text = row["slogan_text"]
        clean_text = clean_ocr_text(raw_text)
        slogan = extract_slogan(clean_text)

        writer.writerow([row["image_path"], clean_text, slogan, row["label"]])

print("OCR post-processing complete")
