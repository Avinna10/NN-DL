import os
import json
import csv
import easyocr
from tqdm import tqdm
import time
from PIL import Image
import numpy as np

DATA_ROOT = "madverse_data"
ANNOT_FILES = [
    "madverse_data/annotations/web_annot_j.json",
    "madverse_data/annotations/adgal_annot_j.json",
]
OUTPUT_CSV = "data/madverse_preprocessed_data.csv"
FLUSH_EVERY = 20

# Low VRAM optimizations
MAX_IMAGE_SIZE = 800  # Reduce resolution before OCR

# Track images already processed to support resumable runs
processed = set()

if not os.path.exists(OUTPUT_CSV):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
else:
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader_csv = csv.reader(f)
        next(reader_csv, None)
        for row in reader_csv:
            processed.add(row[0])

print(f"Already processed: {len(processed)} images")

# Collect work items from annotations (normalized paths, skip missing/processed)
items_to_process = []
for annot_file in ANNOT_FILES:
    with open(annot_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            img_path = item.get("img_path") or item.get("image_path")
            if not img_path:
                continue

            img_path = img_path.replace("\\", "/").replace("../", "")
            full_path = os.path.normpath(os.path.join(DATA_ROOT, img_path))

            if not os.path.exists(full_path) or full_path in processed:
                continue

            hier = item.get("hier_annot", [])
            if hier:
                items_to_process.append({"path": full_path, "label": hier[0]})

print(f"Items left to process: {len(items_to_process)}")

# Initialize EasyOCR in a memory-efficient configuration
reader = easyocr.Reader(
    ["en"],
    gpu=True,
    quantize=True,  # Use quantized model (faster, less memory)
    verbose=False,
)

file_exists = os.path.exists(OUTPUT_CSV)
csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
writer = csv.writer(csv_file)

# Write header only when creating a new CSV
if not file_exists:
    writer.writerow(["image_path", "slogan_text", "label"])

start_time = time.time()
print("Starting OCR...")

for idx, item in enumerate(tqdm(items_to_process[:1]), 1):
    try:
        # Load and resize image
        img = Image.open(item["path"])

        # Downscale if needed to reduce VRAM usage
        if max(img.size) > MAX_IMAGE_SIZE:
            img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.LANCZOS)

        # Convert to numpy array
        img_array = np.array(img)

        # Run OCR
        results = reader.readtext(img_array, detail=0)
        text = " ".join(results)

    except Exception as e:
        text = ""

    # Persist per-image result
    writer.writerow([item["path"], text, item["label"]])

    # Flush intermittently so progress isn't lost on long runs
    if idx % FLUSH_EVERY == 0:
        csv_file.flush()

csv_file.close()

elapsed = time.time() - start_time
print(f"Done in {elapsed:.2f}s ({elapsed/len(items_to_process):.2f}s/img)")
