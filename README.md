# Multimodal Advertisement Classification

A deep learning project for classifying advertisements using multimodal fusion of image and text data from the MADVerse dataset.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Technical Outcomes](#technical-outcomes)
- [Pre-trained Models](#pre-trained-models)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)

## Project Overview

This project classifies advertisements into brand categories by combining visual and textual features using state-of-the-art deep learning architectures.

**Key Features:**
- OCR-based text extraction from advertisement images
- Multi-stage data cleaning and preprocessing
- Three model architectures: Image-only (MobileNetV2), Text-only (DistilBERT), and Multimodal Fusion
- Ready-to-use pretrained models in `checkpoints/` directory

## Dataset

**MADVerse Dataset** - A multimodal advertisement dataset with:
- **23,124 advertisement images** from newspapers and online sources
- **3 modalities**: Image paths, extracted slogans, brand labels
- **Data sources**: NewsPaper Ads, Online Ads, Epaper advertisements (multiple languages)
- **Data quality**: 99.42% complete (only 0.58% missing slogans)

**Dataset Source:** [MADVerse on Zenodo](https://zenodo.org/records/10657763)

## Pipeline

### 1. Text Extraction (OCR)
**Script:** `scripts/easyocr_preprocess.py`

Extracts textual content from 23,124 advertisement images using EasyOCR with GPU acceleration.

**Key Features:**
- Memory-optimized OCR with quantized models and image downscaling (max 800px)
- Resumable processing that tracks already-processed images
- Batch processing with progress tracking

**Output:** `data/madverse_preprocessed_data.csv` (columns: `image_path`, `slogan_text`, `label`)

### 2. Data Cleaning
**Script:** `scripts/ocr_clean.py`

Two-stage text cleaning pipeline to remove OCR noise and irrelevant words.

**Cleaning Stages:**
- **Character-level**: Remove short words (<3 chars), digits, low vowel-ratio words (OCR artifacts)
- **Corpus-level**: Retain only words appearing ≥5 times across dataset (frequency filtering)

**Output:** `data/ocr_ads_cleaned.csv` (includes `clean_text`, `filtered_text`, `label_id`)

### 3. Exploratory Data Analysis (EDA)
**Notebook:** `eda.ipynb`

Comprehensive dataset analysis revealing:
- Dataset size: 23,124 samples, 3 columns, 10.67 MB
- Data quality: 99.42% complete (only 133 missing slogans)
- Well-balanced class distribution
- Both image and text modalities available for all samples

### 4. Model Implementation
**Notebook:** `ad.ipynb`

Three deep learning architectures implemented and trained:

**a) Image-Only Model (MobileNetV2)**
- MobileNetV2 backbone pretrained on ImageNet
- Custom 3-layer classifier with BatchNorm and Dropout
- Input: 224×224 RGB images

**b) Text-Only Model (DistilBERT)**
- DistilBERT encoder (768-d features)
- 2-layer MLP classifier
- Max sequence length: 32 tokens

**c) Multimodal Fusion Model**
- Late fusion architecture combining image + text features
- Frozen pretrained encoders (MobileNetV2 + DistilBERT)
- Trainable fusion layers

**Training Features:**
- Mixed precision training (AMP) for efficiency
- Automatic checkpoint saving/resuming
- 80-20 train-validation split (stratified)
- Training/validation metrics tracking with visualization

## Technical Outcomes

### 1. Text Extraction Pipeline
- Successfully extracted text from 23,124 advertisement images using GPU-accelerated OCR
- Memory-efficient processing with quantization and image resizing
- Resumable workflow with checkpoint tracking

### 2. Data Cleaning Framework
- Two-stage cleaning: character-level (vowel-ratio heuristic) + corpus-level (frequency filtering)
- Effectively removes OCR artifacts and vocabulary noise
- Reduces data dimensionality while preserving signal

### 3. Multimodal Model Architecture
- Three models trained: Image-only, Text-only, and Multimodal Fusion
- Image model: MobileNetV2 backbone with custom classifier
- Text model: DistilBERT encoder with MLP head
- Fusion model: Late fusion combining frozen pretrained encoders

### 4. Training Optimizations
- Mixed precision training (AMP) for 2× speedup
- Checkpoint system for resumable training
- Stratified train-test split (80-20) for balanced evaluation

### 5. Reproducibility
- All models saved with complete training state
- Fixed random seeds and documented hyperparameters
- Ready-to-use pretrained checkpoints in `checkpoints/` directory

## Pre-trained Models

The `checkpoints/` directory contains three ready-to-use pretrained models:

### Available Models

| Model File | Architecture | Input | Output |
|-----------|--------------|-------|--------|
| `mobilenetv2_image_model.pth` | MobileNetV2 + MLP | Image (224×224) | Brand class |
| `distilbert_text_model.pth` | DistilBERT + MLP | Text (max 32 tokens) |  (.pth files):

| Model File | Architecture | Input Modality |
|-----------|--------------|----------------|
| `mobilenetv2_image_model.pth` | MobileNetV2 + MLP | Image (224×224) |
| `distilbert_text_model.pth` | DistilBERT + MLP | Text (max 32 tokens) |
| `multimodal_fusion_model.pth` | MobileNetV2 + DistilBERT Fusion | Image + Text |

### Using the Pretrained Models

Each `.pth` checkpoint file contains:
- Model weights (`model_state_dict`)
- Label encoder for class name mapping
- Training history (losses, epochs)
- Model configuration (num_classes, architecture details)
 & Usage

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM, 10GB+ disk space

### Installation

```bash
# Clone repository
git clone <repository-url>
cd NN-DL

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:** PyTorch, torchvision, transformers, easyocr, scikit-learn, pandas

### Usage

#### 1. Data Preprocessing

```bash
# Extract text from images using OCR
python scripts/easyocr_preprocess.py

# Clean extracted text
python scripts/ocr_clean.py

# Optional: Run EDA notebook
jupyter notebook eda.ipynb
```

#### 2. Model Training

```bash
# Open training notebook
jupyter notebook ad.ipynb
```

Run the notebook cells to train:
- Image-only model → saved to `checkpoints/mobilenetv2_image_model.pth`
- Text-only model → saved to `checkpoints/distilbert_text_model.pth`
- Fusion model → saved to `checkpoints/multimodal_fusion_model.pth`

Training automatically saves checkpoints and can be resumed if interrupted.

#### 3. Inference with Pretrained Models

Load any model from `checkpoints/` and use for predictions:

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/model_name.pth")

# Reconstruct model (see ad.ipynb for model classes)
model = ModelClass(num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get label encoder for class names
label_encoder = checkpoint['label_encoder']

# Preprocess input and run inference
# - Images: resize to 224×224, normalize with ImageNet stats
# - Text: tokenize with DistilBERT (max_length=32)
```

Refer to `ad.ipynb` for complete preprocessing and inference examples. return predictions

# Example usage
image_list = ["ad1.jpg", "ad2.jpg", "ad3.jpg"]
slogan_list = ["Slogan 1", "Slogan 2", "Slogan 3"]
results = batch_predict(image_list, slogan_list)

for img, (brand, conf) in zip(image_list, results):
    print(f"{img}: {brand} ({conf:.2%})")
```

## Project Structure

```
NN-DL/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── requirements.in                    # High-level dependencies
├── config.py                          # Configuration constants
│
├── scripts/                           # Preprocessing scripts
│   ├── easyocr_preprocess.py         # OCR text extraction
│   └── ocr_clean.py                  # Text cleaning pipeline
│
├── data/                              # Processed datasets
│   ├── madverse_preprocessed_data.csv # OCR output (raw)
│   └── ocr_ads_cleaned.csv           # Cleaned text data
│
├── checkpoints/                       # Trained model files
│   ├── mobilenetv2_image_model.pth   # Image-only model
│   ├── distilbert_text_model.pth     # Text-only model
│   └── multimodal_fusion_model.pth   # Fusion model
│
├── madverse_data/                     # Raw MADVerse dataset
│   ├── dataset_readme.md             # Dataset documentation
│   ├── annotations/                  # JSON annotations
│   │   ├── adgal_annot_j.json
│   │   ├── epaper1_annotation.json
│   │   ├── epaper2_annotation.json
│   │   └── web_annot_j.json
│   ├── NewsPaperAds/                 # Newspaper advertisement images
│   ├── OnlineAds/                    # Online advertisement images
│   ├── Epaper1/                      # Regional language papers (8 languages)
│   └── Epaper2/                      # Regional language papers (10 languages)
│
├── eda.ipynb                          # Exploratory Data Analysis
└── ad.ipynb                           # Model training notebook
```

## Requirements
**Hardware Recommendations:**
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- RAM: 16GB+ system memory
- Storage: 15GB+ free space (dataset + models)

---

**Developed as part of Neural Networks and Deep Learning coursework**
