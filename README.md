# Multimodal Advertisement Classification

Multimodal fusion of image and text data for advertisement brand classification using the MADVerse dataset.

## Project Structure

```
NN-DL/
├── ad.ipynb                    # Model training & evaluation notebook
├── eda.ipynb                   # Exploratory data analysis notebook
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── requirements.in             # Base dependencies
├── checkpoints/                # Pretrained model weights
│   ├── mobilenetv2_image_model.pth
│   ├── distilbert_text_model.pth
│   └── multimodal_fusion_model.pth
├── data/                       # Processed datasets
│   ├── madverse_preprocessed_data.csv
│   └── ocr_ads_cleaned.csv
├── scripts/                    # Data processing scripts
│   ├── easyocr_preprocess.py   # OCR text extraction
│   └── ocr_clean.py            # Text cleaning & filtering
├── Reports/                    # Project documentation
│   ├── 20048839 Avinna Maharjan - Proposal.pdf
│   ├── 20048839 Avinna Maharjan - Mid Report.pdf
│   ├── 20048839 Avinna Maharjan - Presentation.pdf
│   └── 20048839 Avinna Maharjan.pdf
├── figures/                    # Generated plots & visualizations
└── madverse_data/              # Raw MADVerse dataset
```

## Dataset

**MADVerse Dataset** - 23,124 advertisement images from newspapers and online sources.

**Source:** [MADVerse on Zenodo](https://zenodo.org/records/10657763)

## Architecture

- **OCR Pipeline**: EasyOCR for text extraction from ad images
- **Image Model**: MobileNetV2 (ImageNet pretrained) + MLP classifier
- **Text Model**: DistilBERT encoder + MLP classifier  
- **Fusion Model**: Late fusion combining image + text features

## Quick Start

```bash
pip install -r requirements.txt
```

## Loading Pretrained Models for Inference

Pretrained models are available in `checkpoints/`:

| Model | File |
|-------|------|
| Image-only | `mobilenetv2_image_model.pth` |
| Text-only | `distilbert_text_model.pth` |
| Multimodal Fusion | `multimodal_fusion_model.pth` |

### Late Fusion Model (MobileNetV2 + DistilBERT)

The Late Fusion Model is available in the [Releases](https://github.com/Avinna10/NN-DL/releases/tag/v1.0) section.

#### Full Inference Code

```python
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2
from transformers import DistilBertTokenizerFast, DistilBertModel

# ==================== Model Definitions ====================

TEXT_DIM = 768

class ImageModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = mobilenet_v2(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()  # 1280-d
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


class TextModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(TEXT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.fc(x)


class FusionModel(nn.Module):
    def __init__(self, image_model, text_model, text_dim=768, num_classes=11):
        super().__init__()
        self.image_encoder = image_model.backbone
        self.text_encoder = text_model.fc[:-1]
        text_hidden_dim = 256

        self.classifier = nn.Sequential(
            nn.Linear(1280 + text_hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, input_ids, attention_mask, distilbert):
        img_feat = self.image_encoder(img)
        with torch.no_grad():
            text_emb = distilbert(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]
        text_feat = self.text_encoder(text_emb)
        fused = torch.cat([img_feat, text_feat], dim=1)
        return self.classifier(fused)


# ==================== Inference Function ====================

def predict_single_image(
    image_path: str,
    text: str = None,
    checkpoint_path: str = "checkpoints/multimodal_fusion_model.pth"
):
    """
    Perform inference on a single advertisement image.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    num_classes = checkpoint["num_classes"]
    label_classes = checkpoint["label_classes"]  # List of class names
    
    # Initialize base models
    image_model = ImageModel(num_classes)
    text_model = TextModel(num_classes)
    
    # Initialize fusion model
    fusion_model = FusionModel(image_model, text_model, num_classes=num_classes)
    fusion_model.load_state_dict(checkpoint["model_state_dict"])
    fusion_model.to(DEVICE)
    fusion_model.eval()
    
    # Load DistilBERT
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    distilbert.to(DEVICE)
    distilbert.eval()
    
    # Image preprocessing (ImageNet normalization)
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)
    
    # Use provided text or extract using OCR
    if text is not None:
        extracted_text = text
    else:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        ocr_results = reader.readtext(image_path)
        extracted_text = " ".join([result[1] for result in ocr_results])
    
    # Tokenize text
    encoding = tokenizer(
        extracted_text if extracted_text.strip() else "advertisement",
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = fusion_model(img_tensor, input_ids, attention_mask, distilbert)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
    
    # Decode prediction using label_classes list
    predicted_class = label_classes[predicted_idx.cpu().item()]
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence.cpu().item(),
        "text_used": extracted_text,
        "all_probabilities": {
            label_classes[i]: prob.item()
            for i, prob in enumerate(probabilities[0])
        }
    }


# ==================== Usage Example ====================

if __name__ == "__main__":
    # Option 1: With OCR (automatic text extraction)
    result = predict_single_image("path/to/your/advertisement.jpg")
    
    # Option 2: With provided text (skips OCR for faster inference)
    result = predict_single_image(
        "path/to/your/advertisement.jpg",
        text="Buy Now! Special Offer on Electronics"
    )
    
    print(f"Predicted Brand: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Text Used: {result['text_used']}")
    print("\nAll class probabilities:")
    for brand, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
        print(f"  {brand}: {prob:.2%}")
```

#### Quick Usage

```python
from inference import predict_single_image  # Save the above code as inference.py

# With automatic OCR text extraction
result = predict_single_image("my_ad_image.jpg")

# Or provide text directly (faster, no OCR needed)
result = predict_single_image("my_ad_image.jpg", text="Your ad text here")

print(f"Brand: {result['predicted_class']} ({result['confidence']:.1%})")
```

Refer to `ad.ipynb` for complete model training and evaluation code.

## Data Preprocessing Scripts

### 1. OCR Text Extraction (`scripts/easyocr_preprocess.py`)

Extracts text from advertisement images using EasyOCR and saves to CSV.

```bash
python scripts/easyocr_preprocess.py
```

### 2. Text Cleaning (`scripts/ocr_clean.py`)

Cleans noisy OCR text and filters low-frequency words.

```bash
python scripts/ocr_clean.py
```

**Pipeline:**
```
Raw OCR → clean_text (character filtering) → filtered_text (frequency filtering)
```
## Reports

Project reports and documentation can be found in the `Reports/` directory.

## Major Technical Outcomes

- **Late Fusion** outperforms unimodal models by combining visual and textual features
- **MobileNetV2 + DistilBERT** achieves efficient inference with low computational overhead
- **OCR-based text extraction** enables end-to-end pipeline from raw ad images to brand classification
- **Corpus-level text filtering** reduces OCR noise and improves text model performance
