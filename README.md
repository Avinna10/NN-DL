# Multimodal Advertisement Classification

Multimodal fusion of image and text data for advertisement brand classification using the MADVerse dataset.

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

### Inference Example

```python
import torch

# Load checkpoint
checkpoint = torch.load("checkpoints/multimodal_fusion_model.pth")

# Reconstruct model (see ad.ipynb for model class definitions)
model = MultimodalFusionModel(num_classes=checkpoint['num_classes'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get label encoder for class names
label_encoder = checkpoint['label_encoder']

# Run inference
# - Images: resize to 224×224, normalize with ImageNet stats
# - Text: tokenize with DistilBERT (max_length=32)
```

Refer to `ad.ipynb` for complete model definitions and preprocessing code.

## Reports

Project reports and documentation can be found in the `Reports/` directory.

## Project Structure

```
NN-DL/
├── ad.ipynb              # Model training & inference
├── eda.ipynb             # Exploratory data analysis
├── checkpoints/          # Pretrained models
├── data/                 # Processed datasets
├── scripts/              # OCR & cleaning scripts
├── Reports/              # Project reports
└── madverse_data/        # Raw dataset
```
