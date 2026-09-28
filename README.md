# CIFAR-10 Image Classification with Vision Transformer

Fine-tuned a pre-trained Vision Transformer (ViT) for CIFAR-10 image classification.

## Overview
- **Task:** CIFAR-10 image classification
- **Model:** `google/vit-base-patch16-224-in21k`
- **Framework:** PyTorch + Hugging Face Transformers
- **Train / Validation split:** 45,000 / 5,000 images
- **Optimizer:** Adam
- **Batch size:** 16
- **Learning rate:** 3e-5
- **Scheduler:** Cosine learning-rate schedule
- **Data augmentation:** Applied during training

## Experiments
| Setting | Epochs | Weight Decay | Validation Accuracy | Training Time |
|---|---:|---:|---:|---:|
| #1 | 1 | - | 98.35% | 16.28 min |
| #2 | 3 | 5e-5 | 98.75% | 52.20 min |

## Project Structure
```text
cifar10-vit-classification/
├── config/
│   └── vit_cifar10.yaml
├── models/
│   ├── __init__.py
│   └── vit.py
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   └── evaluate.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Usage
Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python -m scripts.train --config config/vit_cifar10.yaml
```

Evaluate:

```bash
python -m scripts.evaluate --config config/vit_cifar10.yaml --checkpoint <checkpoint_path>
```

## Notes
The original coursework code was reorganized into a reusable portfolio structure. Local absolute paths and hard-coded run timestamps were removed.


