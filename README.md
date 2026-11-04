# CIFAR-10 Image Classification with Vision Transformer

Fine-tuned a pre-trained Vision Transformer (ViT) for CIFAR-10 image classification.

[Portfolio](https://incredible-march-0ef.notion.site/CIFAR-10-Image-Classification-with-Vision-Transformer-3e968564df5a81de83eded127aa1047f)

## Project Overview

- **Task:** CIFAR-10 image classification
- **Model:** `google/vit-base-patch16-224-in21k`
- **Framework:** PyTorch + Hugging Face Transformers
- **Train / Validation split:** 45,000 / 5,000 images
- **Optimizer:** Adam
- **Batch size:** 16
- **Learning rate:** 3e-5
- **Scheduler:** Cosine learning-rate schedule
- **Data augmentation:** Applied during training

## Approach

CIFAR-10 images are resized from 32×32 to **224×224** and normalized using the pretrained model's image processor. The model's classification head is adapted to 10 labels, then fine-tuned using Adam and a cosine schedule with zero warmup steps.

The 50,000-image training partition is split into **45,000 training** and **5,000 validation** examples using a seeded split. The separate evaluation script uses the official test partition with deterministic resizing and normalization.

The default YAML corresponds to Setting #2. To reproduce the assignment's Setting #1 configuration, use one epoch and confirm its original weight-decay value; the historical result table does not specify that value.

Checkpoints are saved as `runs/vit_cifar10/epoch_<n>.pth`. Pass one of these paths to the evaluation command. Torchvision downloads CIFAR-10; the pretrained model and processor also require initial download access or a local cache.

## Results

| Setting | Epochs | Weight Decay | Validation Accuracy | Training Time |
|---|---:|---:|---:|---:|
| #1 | 1 | - | 98.35% | 16.28 min |
| #2 | 3 | 5e-5 | 98.75% | 52.20 min |

### Interpretation

**Evaluation caveat:** the reported validation accuracies were obtained before the validation-transform correction. The updated code uses deterministic validation transforms; new results require retraining.

Setting #2 has a reported validation accuracy **0.40 percentage points** higher than Setting #1. Epoch count and weight decay change together, so the comparison does not isolate a single factor.

## Repository Structure

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

## Limitations

The current training code uses **separate training and validation dataset instances** sharing a seeded index split. Only training uses random crop/flip; validation uses deterministic resize and normalization. The historical accuracy values above were recorded before this correction and have not been re-measured.

The pipeline saves each epoch rather than selecting a best checkpoint automatically. The reported validation scores should not be compared directly with another project's test accuracy without matching datasets and evaluation protocols.

## Review

This project provided experience adapting pretrained image Transformers, handling image preprocessing, and coordinating fine-tuning with learning-rate scheduling.

## Data and Third-party Resources

- **Data:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), downloaded through Torchvision; check source terms before redistributing images.
- **Model:** pretrained ViT weights are loaded from the Hugging Face model ID in the configuration and are subject to the publisher's terms. The displayed accuracy predates the validation-transform correction.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
