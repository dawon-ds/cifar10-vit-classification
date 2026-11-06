# CIFAR-10 Image Classification with Vision Transformer

[Portfolio Page](https://app.notion.com/p/3e968564df5a81de83eded127aa1047f)

Fine-tuned pretrained Vision Transformer (ViT) `google/vit-base-patch16-224-in21k` for CIFAR-10 image classification.

## Approach

**Technologies:** Python, PyTorch, Torchvision, Hugging Face Transformers.

### 1. Data preparation

CIFAR-10 provides 50,000 training images and a separate 10,000-image test partition. The training script uses a seeded split (seed 54321) of the training partition into **45,000 training** and **5,000 validation** images.

- **Input processing:** Resize each 32×32 image to **224×224** and normalize using the pretrained model's image processor statistics.
- **Training augmentation:** Apply random horizontal flips and random crops with 16-pixel padding.
- **Validation processing:** Use a separate dataset instance with deterministic resize and normalization, without random augmentation.

### 2. Model and fine-tuning

The pipeline loads `google/vit-base-patch16-224-in21k` with Hugging Face's `ViTForImageClassification`. Its classification head is configured for **10 CIFAR-10 classes**, using `ignore_mismatched_sizes=True` when loading pretrained weights.

The training loop processes batches of **16** with Adam (learning rate **3e-5**) and a **cosine learning-rate scheduler** with no warmup. Each step computes the model's supervised classification loss, updates the weights through backpropagation, and advances the scheduler.

After each epoch, the model switches to evaluation mode, calculates validation accuracy from predicted classes, and saves its weights to `runs/vit_cifar10/epoch_<n>.pth`.

### 3. Experiment configuration and evaluation

Two fine-tuning settings were compared: **Setting #1** used one epoch; **Setting #2** used three epochs and weight decay **5e-5**. The original weight-decay value for Setting #1 is not recorded. The included YAML config specifies Setting #2.

The separate evaluation script loads a specified checkpoint and evaluates it on the **official CIFAR-10 test partition** with deterministic preprocessing. This test evaluation is separate from the validation scores presented below. The pretrained weights, image processor, and CIFAR-10 data require initial download access or locally cached files.

## Results

| Setting | Epochs | Weight Decay | Validation Accuracy | Training Time |
|---|---:|---:|---:|---:|
| #1 | 1 | - | 98.35% | 16.28 min |
| #2 | 3 | 5e-5 | 98.75% | 52.20 min |

Setting #2 achieved **0.40 percentage points** higher validation accuracy with approximately **3.2 times** the training time. Epoch count and weight decay changed together, so this comparison does not isolate either factor.

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

## Data and Third-party Resources

- **Data:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), downloaded through Torchvision; check source terms before redistributing images.
- **Model:** Pretrained ViT weights are loaded from the Hugging Face model ID in the configuration and are subject to the publisher's terms.

Dependency lists are in `requirements.txt`; unless versions are explicitly pinned there, they do not reproduce an exact historical software environment. No redistribution permission for third-party inputs is implied by this repository.
