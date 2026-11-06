# CIFAR-10 Image Classification with Vision Transformer

[Portfolio Page](https://app.notion.com/p/3e968564df5a81de83eded127aa1047f)

Fine-tuned pretrained Vision Transformer (ViT) `google/vit-base-patch16-224-in21k` for CIFAR-10 image classification.

## Approach

**Technologies:** Python, PyTorch, Hugging Face Transformers, Torchvision.

- **Data:** Seeded split of 45,000 training and 5,000 validation images from CIFAR-10's training partition.
- **Preprocessing:** Resize 32×32 images to 224×224 and normalize with the pretrained image processor's statistics. Random crop and horizontal flip are used for training only.
- **Model:** Adapt the pretrained ViT classification head to 10 CIFAR-10 labels.
- **Training:** Adam, batch size 16, learning rate 3e-5, cosine scheduler with zero warmup steps.

The separate evaluation script uses the official CIFAR-10 test partition with deterministic preprocessing.

The default YAML corresponds to Setting #2. To reproduce the assignment's Setting #1 configuration, use one epoch and confirm its original weight-decay value; the historical result table does not specify that value.

Checkpoints are saved as `runs/vit_cifar10/epoch_<n>.pth`. Pass one of these paths to the evaluation command. Torchvision downloads CIFAR-10; the pretrained model and processor also require initial download access or a local cache.

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
