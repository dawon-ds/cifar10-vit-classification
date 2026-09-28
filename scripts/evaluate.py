import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoImageProcessor

from models import build_model


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/vit_cifar10.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["model"]["name"]
    image_size = cfg["data"]["image_size"]
    processor = AutoImageProcessor.from_pretrained(model_name)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    model = build_model(model_name, cfg["model"]["num_labels"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(pixel_values=images).logits
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"test_accuracy={correct / total:.4f}")


if __name__ == "__main__":
    main()

