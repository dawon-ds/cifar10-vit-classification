from transformers import ViTForImageClassification


def build_model(model_name: str, num_labels: int = 10):
    """Create a ViT classifier for CIFAR-10 fine-tuning."""
    return ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
