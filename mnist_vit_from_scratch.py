"""Converted from `scratch.ipynb`.

This script reproduces the notebook's MNIST Vision Transformer experiment.
All console output and visualizations are persisted into `output/mnist_vit/`.

Run:
  python mnist_vit_from_scratch.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


OUTPUT_DIR = Path("output/mnist_vit")


def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mnist_vit")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if main() is called multiple times.
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    def _excepthook(exc_type, exc, tb):
        logger.exception("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook
    return logger


def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_grid_image(img: torch.Tensor, path: Path, title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    npimg = img.detach().cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


@dataclass(frozen=True)
class Config:
    batch_size: int = 64
    patch_size: int = 4
    img_size: int = 28
    d_model: int = 64
    n_head: int = 4
    n_layers: int = 2
    n_classes: int = 10
    learning_rate: float = 0.001
    epochs: int = 5


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        d_model: int = 64,
        n_head: int = 4,
        n_layers: int = 2,
        n_classes: int = 10,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.patch_embed = nn.Linear(patch_size * patch_size, d_model)

        # Positional embedding and Class token
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 1, 28, 28)
        b, c, h, w = x.shape
        _ = (c, h, w)

        # Create patches
        # Unfold to get patches: (B, C, H, W) -> (B, n_patches, patch_size*patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(b, -1, self.patch_size * self.patch_size)

        # Embed patches
        x = self.patch_embed(x)  # (B, n_patches, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer_encoder(x)

        # Take CLS token output
        x = x[:, 0]

        # Classify
        out = self.classifier(x)
        return out


def main() -> int:
    _ensure_output_dir()
    logger = _setup_logging(OUTPUT_DIR)
    return _main_inner(logger)


def _main_inner(logger: logging.Logger) -> int:
    start_time = time.time()

    # Reproducibility (best-effort; mps/cuda may still be nondeterministic).
    torch.manual_seed(0)
    np.random.seed(0)

    # Set device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    logger.info("Using device: %s", device)

    cfg = Config()
    (OUTPUT_DIR / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n", encoding="utf-8")

    # Hyperparameters
    BATCH_SIZE = cfg.batch_size
    PATCH_SIZE = cfg.patch_size
    IMG_SIZE = cfg.img_size
    D_MODEL = cfg.d_model
    N_HEAD = cfg.n_head
    N_LAYERS = cfg.n_layers
    N_CLASSES = cfg.n_classes
    LEARNING_RATE = cfg.learning_rate
    EPOCHS = cfg.epochs

    # Transformations: Binarize the image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float()),  # Binarize: 0 or 1
        ]
    )

    # Download and load training data
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Download and load test data
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info("Data loaded.")

    # Visualize some binary samples
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    logger.info("Sample binary images:")
    grid = torchvision.utils.make_grid(images[:8])
    _save_grid_image(grid, OUTPUT_DIR / "sample_binary_images.png", title="Sample binary images")
    label_line = "Labels: " + " ".join(f"{labels[j].item()}" for j in range(8))
    logger.info("%s", label_line)
    (OUTPUT_DIR / "sample_binary_images_labels.txt").write_text(label_line + "\n", encoding="utf-8")

    model = VisionTransformer(IMG_SIZE, PATCH_SIZE, D_MODEL, N_HEAD, N_LAYERS, N_CLASSES).to(device)
    logger.info("Model:\n%s", model)
    (OUTPUT_DIR / "model.txt").write_text(str(model) + "\n", encoding="utf-8")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("Starting training...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:
                logger.info(
                    "[Epoch %d, Batch %d] loss: %.3f, Accuracy: %.2f%%",
                    epoch + 1,
                    i + 1,
                    running_loss / 100,
                    100 * correct / total,
                )
                running_loss = 0.0
                correct = 0
                total = 0

    logger.info("Finished Training")

    # Save weights
    torch.save(model.state_dict(), OUTPUT_DIR / "model_state_dict.pt")

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    logger.info("Accuracy of the network on the 10000 test images: %.2f %%", test_acc)

    # Visualize some test predictions
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images_dev = images.to(device)

    outputs = model(images_dev)
    _, predicted = torch.max(outputs, 1)

    logger.info("Test Examples:")

    # Save first 8 images
    grid = torchvision.utils.make_grid(images[:8])
    _save_grid_image(grid, OUTPUT_DIR / "test_examples.png", title="Test Examples")

    gt = "GroundTruth:  " + " ".join(f"{labels[j].item()}" for j in range(8))
    pr = "Predicted:    " + " ".join(f"{predicted[j].item()}" for j in range(8))
    logger.info("%s", gt)
    logger.info("%s", pr)
    (OUTPUT_DIR / "test_examples.txt").write_text(gt + "\n" + pr + "\n", encoding="utf-8")

    # Collect bad predictions (misclassified examples)
    logger.info("Collecting bad predictions...")
    bad_predictions = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images_dev, labels_dev = images.to(device), labels.to(device)
            outputs = model(images_dev)
            _, predicted = torch.max(outputs, 1)

            # Find misclassified examples
            misclassified_mask = predicted != labels_dev
            misclassified_indices = torch.where(misclassified_mask)[0]

            for idx in misclassified_indices:
                bad_predictions.append({
                    'image': images[idx],
                    'true_label': labels[idx].item(),
                    'predicted_label': predicted[idx].item()
                })

                # Stop after collecting 5 bad predictions
                if len(bad_predictions) >= 5:
                    break

            if len(bad_predictions) >= 5:
                break

    # Visualize bad predictions if we found any
    if bad_predictions:
        logger.info("Bad Predictions (Misclassified Examples):")

        # Create grid of misclassified images
        bad_images = torch.stack([bp['image'] for bp in bad_predictions])
        grid = torchvision.utils.make_grid(bad_images)
        _save_grid_image(grid, OUTPUT_DIR / "bad_predictions.png", title="Bad Predictions")

        # Create text output
        gt_bad = "GroundTruth:  " + " ".join(f"{bp['true_label']}" for bp in bad_predictions)
        pr_bad = "Predicted:    " + " ".join(f"{bp['predicted_label']}" for bp in bad_predictions)
        logger.info("%s", gt_bad)
        logger.info("%s", pr_bad)
        (OUTPUT_DIR / "bad_predictions.txt").write_text(gt_bad + "\n" + pr_bad + "\n", encoding="utf-8")
    else:
        logger.info("No bad predictions found! Model is perfect on the test set.")

    # Persist metrics
    metrics = {
        "test_accuracy_percent": float(test_acc),
        "device": str(device),
        "epochs": int(EPOCHS),
        "elapsed_seconds": float(time.time() - start_time),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    logger.info("Wrote outputs to: %s", OUTPUT_DIR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
