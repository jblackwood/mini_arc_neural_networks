"""Analysis script for ARC model predictions."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .train import ARCTaskDataset, ARCTransformer, flatten_collate


def load_model(model_path: str, device: torch.device):
    """Load the trained model from checkpoint.

    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on

    Returns:
        Loaded model in eval mode
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Create model with saved config
    model = ARCTransformer(
        num_tasks=checkpoint["vocab_size"], **checkpoint["model_config"]
    ).to(device)

    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict_output(
    model: ARCTransformer,
    task_idx: torch.Tensor,
    input_grid: torch.Tensor,
    device: torch.device,
):
    """Generate prediction for an input grid using BERT-style masking.

    Args:
        model: Trained ARCTransformer model
        task_idx: Task index tensor
        input_grid: Input grid tensor (H, W)
        device: Device to run inference on

    Returns:
        Predicted output grid as numpy array
    """
    model.eval()
    with torch.no_grad():
        H, W = input_grid.shape

        # Add batch dimension
        task_idx = task_idx.unsqueeze(0).to(device)
        input_grid = input_grid.unsqueeze(0).to(device)

        # Get predictions
        output_logits = model(task_idx, input_grid)  # (1, H*W, num_colors)

        # Get most probable predictions
        predicted = torch.argmax(output_logits, dim=-1)  # (1, H*W)
        predicted_grid = predicted.view(H, W)  # (H, W)

    return predicted_grid.cpu().numpy()


def calculate_test_loss(
    model: ARCTransformer,
    dataset: ARCTaskDataset,
    device: torch.device,
    batch_size: int = 512,
):
    """Calculate total loss on the test dataset.

    Args:
        model: Trained ARCTransformer model
        dataset: Test dataset
        device: Device to run inference on
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (average loss, prediction statistics dict)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=flatten_collate,
    )

    total_loss = 0
    num_batches = 0

    # Statistics for debugging
    all_black_count = 0
    total_predictions = 0
    color_counts = {i: 0 for i in range(10)}  # Count predictions for each color

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Stack batch items
            task_indices = torch.stack([item["task_idx"] for item in batch]).to(device)
            input_grids = torch.stack([item["input"] for item in batch]).to(device)
            output_grids = torch.stack([item["output"] for item in batch]).to(device)

            # Forward pass
            output_logits = model(task_indices, input_grids)

            # Compute output loss
            output_targets = output_grids.view(output_logits.shape[0], -1)
            loss = criterion(
                output_logits.view(-1, output_logits.shape[-1]), output_targets.view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

            # Get predictions for debugging (use output predictions)
            output_predictions = torch.argmax(
                output_logits, dim=-1
            )  # (batch_size, H*W)

            # Vectorized: Check for all-black predictions
            # Check if all predictions in each grid are 0
            is_all_black = (output_predictions == 0).all(dim=1)  # (batch_size,)
            all_black_count += is_all_black.sum().item()
            total_predictions += output_predictions.shape[0]

            # Vectorized: Count color predictions
            for color in range(10):
                color_counts[color] += (output_predictions == color).sum().item()

            print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches...")

    stats = {
        "all_black_count": all_black_count,
        "total_predictions": total_predictions,
        "all_black_percentage": (
            (all_black_count / total_predictions * 100) if total_predictions > 0 else 0
        ),
        "color_counts": color_counts,
    }

    return total_loss / num_batches, stats


def plot_comparison(input_grid, predicted_grid, true_grid, task_name, ax_row):
    """Plot input, predicted, and true grids side by side.

    Args:
        input_grid: Input grid as numpy array
        predicted_grid: Predicted output grid as numpy array
        true_grid: True output grid as numpy array
        task_name: Name of the task for the title
        ax_row: Row of matplotlib axes to plot on
    """
    # ARC color palette (0-9)
    colors = [
        "#000000",  # 0: black
        "#0074D9",  # 1: blue
        "#FF4136",  # 2: red
        "#2ECC40",  # 3: green
        "#FFDC00",  # 4: yellow
        "#AAAAAA",  # 5: grey
        "#F012BE",  # 6: magenta
        "#FF851B",  # 7: orange
        "#7FDBFF",  # 8: sky
        "#870C25",  # 9: maroon
    ]

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(colors)

    # Plot input
    ax_row[0].imshow(input_grid, cmap=cmap, vmin=0, vmax=9)
    ax_row[0].set_title(f"{task_name}\nInput")
    ax_row[0].axis("off")
    ax_row[0].grid(True, which="both", color="white", linewidth=0.5)

    # Plot predicted
    ax_row[1].imshow(predicted_grid, cmap=cmap, vmin=0, vmax=9)
    ax_row[1].set_title("Predicted Output")
    ax_row[1].axis("off")
    ax_row[1].grid(True, which="both", color="white", linewidth=0.5)

    # Plot true
    ax_row[2].imshow(true_grid, cmap=cmap, vmin=0, vmax=9)
    ax_row[2].set_title("True Output")
    ax_row[2].axis("off")
    ax_row[2].grid(True, which="both", color="white", linewidth=0.5)


def main():
    """Main analysis function."""
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Paths
    model_path = "output/mini_arc_analysis/model_20251226_181029.pt"
    data_path = "output/mini_arc_analysis/train"
    output_path = "output/mini_arc_analysis/analysis/5_random_tasks.png"

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    print("Model loaded successfully!")

    # Load test dataset
    print(f"Loading test dataset from {data_path}...")
    test_dataset = ARCTaskDataset(folder_path=data_path, grid_type="test")
    print(f"Test dataset loaded with {len(test_dataset)} tasks")

    # Calculate test loss and get prediction statistics
    print("\nCalculating test loss and analyzing predictions...")
    test_loss, stats = calculate_test_loss(model, test_dataset, device, batch_size=512)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"\n=== Prediction Statistics ===")
    print(f"Total predictions: {stats['total_predictions']}")
    print(
        f"All-black predictions: {stats['all_black_count']} ({stats['all_black_percentage']:.2f}%)"
    )
    print(f"\nColor distribution in predictions:")
    total_pixels = sum(stats["color_counts"].values())
    for color in range(10):
        count = stats["color_counts"][color]
        percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
        print(f"  Color {color}: {count:,} pixels ({percentage:.2f}%)")
    print()

    # Select 5 random tasks
    num_tasks = min(5, len(test_dataset))
    random_indices = random.sample(range(len(test_dataset)), num_tasks)

    # Create figure
    fig, axes = plt.subplots(num_tasks, 3, figsize=(12, 4 * num_tasks))
    if num_tasks == 1:
        axes = axes.reshape(1, -1)

    # Process each random task
    for i, task_idx in enumerate(random_indices):
        print(f"\nProcessing task {i+1}/{num_tasks} (index {task_idx})...")

        # Get task data (list of examples)
        task_examples = test_dataset[task_idx]

        # Use the first test example
        example = task_examples[0]
        task_id_tensor = example["task_idx"]
        input_grid = example["input"]
        true_output = example["output"]

        # Get task name
        task_file = test_dataset.task_files[task_idx]
        task_name = task_file.stem

        print(f"  Task: {task_name}")
        print(f"  Input shape: {input_grid.shape}")
        print(f"  Output shape: {true_output.shape}")

        # Generate prediction
        predicted_output = predict_output(model, task_id_tensor, input_grid, device)

        # Plot comparison
        plot_comparison(
            input_grid.numpy(),
            predicted_output,
            true_output.numpy(),
            task_name,
            axes[i],
        )

        # Calculate accuracy
        accuracy = (predicted_output == true_output.numpy()).mean() * 100
        print(f"  Pixel accuracy: {accuracy:.2f}%")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
