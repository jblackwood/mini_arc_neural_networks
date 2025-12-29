"""Analysis script for mini_arc_eqm model."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from mini_arc_eqm.train import ARCTaskDataset, TransformerModel


def load_model_and_embedding(model_path: str, device: torch.device):
    """Load the trained model and embedding from checkpoint.

    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on

    Returns:
        Tuple of (model, embedding, config)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config
    config = checkpoint["config"]

    # Create embedding
    embedding = nn.Embedding(config["num_cell_values"], config["d_model"]).to(device)
    embedding.load_state_dict(checkpoint["embedding_state_dict"])
    embedding.eval()

    # Create model
    model = TransformerModel(
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        seq_len=config["seq_len"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, embedding, config


def find_nearest_embedding(embeddings: torch.Tensor, target: torch.Tensor) -> int:
    """Find the nearest embedding to the target vector.

    Args:
        embeddings: Embedding matrix of shape (num_embeddings, d_model)
        target: Target vector of shape (d_model,)

    Returns:
        Index of the nearest embedding
    """
    # Compute distances
    distances = torch.norm(embeddings - target.unsqueeze(0), dim=1)
    return torch.argmin(distances).item()


def plot_all_grids(
    task_data: torch.Tensor, predicted_grid: np.ndarray, output_path: str
):
    """Plot all input/output grids for the task plus the denoised output grid.

    Args:
        task_data: Full task data tensor (200,) containing 4 examples
        predicted_grid: Predicted output grid from denoising (5x5)
        output_path: Path to save the output image
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

    cmap = plt.matplotlib.colors.ListedColormap(colors)

    # Extract all grids from task_data
    # Task has 4 examples, each with input (25 cells) and output (25 cells)
    # Total: 4 * 2 * 25 = 200 cells
    task_np = task_data.cpu().numpy()

    grids = []
    for i in range(4):
        start_idx = i * 50  # Each example has 50 cells (input + output)
        input_grid = task_np[start_idx : start_idx + 25].reshape(5, 5)
        output_grid = task_np[start_idx + 25 : start_idx + 50].reshape(5, 5)
        grids.append(("input", input_grid))
        grids.append(("output", output_grid))

    # Create figure with 3 rows and 3 columns
    # Row 1: Example 1 input, Example 1 output, Example 2 input
    # Row 2: Example 2 output, Example 3 input, Example 3 output
    # Row 3: Example 4 input, Example 4 output, Denoised output
    _, axes = plt.subplots(3, 3, figsize=(12, 12))

    # Plot first 8 grids (4 examples × 2 grids each)
    for idx, (grid_type, grid) in enumerate(grids):
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(grid, cmap=cmap, vmin=0, vmax=9)
        example_num = idx // 2 + 1
        axes[row, col].set_title(f"Example {example_num} {grid_type.capitalize()}")
        axes[row, col].axis("off")

    # Plot denoised output in position [2, 2]
    axes[2, 2].imshow(predicted_grid, cmap=cmap, vmin=0, vmax=9)
    axes[2, 2].set_title("Denoised Output (Example 4)")
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    """Main analysis function."""
    # Configuration
    model_path = "output/mini_arc_eqm/models/20251229_130936_model.pt"
    test_data_path = "output/mini_arc_eqm/test"
    output_dir = Path("output/mini_arc_eqm/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model and embedding
    print(f"Loading model from {model_path}...")
    model, embedding, config = load_model_and_embedding(model_path, device)
    print(f"Model loaded successfully! d_model={config['d_model']}")

    # Load test dataset
    print(f"Loading test dataset from {test_data_path}...")
    test_dataset = ARCTaskDataset(test_data_path)
    print(f"Test dataset loaded with {len(test_dataset)} tasks")

    # Select a random task
    random.seed(41)
    task_idx = random.randint(0, len(test_dataset) - 1)
    task_data = test_dataset[task_idx]  # Shape: (200,)
    print(f"\nSelected task index: {task_idx}")

    # Convert to embeddings
    with torch.no_grad():
        x_clean = embedding(
            task_data.unsqueeze(0).to(device)
        )  # Shape: (1, 200, d_model)

    # Create noised input: set final 25 tokens to 0, then add gaussian noise
    x_i = x_clean.clone()
    x_i[0, -25:, :] = 0  # Set final 25 tokens to 0
    x_i[0, -25:, :] += torch.randn(
        25, config["d_model"], device=device
    )  # Add gaussian noise

    print(f"Created noised input with shape: {x_i.shape}")

    # Optimization parameters
    eta = 0.003
    mu = 0.3
    num_iterations = 300

    # Perform optimization
    print("\nStarting optimization...")
    with torch.no_grad():
        x = x_i.clone()
        grad = model(x)

        for i in range(num_iterations):
            x_last = x.clone()
            x = x - eta * grad

            # Compute gradient at momentum point
            grad = model(x + mu * (x - x_last))

            # Print gradient norm
            grad_norm = torch.norm(grad).item()
            print(f"Iteration {i+1}/{num_iterations}: grad_norm = {grad_norm:.6f}")

    print("\nOptimization complete!")

    # Extract the last 25 grid tokens and find nearest neighbors
    final_tokens = x[0, -25:, :]  # Shape: (25, d_model)
    embedding_matrix = embedding.weight.data  # Shape: (10, d_model)

    predicted_values = []
    for token in final_tokens:
        nearest_idx = find_nearest_embedding(embedding_matrix, token)
        predicted_values.append(nearest_idx)

    # Reshape to 5x5 grid
    predicted_grid = np.array(predicted_values).reshape(5, 5)

    # Get true output grid (last 25 values of the task)
    true_values = task_data[-25:].cpu().numpy()
    true_grid = true_values.reshape(5, 5)

    # Print all grids for the task
    print("\n" + "=" * 60)
    print("ALL GRIDS FOR THE TASK")
    print("=" * 60)

    task_np = task_data.cpu().numpy()
    for i in range(4):
        start_idx = i * 50
        input_grid = task_np[start_idx : start_idx + 25].reshape(5, 5)
        output_grid = task_np[start_idx + 25 : start_idx + 50].reshape(5, 5)

        print(f"\nExample {i+1} Input:")
        print(input_grid)
        print(f"\nExample {i+1} Output:")
        print(output_grid)

    print("\n" + "=" * 60)
    print("DENOISING RESULTS")
    print("=" * 60)
    print("\nDenoised output grid (Example 4):")
    print(predicted_grid)
    print("\nTrue output grid (Example 4):")
    print(true_grid)

    # Plot and save
    output_path = output_dir / "denoising_result.png"
    plot_all_grids(task_data, predicted_grid, str(output_path))

    # Calculate accuracy
    accuracy = (predicted_grid == true_grid).mean()
    print(f"\nAccuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
