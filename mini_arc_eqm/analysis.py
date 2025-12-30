"""Analysis script for mini_arc_eqm model."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mini_arc_eqm.train import ARCTaskDataset, TransformerModel


def load_model(model_path: str, device: torch.device):
    """Load the trained model from checkpoint.

    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on

    Returns:
        Tuple of (model, config)
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract config
    config = checkpoint["config"]

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

    return model, config


def decode_one_hot(vector: torch.Tensor) -> int:
    """Decode a one-hot encoded vector to its cell value.

    Args:
        vector: One-hot encoded vector of shape (d_model,)

    Returns:
        Cell value (0-9) corresponding to the argmax of the first 10 dimensions
    """
    # Take argmax of first 10 dimensions (cell values 0-9)
    return torch.argmax(vector[:10]).item()


def plot_all_grids(task_data: np.ndarray, predicted_grid: np.ndarray, output_path: str):
    """Plot all input/output grids for the task plus the denoised output grid.

    Args:
        task_data: Full task data as numpy array (200,) containing decoded cell values
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
    grids = []
    for i in range(4):
        start_idx = i * 50  # Each example has 50 cells (input + output)
        input_grid = task_data[start_idx : start_idx + 25].reshape(5, 5)
        output_grid = task_data[start_idx + 25 : start_idx + 50].reshape(5, 5)
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
    model_path = (
        "output/mini_arc_eqm/checkpoints/20251229_220918_epoch_20_checkpoint.pt"
    )
    test_data_path = "output/mini_arc_eqm/train"
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

    # Load model
    print(f"Loading model from {model_path}...")
    model, config = load_model(model_path, device)
    print(f"Model loaded successfully! d_model={config['d_model']}")

    # Load test dataset
    print(f"Loading test dataset from {test_data_path}...")
    test_dataset = ARCTaskDataset(test_data_path, d_model=config["d_model"])
    print(f"Test dataset loaded with {len(test_dataset)} tasks")

    # Select a random task
    random.seed(2)
    task_idx = random.randint(0, len(test_dataset) - 1)
    task_data = test_dataset[task_idx]  # Shape: (200, d_model)
    print(f"\nSelected task index: {task_idx}")

    # Get clean one-hot encoded data
    x_clean = task_data.unsqueeze(0).to(device)  # Shape: (1, 200, d_model)

    # Create noised input - only noise the last 25 tokens, keep first 175 unnoised
    gamma = 0.8
    x_i = x_clean.clone()
    # Only add noise to the last 25 tokens
    eps = torch.randn_like(x_clean[:, -25:, :])
    x_i[:, -25:, :] = (1 - gamma) * eps + gamma * x_clean[:, -25:, :]

    print(f"Created noised input with shape: {x_i.shape}")
    print(f"Using gamma = {gamma}")

    # Optimization parameters
    eta = 0.003
    mu = 0.3
    num_iterations = 3000

    # Early stopping parameters
    patience = 50  # Number of iterations to wait for improvement

    # Perform optimization
    print("\nStarting optimization...")
    with torch.no_grad():
        x = x_i.clone()
        grad = model(x)

        # Track best grid and gradient norm
        best_grad_norm = float("inf")
        best_x = x.clone()
        best_iteration = 0

        # Track gradient norm history for early stopping
        grad_norm_history = []
        iterations_without_improvement = 0

        for i in range(num_iterations):
            x_last = x.clone()

            # Zero out gradient for first 175 tokens
            grad[0, :175, :] = 0

            # Update x
            x = x - eta * grad

            # Compute gradient
            grad = model(x + mu * (x - x_last))

            # Calculate gradient norm
            grad_norm = torch.norm(grad).item()
            grad_norm_history.append(grad_norm)

            # Update best grid if current gradient norm is lower
            if grad_norm < best_grad_norm:
                best_grad_norm = grad_norm
                best_x = x.clone()
                best_iteration = i + 1
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Decode current output grid (last 25 tokens)
            current_output_tokens = x[0, -25:, :]
            current_predicted_values = []
            for token in current_output_tokens:
                cell_value = decode_one_hot(token)
                current_predicted_values.append(cell_value)
            current_predicted_grid = np.array(current_predicted_values).reshape(5, 5)

            print(f"Iteration {i+1}/{num_iterations}")
            print(
                f"grad_norm = {grad_norm:.6f} | best_grad_norm = {best_grad_norm:.6f} (iter {best_iteration})"
            )
            print(
                f"iterations_without_improvement = {iterations_without_improvement}/{patience}"
            )

            # Print current predicted output grid
            print(f"\nCurrent predicted output grid (Example 4):")
            print(current_predicted_grid)

            # Early stopping check
            if iterations_without_improvement >= patience:
                print(f"\n*** Early stopping triggered at iteration {i+1} ***")
                print(f"No improvement in gradient norm for {patience} iterations")
                print(
                    f"Best gradient norm: {best_grad_norm:.6f} at iteration {best_iteration}"
                )
                break

        # Use the best x (with lowest gradient norm) as final result
        x = best_x
        print(f"\nOptimization complete!")
        print(
            f"Using grid from iteration {best_iteration} with grad_norm = {best_grad_norm:.6f}"
        )

    # Decode ALL 200 tokens from the final denoised output
    all_predicted_values = []
    for token in x[0]:  # Iterate through all 200 tokens
        cell_value = decode_one_hot(token)
        all_predicted_values.append(cell_value)

    # Convert to numpy array
    all_predicted_values = np.array(all_predicted_values)

    # Extract the last 25 tokens for the main predicted grid (Example 4 output)
    predicted_grid = all_predicted_values[-25:].reshape(5, 5)

    # Get true output grid (last 25 values of the task)
    # Decode the one-hot encoded values
    true_values = []
    for i in range(175, 200):  # Last 25 tokens
        cell_value = decode_one_hot(task_data[i])
        true_values.append(cell_value)
    true_values = np.array(true_values)
    true_grid = true_values.reshape(5, 5)

    print("\n" + "=" * 60)

    print("\n" + "=" * 60)
    print("DENOISING RESULTS (Example 4 Output)")
    print("=" * 60)
    print("\nDenoised output grid (Example 4):")
    print(predicted_grid)
    print("\nTrue output grid (Example 4):")
    print(true_grid)

    # Plot and save
    output_path = output_dir / "denoising_result.png"
    all_true_values = []
    for i in range(200):
        cell_value = decode_one_hot(task_data[i])
        all_true_values.append(cell_value)
    all_true_values = np.array(all_true_values)
    plot_all_grids(all_true_values, predicted_grid, str(output_path))

    # Calculate accuracy
    accuracy = (predicted_grid == true_grid).mean()
    print(f"\nAccuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
