"""Analysis script for mini_arc_eqm model."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from mini_arc_eqm.train import ARCTaskDataset, TransformerModel


@dataclass
class DenoisingResult:
    """Result from denoising evaluation.

    Attributes:
        accuracies: Optional tensor of shape (batch_size,) with accuracy for each task
        predicted_grids: Optional numpy array of shape (batch_size, 5, 5) with predicted output grids
        num_iterations: Optional tensor of shape (batch_size,) with number of iterations for each task
        optimized_output_tokens: Optional tensor of shape (batch_size, 25, d_model) with optimized output tokens
        best_grad_norm: Optional tensor of shape (batch_size,) with best gradient norm for each task
    """

    accuracies: Optional[torch.Tensor] = None
    predicted_grids: Optional[np.ndarray] = None
    num_iterations: Optional[torch.Tensor] = None
    optimized_output_tokens: Optional[torch.Tensor] = None
    best_grad_norm: Optional[torch.Tensor] = None


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
    return int(torch.argmax(vector[:10]).item())


def optimize_output_grid(
    model,
    x_input: torch.Tensor,
    mu: float,
    eta: float,
    num_iterations: int,
    patience: int,
) -> DenoisingResult:
    """Optimize the output grid using gradient descent with early stopping.

    Args:
        model: The transformer model to use for computing gradients
        x_input: Input tensor of shape (batch_size, 200, d_model)
        mu: Momentum parameter for gradient computation
        eta: Learning rate for optimization
        num_iterations: Maximum number of optimization iterations
        patience: Number of iterations to wait for improvement before early stopping

    Returns:
        DenoisingResult with optimized_output_tokens and num_iterations fields populated
    """
    batch_size = x_input.shape[0]

    with torch.no_grad():
        x = x_input.clone()
        grad = model(x)

        # Track best grid and gradient norm
        best_grad_norm = float("inf")
        best_x = x.clone()

        # Track gradient norm history for early stopping
        grad_norm_history = []
        iterations_without_improvement = 0
        iterations_taken = num_iterations  # Default to max iterations

        for iteration in range(num_iterations):
            x_last = x.clone()

            # Zero out gradient for first 175 tokens
            grad[:, :175, :] = 0

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
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            # Early stopping check
            if iterations_without_improvement >= patience:
                iterations_taken = iteration + 1
                break

        # Use the best x (with lowest gradient norm) as final result
        x = best_x

    # Create iterations tensor
    iterations_tensor = torch.full((batch_size,), iterations_taken, dtype=torch.int32)

    # Create best grad norm tensor
    best_grad_norm_tensor = torch.full(
        (batch_size,), best_grad_norm, dtype=torch.float32
    )

    # Return DenoisingResult with optimized tokens, iterations, and best grad norm
    return DenoisingResult(
        optimized_output_tokens=x[:, -25:, :],
        num_iterations=iterations_tensor,
        best_grad_norm=best_grad_norm_tensor,
    )


def evaluate_denoising_accuracy(
    model,
    x_clean: torch.Tensor,
    gamma: float,
    mu: float,
    eta: float,
    num_iterations: int,
    patience: int,
) -> DenoisingResult:
    """Evaluate denoising accuracy by corrupting and denoising output grids.

    Args:
        model: The transformer model to use for computing gradients
        x_clean: Clean input tensor of shape (batch_size, 200, d_model)
        gamma: Noise level parameter (0-1) for corrupting the output grid
        mu: Momentum parameter for gradient computation
        eta: Learning rate for optimization
        num_iterations: Maximum number of optimization iterations
        patience: Number of iterations to wait for improvement before early stopping

    Returns:
        DenoisingResult containing accuracies and predicted grids
    """
    batch_size = x_clean.shape[0]

    # Create noised input - only noise the last 25 tokens, keep first 175 unnoised
    x_i = x_clean.clone()
    eps = torch.randn_like(x_clean[:, -25:, :])
    x_i[:, -25:, :] = (1 - gamma) * eps + gamma * x_clean[:, -25:, :]

    # Perform optimization to denoise
    opt_result = optimize_output_grid(
        model=model,
        x_input=x_i,
        mu=mu,
        eta=eta,
        num_iterations=num_iterations,
        patience=patience,
    )

    # Decode the optimized output grids
    assert opt_result.optimized_output_tokens is not None
    predicted_grids = decode_grids(
        opt_result.optimized_output_tokens
    )  # Shape: (batch_size, 1, 5, 5)
    predicted_grids = predicted_grids[:, 0, :, :]  # Shape: (batch_size, 5, 5)

    # Decode the true output grids (last 25 tokens)
    true_output_tokens = x_clean[:, -25:, :]  # Shape: (batch_size, 25, d_model)
    true_grids = decode_grids(true_output_tokens)  # Shape: (batch_size, 1, 5, 5)
    true_grids = true_grids[:, 0, :, :]  # Shape: (batch_size, 5, 5)

    # Calculate accuracy for each task in the batch
    accuracies = []
    for batch_idx in range(batch_size):
        accuracy = (predicted_grids[batch_idx] == true_grids[batch_idx]).mean()
        accuracies.append(accuracy)

    # Use replace to add accuracies and predicted_grids to the result
    return replace(
        opt_result,
        accuracies=torch.tensor(accuracies),
        predicted_grids=predicted_grids,
    )


def decode_grids(tokens: torch.Tensor) -> np.ndarray:
    """Decode grid tokens to integer values.

    Args:
        tokens: Tensor of shape (batch_size, num_tokens, d_model) containing grid tokens
                where num_tokens should be a multiple of 25 (for 5x5 grids)

    Returns:
        Numpy array of shape (batch_size, num_grids, 5, 5) with decoded integer values (0-9)
        where num_grids = num_tokens // 25
    """
    batch_size = tokens.shape[0]
    num_tokens = tokens.shape[1]

    assert (
        num_tokens % 25 == 0
    ), f"num_tokens must be a multiple of 25, got {num_tokens}"

    num_grids = num_tokens // 25

    decoded_grids = []

    for batch_idx in range(batch_size):
        batch_grids = []
        for grid_idx in range(num_grids):
            start_idx = grid_idx * 25
            end_idx = start_idx + 25
            grid_tokens = tokens[batch_idx, start_idx:end_idx]

            predicted_values = []
            for token in grid_tokens:
                cell_value = decode_one_hot(token)
                predicted_values.append(cell_value)
            predicted_grid = np.array(predicted_values).reshape(5, 5)
            batch_grids.append(predicted_grid)

        decoded_grids.append(batch_grids)

    return np.array(decoded_grids)


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

    cmap = mcolors.ListedColormap(colors)

    # Extract all grids from task_data
    # Task has 4 examples, each with input (25 cells) and output (25 cells)
    # Total: 4 * 2 * 25 = 200 cells
    examples = []
    for i in range(4):
        start_idx = i * 50  # Each example has 50 cells (input + output)
        input_grid = task_data[start_idx : start_idx + 25].reshape(5, 5)
        output_grid = task_data[start_idx + 25 : start_idx + 50].reshape(5, 5)
        examples.append((input_grid, output_grid))

    # Create figure with 5 rows and 2 columns
    # Row 0: Example 1 input, Example 1 output
    # Row 1: Example 2 input, Example 2 output
    # Row 2: Example 3 input, Example 3 output
    # Row 3: Example 4 input, Example 4 output
    # Row 4: Empty, Denoised output
    _, axes = plt.subplots(5, 2, figsize=(8, 20))

    # Plot 4 examples (each row has input and output)
    for example_idx, (input_grid, output_grid) in enumerate(examples):
        # Plot input in first column
        axes[example_idx, 0].imshow(input_grid, cmap=cmap, vmin=0, vmax=9)
        axes[example_idx, 0].set_title(f"Example {example_idx + 1} Input")
        axes[example_idx, 0].axis("off")

        # Plot output in second column
        axes[example_idx, 1].imshow(output_grid, cmap=cmap, vmin=0, vmax=9)
        axes[example_idx, 1].set_title(f"Example {example_idx + 1} Output")
        axes[example_idx, 1].axis("off")

    # Last row: empty first column, denoised output in second column
    axes[4, 0].axis("off")  # Empty
    axes[4, 1].imshow(predicted_grid, cmap=cmap, vmin=0, vmax=9)
    axes[4, 1].set_title("Denoised Output (Example 4)")
    axes[4, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    """Main analysis function."""
    # Configuration
    model_path = "output/mini_arc_eqm2/models/20251230_094732_model.pt"
    test_data_path = "output/mini_arc_eqm2/test"
    output_dir = Path("output/mini_arc_eqm2/analysis")
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

    # Load dataset
    print(f"Loading dataset from {test_data_path}...")
    test_dataset = ARCTaskDataset(test_data_path, d_model=config["d_model"])

    # Filter for original task files
    original_tasks = [
        (idx, file_path)
        for idx, file_path in enumerate(test_dataset.task_files)
        if file_path.name.endswith("original.json")
    ]

    print(f"Found {len(original_tasks)} original task files")

    # Optimization parameters
    gamma = 0.8
    eta = 0.003
    mu = 0.3
    num_iterations = 2000
    patience = 50  # Number of iterations to wait for improvement

    print(f"Using gamma = {gamma}")
    print(f"Using eta = {eta}, mu = {mu}")
    print(f"Using num_iterations = {num_iterations}, patience = {patience}")

    # Create output directory for this gamma value
    gamma_output_dir = output_dir / f"denoise_gamma_{gamma}"
    gamma_output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each original task
    print("\n" + "=" * 80)
    print("EVALUATING ORIGINAL TASKS")
    print("=" * 80)

    for task_idx, task_path in original_tasks:
        print(f"\nProcessing: {task_path.name}")

        # Load task data
        task_data = test_dataset[task_idx]  # Shape: (200, d_model)

        # Get clean one-hot encoded data
        x_clean = task_data.unsqueeze(0).to(device)  # Shape: (1, 200, d_model)

        # Perform denoising evaluation
        result = evaluate_denoising_accuracy(
            model=model,
            x_clean=x_clean,
            gamma=gamma,
            mu=mu,
            eta=eta,
            num_iterations=num_iterations,
            patience=patience,
        )

        # Extract predicted grid from result
        assert result.predicted_grids is not None
        assert result.accuracies is not None
        assert result.num_iterations is not None
        assert result.best_grad_norm is not None

        predicted_grid = result.predicted_grids[0]  # Shape: (5, 5)
        accuracy = result.accuracies[0].item()
        iterations = result.num_iterations[0].item()
        grad_norm = result.best_grad_norm[0].item()

        # Decode all true grids from task data (200 tokens = 8 grids of 25 tokens each)
        all_true_grids = decode_grids(task_data.unsqueeze(0))  # Shape: (1, 8, 5, 5)
        all_true_grids = all_true_grids[0]  # Extract first batch: (8, 5, 5)

        # Plot and save - flatten all true grids for plotting
        all_true_values = all_true_grids.reshape(-1)  # Flatten (8, 5, 5) -> (200,)
        output_path = gamma_output_dir / f"{task_path.stem}.png"
        plot_all_grids(all_true_values, predicted_grid, str(output_path))

        # Print accuracy, iterations, and gradient norm
        print(
            f"  Accuracy: {accuracy * 100:.2f}% | Iterations: {iterations} | Grad Norm: {grad_norm:.6f}"
        )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
