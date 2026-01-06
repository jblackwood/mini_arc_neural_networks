import json
import os
import pprint
import random
import shutil
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    """Configuration for ARC dataset creation and model training."""

    # Dataset creation parameters
    data_dir: Path
    output_dir: Path
    test_ratio: float
    random_seed: int
    max_augmentations: int

    # Model parameters
    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float

    # Training parameters
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    mode: Literal["train", "learning_rate_test", "eval"]
    checkpoint_save_interval: int

    # Data parameters
    seq_len: int
    vocab_size: int

    # Denoising evaluation parameters
    eval_denoise_epoch_interval: int
    eval_denoise_mu: float
    eval_denoise_eta: float
    eval_denoise_num_iterations: int

    # Google Drive location for Colab
    google_drive_dir: str

    # Optional model loading
    load_model_path: Optional[str] = None

    # Paths (computed)
    timestamp: str = ""
    tensorboard_log_dir: str = ""
    model_save_dir: str = ""
    model_save_path: str = ""
    checkpoint_dir: str = ""
    train_data_dir: str = ""
    test_data_dir: str = ""

    def __post_init__(self):
        """Initialize computed paths."""
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not self.tensorboard_log_dir:
            self.tensorboard_log_dir = f"{self.output_dir}/runs/{self.timestamp}_model"
        if not self.model_save_dir:
            self.model_save_dir = f"{self.output_dir}/models"
        if not self.model_save_path:
            self.model_save_path = f"{self.model_save_dir}/{self.timestamp}_model.pt"
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"{self.output_dir}/checkpoints"
        if not self.train_data_dir:
            self.train_data_dir = f"{self.output_dir}/train"
        if not self.test_data_dir:
            self.test_data_dir = f"{self.output_dir}/test"


@dataclass
class ARCExample:
    """Represents a single ARC example with input and output grids."""

    input: List[List[int]]
    output: List[List[int]]


@dataclass
class ARCTask:
    """Represents an ARC task with training and test examples."""

    train: List[ARCExample]
    test: List[ARCExample]
    task_id: Optional[str] = None
    task_type: Optional[Literal["original", "augmentation"]] = None
    transformation: Optional[str] = None
    color_permutation: Optional[Dict[int, int]] = None


@dataclass
class DenoisingResult:
    """Result from denoising evaluation.

    Attributes:
        accuracies: Optional tensor of shape (batch_size,) with accuracy for each task
        predicted_grids: Optional tensor of shape (batch_size, 5, 5) with predicted output grids
        optimized_output_tokens: Optional tensor of shape (batch_size, 25, d_model) with optimized output tokens
        best_grad_norm: Optional tensor of shape (batch_size,) with best gradient norm for each task
        best_iteration: Optional tensor of shape (batch_size,) with iteration of best grad norm for each task
    """

    accuracies: Optional[torch.Tensor] = None
    predicted_grids: Optional[torch.Tensor] = None
    optimized_output_tokens: Optional[torch.Tensor] = None
    best_grad_norm: Optional[torch.Tensor] = None
    best_iteration: Optional[torch.Tensor] = None


def parse_arc_json(file_path: Path) -> ARCTask:
    """Parse an ARC JSON file into an ARCTask dataclass.

    Args:
        file_path: Path to the JSON file

    Returns:
        ARCTask object containing train and test examples, and optional metadata
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    return ARCTask(
        train=[ARCExample(**ex) for ex in data.pop("train")],
        test=[ARCExample(**ex) for ex in data.pop("test")],
        **data,
    )


def _task_to_dict(task: ARCTask) -> Dict:
    """Convert an ARCTask to a dictionary suitable for JSON serialization.

    Args:
        task: ARCTask object to convert

    Returns:
        Dictionary with train and test examples, and optional metadata
    """
    result = asdict(task)
    # Remove None values from optional fields
    return {k: v for k, v in result.items() if v is not None}


def save_task_json(task: ARCTask, file_path: Path) -> None:
    """Save an ARCTask to a JSON file.

    Args:
        task: ARCTask object to save
        file_path: Path where the JSON file should be saved
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(_task_to_dict(task), f, indent=2)
        f.write("\n")


def download_mini_arc(data_dir: Path) -> None:
    """Download the MINI-ARC dataset from GitHub."""
    miniarc_zip_url = (
        "https://github.com/KSB21ST/MINI-ARC/archive/refs/heads/master.zip"
    )

    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(
            f"MINI-ARC dataset already exists in '{data_dir}'. Skipping download."
        )
    else:
        print(f"Downloading MINI-ARC from GitHub...")

        # Create parent directory
        parent_dir = os.path.dirname(data_dir)
        os.makedirs(parent_dir, exist_ok=True)

        # Download the zip file
        miniarc_zip_filename = os.path.join(parent_dir, "mini-arc-master.zip")
        urllib.request.urlretrieve(miniarc_zip_url, miniarc_zip_filename)

        print(f"Downloaded to: {miniarc_zip_filename}")

        # Extract the zip file
        with zipfile.ZipFile(miniarc_zip_filename, "r") as zip_ref:
            zip_ref.extractall(parent_dir)

        # Rename the extracted folder to data_dir
        extracted_folder = os.path.join(parent_dir, "MINI-ARC-master")
        os.rename(extracted_folder, data_dir)

        # Remove the zip file
        os.remove(miniarc_zip_filename)

        print(f"MINI-ARC dataset extracted to '{data_dir}' directory.")


# Transformation functions for grid augmentation
def rotate_90(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90 degrees clockwise."""
    return [list(row) for row in zip(*grid[::-1])]


def rotate_180(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 180 degrees."""
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 270 degrees clockwise (90 counter-clockwise)."""
    return [list(row) for row in zip(*grid)][::-1]


def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid horizontally."""
    return [row[::-1] for row in grid]


def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid vertically."""
    return grid[::-1]


def flip_diagonal(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid along main diagonal (top-left to bottom-right)."""
    return [list(row) for row in zip(*grid)]


def flip_antidiagonal(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid along anti-diagonal (top-right to bottom-left)."""
    return [list(row) for row in zip(*grid[::-1])][::-1]


def identity(grid: List[List[int]]) -> List[List[int]]:
    """Return grid unchanged."""
    return [row[:] for row in grid]


# All 8 transformation operations
TRANSFORMATIONS = [
    ("identity", identity),
    ("rot90", rotate_90),
    ("rot180", rotate_180),
    ("rot270", rotate_270),
    ("fliph", flip_horizontal),
    ("flipv", flip_vertical),
    ("flipd", flip_diagonal),
    ("flipa", flip_antidiagonal),
]


def get_unique_colors(grid: List[List[int]]) -> Set[int]:
    """Get all unique colors in a grid."""
    colors = set()
    for row in grid:
        colors.update(row)
    return colors


def apply_color_permutation(
    grid: List[List[int]], color_map: Dict[int, int]
) -> List[List[int]]:
    """Apply a color permutation to a grid."""
    return [[color_map.get(cell, cell) for cell in row] for row in grid]


def get_task_colors(task: ARCTask) -> Set[int]:
    """Get all unique colors used in a task (across all examples)."""
    colors = set()
    for example in task.train + task.test:
        colors.update(get_unique_colors(example.input))
        colors.update(get_unique_colors(example.output))
    return colors


def create_color_permutation(colors: List[int], rng: random.Random) -> Dict[int, int]:
    """Create a random color permutation mapping."""
    shuffled = colors[:]
    rng.shuffle(shuffled)
    return dict(zip(colors, shuffled))


def is_identity_permutation(color_map: Dict[int, int]) -> bool:
    """Check if a color permutation is the identity mapping."""
    return all(k == v for k, v in color_map.items())


def grid_to_tuple(grid: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Convert grid to hashable tuple for deduplication."""
    return tuple(tuple(row) for row in grid)


def create_augmentation_id(transform_name: str, color_map: Dict[int, int]) -> str:
    """Create a unique augmentation ID from transformation and color mapping."""
    if transform_name == "identity" and is_identity_permutation(color_map):
        return "original"

    # Create a compact representation of the color mapping
    # Sort by key to ensure consistency
    sorted_items = sorted(color_map.items())
    color_str = "_".join(f"{k}to{v}" for k, v in sorted_items if k != v)

    if not color_str:
        return transform_name
    elif transform_name == "identity":
        return f"perm_{color_str}"
    else:
        return f"{transform_name}_{color_str}"


def get_task_signature(
    train_examples: List[Tuple[List[List[int]], List[List[int]]]],
    test_examples: List[Tuple[List[List[int]], List[List[int]]]],
) -> Tuple[
    Tuple[Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]], ...],
    Tuple[Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]], ...],
]:
    """Create a hashable signature for a task to detect duplicates."""
    train_sig = tuple(
        (grid_to_tuple(inp), grid_to_tuple(out)) for inp, out in train_examples
    )
    test_sig = tuple(
        (grid_to_tuple(inp), grid_to_tuple(out)) for inp, out in test_examples
    )
    return (train_sig, test_sig)


def generate_augmentations(
    task: ARCTask,
    filename: str,
    task_colors: List[int],
    max_augmentations: int,
    rng: random.Random,
) -> List[ARCTask]:
    """Generate augmented versions of a task.

    Args:
        task: The original ARCTask
        filename: The filename (without extension) of the task
        task_colors: List of unique colors in the task
        max_augmentations: Maximum number of augmentations to generate
        rng: Random number generator

    Returns:
        List of ARCTask objects including the original and augmented versions
    """
    result: List[ARCTask] = []

    # Track seen augmentations to avoid duplicates
    seen_augmentations = set()

    # Always include the original task first
    original_task_id = f"miniarc-{filename}-original"
    original_train = [(ex.input, ex.output) for ex in task.train]
    original_test = [(ex.input, ex.output) for ex in task.test]

    # Add original to seen set
    original_signature = get_task_signature(original_train, original_test)
    seen_augmentations.add(original_signature)

    # Create original ARCTask
    original_arc_task = ARCTask(
        train=[ARCExample(input=inp, output=out) for inp, out in original_train],
        test=[ARCExample(input=inp, output=out) for inp, out in original_test],
        task_id=original_task_id,
        task_type="original",
    )
    result.append(original_arc_task)

    # Generate augmented tasks
    augmentations_generated = 0  # Count the original
    attempts = 0
    max_attempts = max_augmentations * 2  # Prevent infinite loops

    while augmentations_generated < max_augmentations and attempts < max_attempts:
        attempts += 1

        # Randomly select a transformation
        transform_name, transform_func = rng.choice(TRANSFORMATIONS)

        # Randomly create a color permutation
        color_map = create_color_permutation(task_colors, rng)

        # Skip if this is the identity operation
        if transform_name == "identity" and is_identity_permutation(color_map):
            continue

        # Apply transformation and color permutation to all examples
        aug_train = []
        for example in task.train:
            transformed_input = transform_func(example.input)
            transformed_output = transform_func(example.output)
            colored_input = apply_color_permutation(transformed_input, color_map)
            colored_output = apply_color_permutation(transformed_output, color_map)
            aug_train.append((colored_input, colored_output))

        aug_test = []
        for example in task.test:
            transformed_input = transform_func(example.input)
            transformed_output = transform_func(example.output)
            colored_input = apply_color_permutation(transformed_input, color_map)
            colored_output = apply_color_permutation(transformed_output, color_map)
            aug_test.append((colored_input, colored_output))

        # Check if this augmentation is unique
        aug_signature = get_task_signature(aug_train, aug_test)
        if aug_signature in seen_augmentations:
            continue

        seen_augmentations.add(aug_signature)

        # Create augmentation ID
        aug_id = create_augmentation_id(transform_name, color_map)
        task_id = f"miniarc-{filename}-{aug_id}"

        # Determine transformation name (None if identity)
        transformation = None if transform_name == "identity" else transform_name

        # Determine color permutation (None if identity)
        color_perm = None if is_identity_permutation(color_map) else color_map

        # Create ARCTask object
        arc_task = ARCTask(
            train=[ARCExample(input=inp, output=out) for inp, out in aug_train],
            test=[ARCExample(input=inp, output=out) for inp, out in aug_test],
            task_id=task_id,
            task_type="augmentation",
            transformation=transformation,
            color_permutation=color_perm,
        )
        result.append(arc_task)
        augmentations_generated += 1

    return result


def process_files(
    files: List[Path], output_dir: Path, max_augmentations: int, seed: int
) -> None:
    """Process a list of JSON files and write individual task JSON files.

    Args:
        files: List of JSON file paths to process
        output_dir: Path to the output directory for task JSON files
        max_augmentations: Maximum number of augmented tasks per original task
        seed: Random seed for augmentation generation
    """
    total_tasks = 0
    rng = random.Random(seed)

    # Create output directory for individual JSON files
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in files:
        try:
            # Parse the task
            task = parse_arc_json(json_file)

            # Extract filename (remove .json extension)
            filename = json_file.stem

            # Get all colors used in this task
            task_colors = sorted(get_task_colors(task))

            # Generate augmentations
            augmented_tasks = generate_augmentations(
                task, filename, task_colors, max_augmentations, rng
            )

            # Write all augmented tasks
            for arc_task in augmented_tasks:
                # Save individual task JSON file
                task_json_path = output_dir / f"{arc_task.task_id}.json"
                save_task_json(arc_task, task_json_path)

                total_tasks += 1

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    print(f"Wrote {total_tasks} task JSON files to {output_dir}")


def create_dataset(
    data_dir: Path,
    output_dir: Path,
    test_ratio: float,
    random_seed: int,
    max_augmentations: int,
) -> None:
    """Create train and test datasets from MiniARC tasks with augmentations.

    Args:
        data_dir: Path to the MiniARC data directory
        output_dir: Path to the output directory
        test_ratio: Ratio of tasks to put in test set (default: 0.2)
        random_seed: Random seed for deterministic splitting (default: 42)
        max_augmentations: Maximum number of augmented tasks per original task (default: 100)
    """
    # Create output directories
    train_output_dir = output_dir / "train"
    test_output_dir = output_dir / "test"

    # Check if both directories already exist and contain files
    if train_output_dir.exists() and test_output_dir.exists():
        train_files_exist = list(train_output_dir.glob("*.json"))
        test_files_exist = list(test_output_dir.glob("*.json"))

        if train_files_exist and test_files_exist:
            print(f"Output directories already exist and contain data:")
            print(
                f"  Train directory: {train_output_dir} ({len(train_files_exist)} files)"
            )
            print(
                f"  Test directory: {test_output_dir} ({len(test_files_exist)} files)"
            )
            print(f"Skipping dataset creation.")
            return

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Get all JSON files
    json_files = sorted(data_dir.glob("*.json"))
    print(f"Found {len(json_files)} task files")

    # Shuffle and split into train and test
    random.shuffle(json_files)
    num_test = int(len(json_files) * test_ratio)
    test_files = json_files[:num_test]
    train_files = json_files[num_test:]

    print(f"Train tasks: {len(train_files)}")
    print(f"Test tasks: {len(test_files)}")

    # Process train files
    print(f"\nProcessing train files...")
    process_files(train_files, train_output_dir, max_augmentations, random_seed)

    # Process test files
    print(f"\nProcessing test files...")
    process_files(test_files, test_output_dir, max_augmentations, random_seed + 1)

    print(f"\nDatasets created successfully!")
    print(f"Train directory: {train_output_dir}")
    print(f"Test directory: {test_output_dir}")


class ARCTaskDataset(Dataset):
    """PyTorch dataset that loads ARC tasks and returns one-hot encoded cell values.

    Each task returns a (200, vocab_size) tensor:
    - 4 examples (train + test combined)
    - Each example has input and output grids (2 grids per example)
    - Each grid is 5x5 = 25 cells
    - Total: 4 examples * 2 grids * 25 cells = 200 cells
    - Each cell is one-hot encoded with vocab_size values
    """

    def __init__(self, folder_path: str, vocab_size: int = 11):
        """Initialize the dataset.

        Args:
            folder_path: Path to folder containing task JSON files
            vocab_size: Size of vocabulary (default 11: 0-9 for colors, 10 for mask token)
        """
        self.folder_path = Path(folder_path)
        self.task_files = sorted(self.folder_path.glob("*.json"))
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.task_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a task as a (200, vocab_size) tensor of one-hot encoded cell values.

        Args:
            idx: Index of the task

        Returns:
            Tensor of shape (200, vocab_size) containing one-hot encoded cell values.
        """
        task_file = self.task_files[idx]
        task_data = parse_arc_json(task_file)

        # Collect all examples (train + test)
        all_examples = task_data.train + task_data.test

        # Handle cases with more or fewer than 4 examples
        if len(all_examples) > 4:
            # Use first 4 examples
            all_examples = all_examples[:4]
        elif len(all_examples) < 4:
            # Repeat last example until we have 4
            while len(all_examples) < 4:
                all_examples.append(all_examples[-1])

        # Collect all grids (input and output for each example)
        grids = []
        for example in all_examples:
            grids.append(example.input)
            grids.append(example.output)

        # Flatten and concatenate all grids
        all_cells = []
        for grid in grids:
            # Check that grid is exactly 5x5
            height = len(grid)
            width = len(grid[0]) if height > 0 else 0
            if height != 5 or width != 5:
                raise ValueError(
                    f"Grid must be 5x5, but got {height}x{width} in task {task_file.name}"
                )

            # Flatten the grid
            for row in grid:
                all_cells.extend(row)

        # Convert to one-hot encoding
        # all_cells should have 200 elements (4 examples * 2 grids * 25 cells)
        one_hot = torch.zeros(200, self.vocab_size, dtype=torch.float32)
        for i, cell_value in enumerate(all_cells):
            one_hot[i, cell_value] = 1.0

        return one_hot


class TransformerModel(nn.Module):
    """Non-causal transformer encoder for ARC tasks."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        seq_len: int,
        vocab_size: int,
        dropout: float,
    ):
        """Initialize the transformer model.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            seq_len: Sequence length (200 positions)
            vocab_size: Number of possible cell values (10 for ARC)
            dropout: Dropout rate
        """
        super().__init__()

        # Store vocab_size for later use
        self.vocab_size = vocab_size

        # Linear input projection (vocab_size -> d_model)
        self.input_proj = nn.Linear(vocab_size, d_model, bias=False)

        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Linear output projection (d_model -> vocab_size)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, vocab_size) - one-hot encoded
            src_key_padding_mask: Optional mask of shape (batch_size, seq_len) where True means ignore.
                                   Prevents attention to masked positions.

        Returns:
            Output tensor of shape (batch_size, 25, vocab_size) - gradient for last 25 tokens only
        """
        # Apply input projection
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)

        # Add positional embedding
        x = x + self.pos_embedding.unsqueeze(0)  # (batch_size, seq_len, d_model)

        # Apply transformer encoder with attention mask
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (batch_size, seq_len, d_model)

        # Apply output projection only to last 25 tokens
        x = self.output_proj(x[:, -25:, :])  # (batch_size, 25, vocab_size)

        # Apply hard tanh to clamp values between -1 and 1
        x = torch.nn.functional.hardtanh(x, min_val=-1.0, max_val=1.0)

        return x


def optimize_output_grid(
    model,
    x_input: torch.Tensor,
    mu: float,
    eta: float,
    num_iterations: int,
) -> DenoisingResult:
    """Optimize the output grid using gradient descent.

    Args:
        model: The transformer model to use for computing gradients
        x_input: Input tensor of shape (batch_size, 200, vocab_size)
        mu: Momentum parameter for gradient computation
        eta: Learning rate for optimization
        num_iterations: Number of optimization iterations

    Returns:
        DenoisingResult with optimized_output_tokens and best_grad_norm fields populated
    """
    batch_size = x_input.shape[0]

    with torch.no_grad():
        x = x_input.clone()
        grad = model(x)  # grad shape: (batch_size, 25, vocab_size)

        # Track best grid and gradient norm per sample in batch
        best_grad_norm = torch.full((batch_size,), float("inf"), device=x_input.device)
        best_iteration = torch.zeros((batch_size,), dtype=torch.long, device=x_input.device)
        best_x = x.clone()

        for iteration in range(num_iterations):
            x_last = x.clone()

            # Update x - only update last 25 tokens
            x[:, -25:, :] = x[:, -25:, :] - eta * grad

            # Compute gradient
            grad = model(x + mu * (x - x_last))  # grad shape: (batch_size, 25, vocab_size)

            # Calculate gradient norm per sample in batch
            grad_norm_per_sample = torch.norm(grad.view(batch_size, -1), dim=1)  # Shape: (batch_size,)

            # Update best grid for each sample if current gradient norm is lower
            improved_mask = grad_norm_per_sample < best_grad_norm  # Shape: (batch_size,)
            
            # Update best_grad_norm for improved samples
            best_grad_norm = torch.where(improved_mask, grad_norm_per_sample, best_grad_norm)
            
            # Update best_iteration for improved samples
            best_iteration = torch.where(improved_mask, torch.tensor(iteration, device=x_input.device), best_iteration)
            
            # Update best_x for improved samples
            # Expand mask to match x dimensions: (batch_size, 200, vocab_size)
            improved_mask_expanded = improved_mask.view(batch_size, 1, 1).expand_as(x)
            best_x = torch.where(improved_mask_expanded, x, best_x)

        # Use the best x (with lowest gradient norm) as final result
        x = best_x

    # Return DenoisingResult with optimized tokens and best grad norm per sample
    return DenoisingResult(
        optimized_output_tokens=x[:, -25:, :],
        best_grad_norm=best_grad_norm,
        best_iteration=best_iteration,
    )


def decode_grids(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Decode grid tokens to integer values using vectorized operations.

    Args:
        tokens: Tensor of shape (batch_size, num_tokens, vocab_size) containing grid tokens
                where num_tokens should be a multiple of 25 (for 5x5 grids)
        vocab_size: Number of possible cell values

    Returns:
        Tensor of shape (batch_size, num_grids, 5, 5) with decoded integer values
        where num_grids = num_tokens // 25
    """
    batch_size = tokens.shape[0]
    num_tokens = tokens.shape[1]

    assert (
        num_tokens % 25 == 0
    ), f"num_tokens must be a multiple of 25, got {num_tokens}"

    num_grids = num_tokens // 25

    # Take argmax of all dimensions for all tokens at once
    # Shape: (batch_size, num_tokens)
    decoded_values = torch.argmax(tokens[:, :, :vocab_size], dim=2)
    
    # Reshape to (batch_size, num_grids, 5, 5)
    decoded_grids = decoded_values.view(batch_size, num_grids, 5, 5)
    
    return decoded_grids


def evaluate_denoising_accuracy(
    model,
    x_clean: torch.Tensor,
    vocab_size: int,
    mu: float,
    eta: float,
    num_iterations: int,
) -> DenoisingResult:
    """Evaluate denoising accuracy by corrupting and denoising output grids.

    Args:
        model: The transformer model to use for computing gradients
        x_clean: Clean input tensor of shape (batch_size, 200, vocab_size)
        vocab_size: Number of possible cell values
        mu: Momentum parameter for gradient computation
        eta: Learning rate for optimization
        num_iterations: Number of optimization iterations

    Returns:
        DenoisingResult containing accuracies and predicted grids
    """

    # Create noised input by noising last 25 tokens
    x_i = noise_last_25_tokens(x_clean, x_clean.device)

    # Perform optimization to denoise
    opt_result = optimize_output_grid(
        model=model,
        x_input=x_i,
        mu=mu,
        eta=eta,
        num_iterations=num_iterations,
    )

    # Decode the optimized output grids
    assert opt_result.optimized_output_tokens is not None
    predicted_grids = decode_grids(
        opt_result.optimized_output_tokens, vocab_size
    )  # Shape: (batch_size, 1, 5, 5)
    predicted_grids = predicted_grids[:, 0, :, :]  # Shape: (batch_size, 5, 5)

    # Decode the true output grids (last 25 tokens)
    true_output_tokens = x_clean[:, -25:, :]  # Shape: (batch_size, 25, vocab_size)
    true_grids = decode_grids(true_output_tokens, vocab_size)  # Shape: (batch_size, 1, 5, 5)
    true_grids = true_grids[:, 0, :, :]  # Shape: (batch_size, 5, 5)

    # Calculate accuracy for each task in the batch (vectorized)
    # Shape: (batch_size,) - mean over the 5x5 grid for each batch element
    accuracies = (predicted_grids == true_grids).float().mean(dim=(1, 2))

    # Use replace to add accuracies and predicted_grids to the result
    return replace(
        opt_result,
        accuracies=accuracies,
        predicted_grids=predicted_grids,
    )


def random_shift_examples(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Randomly shift the order of examples in each task while keeping input/output pairs together.
    
    Each task has 4 examples, with each example consisting of an input grid (25 cells) 
    followed by an output grid (25 cells), for a total of 50 cells per example.
    This function shifts by 2, 4, or 6 slots (i.e., 50, 100, or 150 cells), 
    rotating examples that go past the end back to the beginning.
    
    Args:
        batch: Batch tensor of shape (batch_size, 200, vocab_size)
        device: Device to perform operations on
    
    Returns:
        Shifted batch tensor of same shape
    """
    batch_size, seq_len, vocab_size = batch.shape
    
    # Reshape to (batch_size, 4 examples, 50 cells per example, vocab_size)
    # Each example has 50 cells: input grid (25 cells) + output grid (25 cells)
    batch_reshaped = batch.view(batch_size, 4, 50, vocab_size)
    
    # Randomly choose shift amount for each item in batch (0, 1, 2, or 3 example shifts)
    # We use 0-3 because we have 4 examples. Shifting by 0 means no shift.
    # This corresponds to shifts of 0, 2, 4, or 6 slots (0, 100, 200, or 300 cells)
    shift_amounts = torch.randint(0, 4, (batch_size,), device=device)
    
    # Create shifted batch using torch.roll for each batch item
    # We need to handle each batch item separately because they have different shift amounts
    shifted_batch = torch.zeros_like(batch_reshaped)
    for i in range(batch_size):
        shifted_batch[i] = torch.roll(batch_reshaped[i], shifts=int(shift_amounts[i].item()), dims=0)
    
    # Reshape back to original shape
    return shifted_batch.view(batch_size, seq_len, vocab_size)


def random_shift_last_three_examples(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Randomly shift the order of the last 3 examples while keeping the first example unchanged.
    
    Each task has 4 examples. This function only shifts the last 3 examples (last 150 tokens),
    leaving the first example (first 50 tokens) unchanged.
    
    Args:
        batch: Batch tensor of shape (batch_size, 200, vocab_size)
        device: Device to perform operations on
    
    Returns:
        Shifted batch tensor of same shape
    """
    batch_size, seq_len, vocab_size = batch.shape
    
    # Split into first 50 tokens (1 example) and last 150 tokens (3 examples)
    first_one = batch[:, :50, :]      # (batch_size, 50, vocab_size)
    last_three = batch[:, 50:, :]     # (batch_size, 150, vocab_size)
    
    # Reshape last 3 examples to (batch_size, 3 examples, 50 cells per example, vocab_size)
    last_three_reshaped = last_three.view(batch_size, 3, 50, vocab_size)
    
    # Randomly choose shift amount for each item in batch (0, 1, or 2 example shifts)
    # We use 0-2 because we have 3 examples. Shifting by 0 means no shift.
    shift_amounts = torch.randint(0, 3, (batch_size,), device=device)
    
    # Create shifted batch using torch.roll for each batch item
    shifted_last_three = torch.zeros_like(last_three_reshaped)
    for i in range(batch_size):
        shifted_last_three[i] = torch.roll(last_three_reshaped[i], shifts=int(shift_amounts[i].item()), dims=0)
    
    # Reshape last three back to (batch_size, 150, vocab_size)
    shifted_last_three = shifted_last_three.view(batch_size, 150, vocab_size)
    
    # Concatenate unchanged first example with shifted last 3 examples
    return torch.cat([first_one, shifted_last_three], dim=1)


def noise_last_25_tokens(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Noise the last 25 tokens by randomly replacing 0-25 of them with random values.
    
    Args:
        batch: Batch tensor of shape (batch_size, 200, vocab_size)
        device: Device to perform operations on
    
    Returns:
        Noised batch tensor of same shape
    """
    batch_size = batch.shape[0]
    noised_batch = batch.clone()
    
    # Sample number of tokens to randomly replace in last 25 tokens (0-25) for each batch item
    num_random_tokens = torch.randint(0, 26, (batch_size,), device=device)  # 0-25

    # For each batch item, select random positions in last 25 tokens and replace with random values
    for i in range(batch_size):
        n_random = num_random_tokens[i].item()
        
        if n_random > 0:
            # Select random positions in last 25 tokens (indices 175-199)
            positions = torch.randperm(25, device=device)[:n_random] + 175
            
            # Replace with random tokens (0-10, where 10 is mask token)
            random_tokens = torch.randint(0, 11, (int(n_random),), device=device)
            noised_batch[i, positions, :] = 0
            noised_batch[i, positions, random_tokens] = 1
    
    return noised_batch


def compute_loss_for_batch(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute loss for a single batch.

    Args:
        model: The transformer model
        batch: Batch of one-hot encoded cell values, shape (batch_size, 200, vocab_size)
        device: Device to compute on
        attention_mask: Optional attention mask of shape (batch_size, 200) where True means ignore

    Returns:
        Loss tensor (scalar)
    """
    x = batch.to(device)  # (batch_size, 200, vocab_size)

    # Create noisy input by noising last 25 tokens
    xg = noise_last_25_tokens(x, device)

    # Create target as difference for last 25 tokens only
    target = xg[:, -25:, :] - x[:, -25:, :]  # (batch_size, 25, vocab_size)

    # Forward pass - model now outputs (batch_size, 25, vocab_size)
    output = model(xg, src_key_padding_mask=attention_mask)  # (batch_size, 25, vocab_size)

    # Compute loss
    loss = ((output - target) ** 2).mean()

    return loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The transformer model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        # Randomly shift examples in each task
        batch = random_shift_examples(batch, device)
        
        # For ~50% of samples, mask the first 50 tokens
        batch_size = batch.shape[0]
        mask_first_50 = torch.rand(batch_size, device=device) < 0.5
        
        # Create attention mask (True means ignore)
        attention_mask = torch.zeros(batch_size, 200, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            if mask_first_50[i]:
                batch[i, :50, :] = 0
                batch[i, :50, 10] = 1
                # Prevent attention to first 50 tokens when they are masked
                attention_mask[i, :50] = True
        
        # Compute loss
        loss = compute_loss_for_batch(model, batch, device, attention_mask=attention_mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def test_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Evaluate on test set.

    Args:
        model: The transformer model
        test_loader: Test data loader
        optimizer: Optimizer for training on last 25 tokens
        device: Device to evaluate on

    Returns:
        Average test loss
    """
    # Training section: train on first three examples with first 50 tokens masked
    model.train()
    for batch in test_loader:
        batch_masked = batch.clone()
        
        # Copy first 150 tokens to last 150 tokens
        batch_masked[:, 50:, :] = batch[:, :150, :]
        
        # Mask the first 50 tokens
        batch_masked[:, :50, :] = 0
        batch_masked[:, :50, 10] = 1
        
        # Randomly shift last 3 examples (positions 50-199)
        batch_masked = random_shift_last_three_examples(batch_masked, device)
        
        # Create attention mask to prevent attention to first 50 tokens
        batch_size = batch_masked.shape[0]
        attention_mask = torch.zeros(batch_size, 200, dtype=torch.bool, device=device)
        attention_mask[:, :50] = True  # Ignore first 50 tokens
        
        # Compute loss
        loss = compute_loss_for_batch(model, batch_masked, device, attention_mask=attention_mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation section
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            # Compute loss
            loss = compute_loss_for_batch(model, batch, device)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def learning_rate_test(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    weight_decay: float,
):
    """Test learning rate by starting at 1e-7 and doubling every batch.

    Args:
        model: The transformer model
        train_loader: Training data loader
        device: Device to train on
    """
    # Start learning rate test
    print("\nStarting learning rate test...")
    lr = 1e-7
    model.train()

    batch_count = 0
    for batch in train_loader:
        if batch_count >= 20:
            break

        # Create optimizer with current learning rate
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Randomly shift examples in each task
        batch = random_shift_examples(batch, device)

        # Compute loss
        loss = compute_loss_for_batch(model, batch, device)

        # Backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Print learning rate and loss
        print(f"Batch {batch_count + 1}: LR = {lr:.2e}, Loss = {loss.item():.6f}")

        # Double learning rate for next batch
        lr *= 2
        batch_count += 1

    print("\nLearning rate test complete!")


def evaluate_denoising(
    model: nn.Module,
    train_dataset: ARCTaskDataset,
    test_dataset: ARCTaskDataset,
    device: torch.device,
    vocab_size: int,
    eval_denoise_mu: float,
    eval_denoise_eta: float,
    eval_denoise_num_iterations: int,
    writer: Optional[SummaryWriter] = None,
    epoch: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate denoising accuracy on original tasks.

    Args:
        model: The transformer model
        train_dataset: Training dataset
        test_dataset: Test dataset
        device: Device to evaluate on
        vocab_size: Number of possible cell values
        eval_denoise_mu: Momentum parameter for gradient computation
        eval_denoise_eta: Learning rate for optimization
        eval_denoise_num_iterations: Number of optimization iterations
        writer: Optional tensorboard writer for logging
        epoch: Optional epoch number for logging

    Returns:
        Tuple of (average_train_accuracy, average_test_accuracy)
    """
    # Filter for original tasks in train dataset
    train_original_tasks = [
        (idx, file_path)
        for idx, file_path in enumerate(train_dataset.task_files)
        if file_path.name.endswith("original.json")
    ]
    assert len(train_original_tasks) > 10 and len(train_original_tasks) < 200, "Expected between 10 and 200 original tasks in train dataset"

    # Filter for original tasks in test dataset
    test_original_tasks = [
        (idx, file_path)
        for idx, file_path in enumerate(test_dataset.task_files)
        if file_path.name.endswith("original.json")
    ]
    assert len(test_original_tasks) > 10 and len(test_original_tasks) < 50, "Expected between 10 and 50 original tasks in test dataset"

    # Evaluate denoising accuracy
    eval_start_time = time.time()

    # Evaluate train tasks in batch
    model.eval()
    with torch.no_grad():
        # Load all train tasks into a batch
        train_batch = torch.stack([
            train_dataset[task_idx].to(device)
            for task_idx, _ in train_original_tasks
        ])

        train_result = evaluate_denoising_accuracy(
            model=model,
            x_clean=train_batch,
            vocab_size=vocab_size,
            mu=eval_denoise_mu,
            eta=eval_denoise_eta,
            num_iterations=eval_denoise_num_iterations,
        )

        assert train_result.accuracies is not None
        train_accuracies = train_result.accuracies.cpu().numpy()

        # Load all test tasks into a batch
        test_batch = torch.stack([
            test_dataset[task_idx].to(device)
            for task_idx, _ in test_original_tasks
        ])

        test_result = evaluate_denoising_accuracy(
            model=model,
            x_clean=test_batch,
            vocab_size=vocab_size,
            mu=eval_denoise_mu,
            eta=eval_denoise_eta,
            num_iterations=eval_denoise_num_iterations,
        )

        assert test_result.accuracies is not None
        test_accuracies = test_result.accuracies.cpu().numpy()

    # Compute average accuracies
    avg_train_acc = np.mean(train_accuracies) if len(train_accuracies) > 0 else 0.0
    avg_test_acc = np.mean(test_accuracies) if len(test_accuracies) > 0 else 0.0

    # Compute % of grids with 100% accuracy
    train_perfect_pct = (np.sum(train_accuracies == 1.0) / len(train_accuracies) * 100) if len(train_accuracies) > 0 else 0.0
    test_perfect_pct = (np.sum(test_accuracies == 1.0) / len(test_accuracies) * 100) if len(test_accuracies) > 0 else 0.0

    # Get max iteration across all samples
    assert train_result.best_iteration is not None
    assert test_result.best_iteration is not None
    max_train_iter = train_result.best_iteration.max().item()
    max_test_iter = test_result.best_iteration.max().item()

    # Compute average and std of best iteration
    train_best_iterations = train_result.best_iteration.cpu().numpy()
    test_best_iterations = test_result.best_iteration.cpu().numpy()
    avg_train_iter = np.mean(train_best_iterations) if len(train_best_iterations) > 0 else 0.0
    std_train_iter = np.std(train_best_iterations) if len(train_best_iterations) > 0 else 0.0
    avg_test_iter = np.mean(test_best_iterations) if len(test_best_iterations) > 0 else 0.0
    std_test_iter = np.std(test_best_iterations) if len(test_best_iterations) > 0 else 0.0

    # Calculate evaluation time
    eval_time = time.time() - eval_start_time

    # Print to terminal
    if epoch is not None:
        print(
            f"  Train Accuracy: {avg_train_acc * 100:.2f}% (100% acc: {train_perfect_pct:.1f}%), "
            f"Test Accuracy: {avg_test_acc * 100:.2f}% (100% acc: {test_perfect_pct:.1f}%)"
        )
        print(
            f"  Train Best Iter: {avg_train_iter:.1f}±{std_train_iter:.1f} (max: {max_train_iter}), "
            f"Test Best Iter: {avg_test_iter:.1f}±{std_test_iter:.1f} (max: {max_test_iter}), "
            f"Time: {eval_time:.2f}s"
        )
    else:
        print(
            f"Train Accuracy: {avg_train_acc * 100:.2f}% (100% acc: {train_perfect_pct:.1f}%), "
            f"Test Accuracy: {avg_test_acc * 100:.2f}% (100% acc: {test_perfect_pct:.1f}%)"
        )
        print(
            f"Train Best Iter: {avg_train_iter:.1f}±{std_train_iter:.1f} (max: {max_train_iter}), "
            f"Test Best Iter: {avg_test_iter:.1f}±{std_test_iter:.1f} (max: {max_test_iter}), "
            f"Time: {eval_time:.2f}s"
        )

    # Log to tensorboard if writer provided
    if writer is not None and epoch is not None:
        writer.add_scalar(
            f"DenoiseAccuracy/train", avg_train_acc, epoch
        )
        writer.add_scalar(
            f"DenoiseAccuracy/test", avg_test_acc, epoch
        )
        writer.add_scalar(
            f"DenoisePerfect/train", train_perfect_pct, epoch
        )
        writer.add_scalar(
            f"DenoisePerfect/test", test_perfect_pct, epoch
        )
        writer.add_scalar(
            f"DenoiseBestIter/train_mean", avg_train_iter, epoch
        )
        writer.add_scalar(
            f"DenoiseBestIter/train_std", std_train_iter, epoch
        )
        writer.add_scalar(
            f"DenoiseBestIter/test_mean", avg_test_iter, epoch
        )
        writer.add_scalar(
            f"DenoiseBestIter/test_std", std_test_iter, epoch
        )

    return float(avg_train_acc), float(avg_test_acc)


def train(config: Config):
    """Train transformer model on ARC tasks.

    Args:
        config: Configuration object containing all training parameters
    """

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision("high")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = ARCTaskDataset(config.train_data_dir, vocab_size=config.vocab_size)
    test_dataset = ARCTaskDataset(config.test_data_dir, vocab_size=config.vocab_size)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Create model
    model = TransformerModel(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
    ).to(device)

    # Compile model for better performance (PyTorch 2.0+)
    print("Compiling model with torch.compile...")
    model = cast(TransformerModel, torch.compile(model))

    # Count parameters
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {model_params:,} trainable parameters")

    # Check if running learning rate test
    if config.mode == "learning_rate_test":
        learning_rate_test(model, train_loader, device, config.weight_decay)
        return

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Load existing model if specified
    start_epoch = 0
    if config.load_model_path:
        if Path(config.load_model_path).exists():
            print(f"\nLoading existing model from {config.load_model_path}")
            checkpoint = torch.load(config.load_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            print(f"Resumed from epoch {start_epoch}")
            print(f"Previous train loss: {checkpoint.get('train_loss', 'N/A')}")
            print(f"Previous test loss: {checkpoint.get('test_loss', 'N/A')}")
        else:
            print(
                f"\nWarning: Model path {config.load_model_path} does not exist. Starting from scratch."
            )

    # Check if running evaluation mode (after model loading)
    if config.mode == "eval":
        print("\nRunning evaluation mode...")
        evaluate_denoising(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            vocab_size=config.vocab_size,
            eval_denoise_mu=config.eval_denoise_mu,
            eval_denoise_eta=config.eval_denoise_eta,
            eval_denoise_num_iterations=config.eval_denoise_num_iterations,
        )
        return

    # Create tensorboard writer
    Path(config.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Test
        test_loss = test_epoch(model, test_loader, optimizer, device)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Log to console
        print(
            f"Epoch {epoch + 1}/{start_epoch + config.num_epochs} - "
            f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)

        # Evaluate denoising accuracy periodically
        if (epoch + 1) % config.eval_denoise_epoch_interval == 0:
            print(f"\nEvaluating denoising accuracy at epoch {epoch + 1}...")

            evaluate_denoising(
                model=model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                device=device,
                vocab_size=config.vocab_size,
                eval_denoise_mu=config.eval_denoise_mu,
                eval_denoise_eta=config.eval_denoise_eta,
                eval_denoise_num_iterations=config.eval_denoise_num_iterations,
                writer=writer,
                epoch=epoch,
            )

            print()  # Empty line for readability

        # Save checkpoint every N epochs (configurable)
        if (epoch + 1) % config.checkpoint_save_interval == 0:
            checkpoint_path = f"{config.checkpoint_dir}/{config.timestamp}_epoch_{epoch + 1}_checkpoint.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "config": {
                        "d_model": config.d_model,
                        "nhead": config.nhead,
                        "num_layers": config.num_layers,
                        "dim_feedforward": config.dim_feedforward,
                        "seq_len": config.seq_len,
                        "vocab_size": config.vocab_size,
                        "dropout": config.dropout,
                    },
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Copy checkpoint to Google Drive if the directory exists
            if os.path.exists(config.google_drive_dir):
                gdrive_checkpoint_path = f"{config.google_drive_dir}/{config.timestamp}_epoch_{epoch + 1}_checkpoint.pt"
                shutil.copy2(checkpoint_path, gdrive_checkpoint_path)
                print(f"Copied checkpoint to Google Drive: {gdrive_checkpoint_path}")

    writer.close()
    print("\nTraining complete!")

    # Save model
    Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "d_model": config.d_model,
                "nhead": config.nhead,
                "num_layers": config.num_layers,
                "dim_feedforward": config.dim_feedforward,
                "seq_len": config.seq_len,
                "vocab_size": config.vocab_size,
                "dropout": config.dropout,
            },
        },
        config.model_save_path,
    )
    print(f"Model saved to {config.model_save_path}")


def main():
    # Create configuration
    config = Config(
        # Dataset creation parameters
        data_dir=Path("data/MINI-ARC"),
        output_dir=Path("output/mini_arc_eqm5"),
        test_ratio=0.2,
        random_seed=42,
        max_augmentations=500,
        # Model parameters
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=1024,
        dropout=0.1,
        # Data parameters
        seq_len=200,
        vocab_size=11,
        # Denoising evaluation parameters
        eval_denoise_epoch_interval=1,
        eval_denoise_mu=0,
        eval_denoise_eta=1,
        eval_denoise_num_iterations=500,
        # Training parameters
        num_epochs=300,
        batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.1,
        mode="train",
        checkpoint_save_interval=30,
        # Google Drive location for Colab
        google_drive_dir="/content/drive/MyDrive/sparse_arc",
        # Optional: Load existing model to continue training
        load_model_path=None,
    )

    # Print configuration
    print("Configuration:")
    pprint.pprint(asdict(config), width=100, sort_dicts=False)
    print()

    # Download dataset
    download_mini_arc(config.data_dir)

    # Create train/test split with augmentations
    create_dataset(
        data_dir=config.data_dir / "data" / "MiniARC",
        output_dir=config.output_dir,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        max_augmentations=config.max_augmentations,
    )

    # # Train model
    train(config)

    # Shut down Google Colab runtime if running in Colab
    try:
        from google.colab import runtime # type: ignore
        print("\nShutting down Google Colab runtime...")
        runtime.unassign()
    except ImportError:
        # Not running in Colab, skip shutdown
        pass


if __name__ == "__main__":
    main()
