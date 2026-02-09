import json
import os
import pprint
import random
import shutil
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, TypedDict, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
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
    embedding_dim: int  # Output embedding dimension (e.g., 512)

    # Training parameters
    jepa_batch_size: int
    pred_batch_size: int
    num_epochs: int
    learning_rate: float
    lambd: float  # Weight for loss_sig_reg in loss calculation
    mode: Literal["train", "learning_rate_test"]
    checkpoint_save_interval: int

    # Data parameters
    vocab_size: int

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


class ARCTaskData(TypedDict):
    """Typed dict for a task with lists of grids."""
    train_input_grids: List[torch.Tensor]  # List of (25,) tensors
    train_output_grids: List[torch.Tensor]  # List of (25,) tensors
    test_input_grid: torch.Tensor  # Single (25,) tensor
    test_output_grid: torch.Tensor  # Single (25,) tensor


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
    print(f"Found {len(json_files)} total task files")

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
    """PyTorch dataset that loads ARC tasks and returns list of example dicts.

    Each task returns a list of dicts, where each dict contains:
    - input_grid: (25,) token values for the flattened input grid
    - output_grid: (25,) token values for the flattened output grid
    - example_type: 'train' or 'test' indicating which set the example came from
    """

    def __init__(self, folder_path: str, vocab_size: int = 10):
        """Initialize the dataset.

        Args:
            folder_path: Path to folder containing task JSON files
            vocab_size: Size of vocabulary (default 10: 0-9 for colors)
        """
        self.folder_path = Path(folder_path)
        self.task_files = sorted(self.folder_path.glob("*.json"))
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.task_files)

    def __getitem__(self, idx: int) -> ARCTaskData:
        """Get a task as a dict with train/test grids.

        Args:
            idx: Index of the task

        Returns:
            ArcTaskData dict with train_input_grids, train_output_grids, test_input_grids, test_output_grids.
            Each value is a list of tensors (25,) with token values.
        """
        task_file = self.task_files[idx]
        task_data = parse_arc_json(task_file)
        
        train_input_grids = []
        train_output_grids = []
        
        # Assert there is exactly one test example
        assert len(task_data.test) == 1, f"Expected 1 test example, got {len(task_data.test)} in task {task_file.name}"
        
        # Process train examples
        for example in task_data.train:
                # Check that grids are exactly 5x5
                input_height = len(example.input)
                input_width = len(example.input[0]) if input_height > 0 else 0
                output_height = len(example.output)
                output_width = len(example.output[0]) if output_height > 0 else 0
                
                if input_height != 5 or input_width != 5:
                    raise ValueError(
                        f"Input grid must be 5x5, but got {input_height}x{input_width} in task {task_file.name}"
                    )
                if output_height != 5 or output_width != 5:
                    raise ValueError(
                        f"Output grid must be 5x5, but got {output_height}x{output_width} in task {task_file.name}"
                    )

                # Flatten input grid into token sequence
                input_cells = []
                for row in example.input:
                    input_cells.extend(row)
                
                # Flatten output grid into token sequence
                output_cells = []
                for row in example.output:
                    output_cells.extend(row)

                # Convert to tensors of token values
                train_input_grids.append(torch.tensor(input_cells, dtype=torch.long))
                train_output_grids.append(torch.tensor(output_cells, dtype=torch.long))
        
        # Process the single test example
        test_example = task_data.test[0]
        
        # Check that grids are exactly 5x5
        input_height = len(test_example.input)
        input_width = len(test_example.input[0]) if input_height > 0 else 0
        output_height = len(test_example.output)
        output_width = len(test_example.output[0]) if output_height > 0 else 0
        
        if input_height != 5 or input_width != 5:
            raise ValueError(
                f"Input grid must be 5x5, but got {input_height}x{input_width} in task {task_file.name}"
            )
        if output_height != 5 or output_width != 5:
            raise ValueError(
                f"Output grid must be 5x5, but got {output_height}x{output_width} in task {task_file.name}"
            )

        # Flatten input grid into token sequence
        test_input_cells = []
        for row in test_example.input:
            test_input_cells.extend(row)
        
        # Flatten output grid into token sequence
        test_output_cells = []
        for row in test_example.output:
            test_output_cells.extend(row)

        return {
            "train_input_grids": train_input_grids,
            "train_output_grids": train_output_grids,
            "test_input_grid": torch.tensor(test_input_cells, dtype=torch.long),
            "test_output_grid": torch.tensor(test_output_cells, dtype=torch.long),
        }


def arc_task_collate_fn(batch: List[ARCTaskData]) -> List[ARCTaskData]:
    """Custom collate function for batching ARCTaskData dictionaries.
    
    Since each task has variable-length lists, we don't stack them.
    Just return the batch as a list of dictionaries.
    
    Args:
        batch: List of ARCTaskData dictionaries
    
    Returns:
        The batch as-is (list of dictionaries)
    """
    return batch


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for 5x5 grids.
    
    Applies rotary embeddings based on 2D positions (row, col) in a 5x5 grid.
    Works with sequences of 25 tokens representing grid positions (0,0) to (4,4).
    """
    
    inv_freq: torch.Tensor
    sin_cached: torch.Tensor
    cos_cached: torch.Tensor

    def __init__(self, d_model: int, max_grid_size: int = 5):
        """Initialize 2D RoPE.
        
        Args:
            d_model: Model dimension (must be divisible by 4 for 2D RoPE)
            max_grid_size: Maximum grid size (default 5 for 5x5 grids)
        """
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D RoPE"
        
        self.d_model = d_model
        self.max_grid_size = max_grid_size
        
        # Split dimensions: half for x-axis, half for y-axis
        self.d_rope = d_model // 2  # Dimension per axis
        
        # Compute frequency bands
        # For each axis, use d_rope/2 frequency bands
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_rope, 2).float() / self.d_rope))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position encodings for 25 grid positions (row-major order)
        self._cache_rope_embeddings()
    
    def _cache_rope_embeddings(self):
        """Precompute RoPE embeddings for 25 grid positions.
        
        Positions 0-24: grid cells in row-major order, positions (0,0) to (4,4)
        """
        positions = []
        
        # Positions 0-24: grid positions in row-major order
        for row in range(self.max_grid_size):
            for col in range(self.max_grid_size):
                positions.append((row, col))
        
        # Convert to tensors
        rows = torch.tensor([p[0] for p in positions], dtype=torch.float32)
        cols = torch.tensor([p[1] for p in positions], dtype=torch.float32)
        
        # Compute sin/cos for each axis
        # Shape: (seq_len, d_rope/2)
        row_emb = torch.outer(rows, self.inv_freq)
        col_emb = torch.outer(cols, self.inv_freq)
        
        # Combine: (seq_len, d_rope/2) -> (seq_len, d_rope) for each axis
        row_sin = torch.sin(row_emb)
        row_cos = torch.cos(row_emb)
        col_sin = torch.sin(col_emb)
        col_cos = torch.cos(col_emb)
        
        # Stack to create full embedding: (seq_len, d_model)
        # Interleave sin/cos for each dimension
        sin_emb = torch.zeros(len(positions), self.d_model)
        cos_emb = torch.zeros(len(positions), self.d_model)
        
        # First half: row embeddings
        for i in range(self.d_rope // 2):
            sin_emb[:, 4*i] = row_sin[:, i]
            sin_emb[:, 4*i + 1] = row_sin[:, i]
            cos_emb[:, 4*i] = row_cos[:, i]
            cos_emb[:, 4*i + 1] = row_cos[:, i]
        
        # Second half: column embeddings
        for i in range(self.d_rope // 2):
            sin_emb[:, 4*i + 2] = col_sin[:, i]
            sin_emb[:, 4*i + 3] = col_sin[:, i]
            cos_emb[:, 4*i + 2] = col_cos[:, i]
            cos_emb[:, 4*i + 3] = col_cos[:, i]
        
        self.register_buffer('sin_cached', sin_emb)
        self.register_buffer('cos_cached', cos_emb)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings to a 5x5 grid.
        
        Args:
            x: Input tensor of shape (batch_size, 25, d_model) representing a flattened 5x5 grid
        
        Returns:
            Tensor with RoPE applied, same shape as input
        """
        assert x.shape[1] == 25, f"Expected sequence length 25 for 5x5 grid, got {x.shape[1]}"
        
        # Reshape for rotation: (batch_size, 25, d_model/2, 2)
        x_rope = x.reshape(x.shape[0], 25, -1, 2)
        
        # Get sin/cos: (25, d_model)
        sin = self.sin_cached.unsqueeze(0)  # (1, 25, d_model)
        cos = self.cos_cached.unsqueeze(0)  # (1, 25, d_model)
        
        # Reshape sin/cos: (1, 25, d_model/2, 2)
        sin = sin.reshape(1, 25, -1, 2)
        cos = cos.reshape(1, 25, -1, 2)
        
        # Apply rotation: x' = x * cos + rotate_half(x) * sin
        # rotate_half swaps the two elements and negates the first
        x1, x2 = x_rope[..., 0], x_rope[..., 1]
        
        # Rotation formula
        x_rotated = torch.stack([
            x1 * cos[..., 0] - x2 * sin[..., 0],
            x2 * cos[..., 1] + x1 * sin[..., 1]
        ], dim=-1)
        
        # Reshape back to (batch_size, 25, d_model)
        return x_rotated.reshape(x.shape[0], 25, self.d_model)


class JepaModel(nn.Module):
    """Non-causal transformer encoder for ARC tasks (JEPA model)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        vocab_size: int,
        dropout: float,
        embedding_dim: int,
    ):
        """Initialize the transformer model.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            vocab_size: Number of possible cell values (10 for ARC: 0-9 colors)
            dropout: Dropout rate
            embedding_dim: Output embedding dimension (e.g., 512)
        """
        super().__init__()

        # Store vocab_size for later use
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        # Token embedding layer (vocab_size -> d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learnable position embeddings for grid types
        self.input_grid_embedding = nn.Parameter(torch.randn(d_model))
        self.output_grid_embedding = nn.Parameter(torch.randn(d_model))

        # Learnable example embeddings (3 examples per global view)
        self.example_embeddings = nn.Parameter(torch.randn(3, d_model))

        # 2D Rotary Position Embeddings
        self.rope = RoPE2D(d_model=d_model, max_grid_size=5)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Linear projection from d_model to embedding_dim (applied per token)
        self.output_proj = nn.Linear(d_model, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 150) - tokens for 3 examples (3 * 50 tokens)

        Returns:
            Tensor of shape (batch_size, embedding_dim) with embeddings
        """
        # Apply token embedding to grid tokens
        grid_embeddings = self.token_embedding(x)  # (batch_size, 150, d_model)
        
        batch_size = grid_embeddings.shape[0]
        
        # Process each of the 3 examples (150 tokens = 3 examples * 50 tokens/example)
        processed_grids = []
        for ex_idx in range(3):
            start_idx = ex_idx * 50
            # Split into input and output grids for this example
            input_grid = grid_embeddings[:, start_idx:start_idx + 25, :]  # (batch_size, 25, d_model)
            output_grid = grid_embeddings[:, start_idx + 25:start_idx + 50, :]  # (batch_size, 25, d_model)
            
            # Add grid type embeddings
            input_grid = input_grid + self.input_grid_embedding
            output_grid = output_grid + self.output_grid_embedding
            
            # Add example embeddings
            input_grid = input_grid + self.example_embeddings[ex_idx]
            output_grid = output_grid + self.example_embeddings[ex_idx]
            
            # Apply 2D rotary position embeddings separately to each grid
            input_grid = self.rope(input_grid)
            output_grid = self.rope(output_grid)
            
            # Concatenate input and output for this example
            example_grid = torch.cat([input_grid, output_grid], dim=1)  # (batch_size, 50, d_model)
            processed_grids.append(example_grid)
        
        # Concatenate all 3 examples
        x = torch.cat(processed_grids, dim=1)  # (batch_size, 150, d_model)

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 150, d_model)

        # Project each token to embedding_dim
        x = self.output_proj(x)  # (batch_size, 150, embedding_dim)
        
        # Mean pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, embedding_dim)
        
        return x


class PredictionModel(nn.Module):
    """Transformer model that takes JEPA embedding and input grid to predict output grid."""

    def __init__(
        self,
        jepa_embedding_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        vocab_size: int,
        dropout: float,
    ):
        """Initialize the prediction model.

        Args:
            jepa_embedding_dim: Dimension of JEPA embedding input
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            vocab_size: Number of possible cell values (10 for ARC: 0-9 colors)
            dropout: Dropout rate
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.jepa_embedding_dim = jepa_embedding_dim
        
        # Calculate number of JEPA tokens
        assert jepa_embedding_dim % d_model == 0, f"jepa_embedding_dim ({jepa_embedding_dim}) must be divisible by d_model ({d_model})"
        self.num_jepa_tokens = jepa_embedding_dim // d_model

        # Token embedding layer for grid tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2D Rotary Position Embeddings for grid positions
        self.rope = RoPE2D(d_model=d_model, max_grid_size=5)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection to vocab_size
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        jepa_embedding: torch.Tensor,
        input_grid: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            jepa_embedding: Tensor of shape (batch_size, jepa_embedding_dim)
            input_grid: Tensor of shape (batch_size, 25) with token values

        Returns:
            Tensor of shape (batch_size, 25, vocab_size) with output grid logits
        """
        batch_size = input_grid.shape[0]
        
        # Reshape JEPA embedding to tokens
        jepa_tokens = jepa_embedding.view(batch_size, self.num_jepa_tokens, self.d_model)  # (batch_size, num_jepa_tokens, d_model)
        
        # Embed input grid
        input_emb = self.token_embedding(input_grid)  # (batch_size, 25, d_model)
        
        # Apply RoPE to input grid
        input_emb = self.rope(input_emb)  # (batch_size, 25, d_model)
        
        # Concatenate JEPA tokens and input tokens
        x = torch.cat([jepa_tokens, input_emb], dim=1)  # (batch_size, num_jepa_tokens + 25, d_model)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, num_jepa_tokens + 25, d_model)
        
        # Extract output positions (last 25 tokens corresponding to the grid)
        x = x[:, self.num_jepa_tokens:, :]  # (batch_size, 25, d_model)
        
        # Project to logits
        logits = self.output_proj(x)  # (batch_size, 25, vocab_size)
        
        return logits


def sig_reg(x: torch.Tensor, num_slices: int = 256) -> torch.Tensor:
    """Sliced Independence Generator Regularization (SIG-Reg).
    
    Computes the Epps-Pulley test statistic to measure deviation from standard normal.
    
    Args:
        x: Input tensor of shape (batch_size, embedding_dim)
        num_slices: Number of random projections (default 256)
    
    Returns:
        Test statistic tensor of shape (num_slices,) representing deviation from normality
    """
    device = x.device
    
    # Slice sampling with random projections
    proj_shape = (x.size(1), num_slices)
    A = torch.randn(proj_shape, device=device)
    A = A / A.norm(p=2, dim=0)
    
    # Epps-Pulley statistic - see Sec. 4.3 for alternatives
    # Integration points
    t = torch.linspace(-5, 5, 17, device=device)
    
    # Theoretical Characteristic Function (CF) for N(0, 1) with Gauss window
    exp_f = torch.exp(-0.5 * t**2)
    
    # Empirical CF
    # x_t shape: (batch_size, num_slices, num_integration_points)
    x_t = (x @ A).unsqueeze(2) * t
    ecf = (1j * x_t).exp().mean(0)  # (num_slices, num_integration_points)
    
    # Weighted L2 distance
    err = (ecf - exp_f).abs().square() * exp_f
    
    # Scaling by batch size
    n_total = x.size(0)
    t_stat = torch.trapz(err, t, dim=1) * n_total
    
    return t_stat


def compute_global_views_and_centers(
    jepa_model: JepaModel,
    batch: List[ARCTaskData],
    device: torch.device,
) -> torch.Tensor:
    """Compute global views and task centers for a batch.

    Creates 8 global views, each with 3 examples (150 tokens) sampled with replacement,
    and computes the mean representation (center) for each task.

    Args:
        jepa_model: The JEPA transformer model
        batch: List of task dicts, each with train_input_grids, train_output_grids, etc.
        device: Device to compute on

    Returns:
        Tensor of shape (batch_size, embedding_dim) with task centers
    """
    bs = len(batch)  # number of tasks
    num_views = 8
    
    # Stack all train examples per task: List[Tensor(num_examples, 50)]
    task_examples_list = []
    for task_dict in batch:
        # Stack input and output grids: (num_examples, 25) each
        input_grids = torch.stack(task_dict["train_input_grids"], dim=0)
        output_grids = torch.stack(task_dict["train_output_grids"], dim=0)
        # Concatenate to get (num_examples, 50)
        task_examples = torch.cat([input_grids, output_grids], dim=1)
        task_examples_list.append(task_examples)
    
    # Create 8 global views by randomly sampling 3 examples with replacement
    # Generate random indices for all views and tasks at once
    global_views = []
    for view_idx in range(num_views):
        view_batch = []
        for task_examples in task_examples_list:
            num_examples = task_examples.shape[0]
            # Sample 3 example indices with replacement
            indices = torch.randint(0, num_examples, (3,))
            # Gather and flatten to (150,)
            sampled = task_examples[indices].reshape(-1)
            view_batch.append(sampled)
        global_views.append(torch.stack(view_batch, dim=0))  # (bs, 150)
    
    # Stack all views into single batch for forward pass
    all_batch = torch.cat(global_views, dim=0).to(device)  # (num_views * bs, 150)
    
    # Forward pass for all views
    all_embeddings = jepa_model(all_batch)  # (num_views * bs, embedding_dim)
    
    K = all_embeddings.size(1)  # embedding_dim
    
    # Reshape to (num_views, bs, K)
    all_emb_reshaped = all_embeddings.view(num_views, bs, K)
    
    # Centers: Mean representation of all views per task
    centers = all_emb_reshaped.mean(0)  # (bs, K)
    
    return centers


def compute_jepa_loss_for_batch(
    jepa_model: JepaModel,
    batch: List[ARCTaskData],
    device: torch.device,
    lambd: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute LeJEPA loss for a single batch.

    Creates 8 global views, each with 3 examples (150 tokens) sampled with replacement.

    Args:
        jepa_model: The JEPA transformer model
        batch: List of task dicts, each with train_input_grids, train_output_grids, etc.
        device: Device to compute on
        lambd: Weight for sig_reg loss in JEPA loss

    Returns:
        Tuple of (jepa_loss, jepa_sim_loss, jepa_sig_reg_loss)
    """
    bs = len(batch)  # number of tasks
    num_views = 8
    
    # Stack all train examples per task: List[Tensor(num_examples, 50)]
    task_examples_list = []
    for task_dict in batch:
        # Stack input and output grids: (num_examples, 25) each
        input_grids = torch.stack(task_dict["train_input_grids"], dim=0)
        output_grids = torch.stack(task_dict["train_output_grids"], dim=0)
        # Concatenate to get (num_examples, 50)
        task_examples = torch.cat([input_grids, output_grids], dim=1)
        task_examples_list.append(task_examples)
    
    # Create 8 global views by randomly sampling 3 examples with replacement
    # Generate random indices for all views and tasks at once
    global_views = []
    for view_idx in range(num_views):
        view_batch = []
        for task_examples in task_examples_list:
            num_examples = task_examples.shape[0]
            # Sample 3 example indices with replacement
            indices = torch.randint(0, num_examples, (3,))
            # Gather and flatten to (150,)
            sampled = task_examples[indices].reshape(-1)
            view_batch.append(sampled)
        global_views.append(torch.stack(view_batch, dim=0))  # (bs, 150)
    
    # Stack all views into single batch for forward pass
    all_batch = torch.cat(global_views, dim=0).to(device)  # (num_views * bs, 150)
    
    # Forward pass for all views
    all_embeddings = jepa_model(all_batch)  # (num_views * bs, embedding_dim)
    
    K = all_embeddings.size(1)  # embedding_dim
    
    # Reshape to (num_views, bs, K)
    all_emb_reshaped = all_embeddings.view(num_views, bs, K)
    
    # Centers: Mean representation of all views per task
    centers = all_emb_reshaped.mean(0)  # (bs, K)
    
    # Similarity term: MSE between centers and ALL view embeddings
    sim = (centers.unsqueeze(0) - all_emb_reshaped).square().mean()
    
    # Regularization term: SIG-Reg applied to each view independently
    sig_reg_vals = []
    for view_idx in range(num_views):
        view_emb = all_emb_reshaped[view_idx]  # (bs, K)
        sig_reg_val = sig_reg(view_emb)  # (num_slices,)
        sig_reg_vals.append(sig_reg_val.mean())
    
    sig_reg_loss = torch.stack(sig_reg_vals).mean()
    
    # Final weighted JEPA loss
    jepa_loss = (1 - lambd) * sim + lambd * sig_reg_loss
    
    return jepa_loss, sim, sig_reg_loss


def compute_pred_loss_for_batch(
    jepa_model: JepaModel,
    pred_model: PredictionModel,
    batch: List[ARCTaskData],
    device: torch.device,
) -> Tuple[torch.Tensor, int, int, int]:
    """Compute prediction loss for a single batch.

    For each task, uses the JEPA model to compute task centers, then predicts the test output.

    Args:
        jepa_model: The JEPA transformer model (used without gradients)
        pred_model: The prediction model
        batch: List of task dicts, each with train_input_grids, train_output_grids, etc.
        device: Device to compute on

    Returns:
        Tuple of (pred_loss, num_correct_tokens, num_total_tokens, num_perfect_tasks)
    """
    # Compute centers using JEPA model without gradients
    with torch.no_grad():
        centers = compute_global_views_and_centers(jepa_model, batch, device)
    
    # Stack test grids directly from each task
    pred_input_grids = [task_dict["test_input_grid"] for task_dict in batch]
    pred_output_grids = [task_dict["test_output_grid"] for task_dict in batch]
    
    # Stack into batches
    pred_input_batch = torch.stack(pred_input_grids, dim=0).to(device)  # (bs, 25)
    pred_output_batch = torch.stack(pred_output_grids, dim=0).to(device)  # (bs, 25)
    
    # Forward pass through prediction model
    pred_logits = pred_model(centers, pred_input_batch)  # (bs, 25, vocab_size)
    
    # Compute cross entropy loss
    pred_loss = F.cross_entropy(
        pred_logits.view(-1, pred_model.vocab_size),  # (bs * 25, vocab_size)
        pred_output_batch.view(-1),  # (bs * 25,)
    )
    
    # Compute accuracy metrics
    predicted_output_grids = pred_logits.argmax(dim=2)  # (bs, 25)
    correct_per_task = (predicted_output_grids == pred_output_batch).sum(dim=1)  # (bs,)
    num_correct_tokens = correct_per_task.sum().item()
    num_total_tokens = len(batch) * 25
    num_perfect_tasks = (correct_per_task == 25).sum().item()
    
    return pred_loss, num_correct_tokens, num_total_tokens, num_perfect_tasks


def jepa_train_epoch(
    jepa_model: JepaModel,
    jepa_train_loader: DataLoader,
    jepa_optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambd: float,
) -> Tuple[float, float, float]:
    """Train JEPA model for one epoch.

    Args:
        jepa_model: The JEPA transformer model
        jepa_train_loader: Data loader for JEPA training (concatenation of train and test)
        jepa_optimizer: Optimizer for JEPA model
        device: Device to train on
        lambd: Weight for sig_reg loss in JEPA loss

    Returns:
        Tuple of (avg_jepa_loss, avg_jepa_sim_loss, avg_jepa_sig_reg_loss)
    """
    jepa_model.train()
    total_jepa_loss = 0.0
    total_jepa_sim = 0.0
    total_jepa_sig_reg = 0.0
    num_batches = 0

    for batch_idx, examples in enumerate(jepa_train_loader):
        # Compute JEPA loss
        jepa_loss, jepa_sim, jepa_sig_reg = compute_jepa_loss_for_batch(
            jepa_model, examples, device, lambd
        )

        # Optimize JEPA model
        jepa_optimizer.zero_grad()
        jepa_loss.backward()
        torch.nn.utils.clip_grad_norm_(jepa_model.parameters(), max_norm=1.0)
        jepa_optimizer.step()

        total_jepa_loss += jepa_loss.item()
        total_jepa_sim += jepa_sim.item()
        total_jepa_sig_reg += jepa_sig_reg.item()
        num_batches += 1

    return (
        total_jepa_loss / num_batches,
        total_jepa_sim / num_batches,
        total_jepa_sig_reg / num_batches,
    )


def pred_train_epoch(
    jepa_model: JepaModel,
    pred_model: PredictionModel,
    pred_train_loader: DataLoader,
    pred_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train prediction model for one epoch.

    Args:
        jepa_model: The JEPA transformer model (used without gradients)
        pred_model: The prediction model
        pred_train_loader: Data loader for prediction training
        pred_optimizer: Optimizer for prediction model
        device: Device to train on

    Returns:
        Tuple of (avg_pred_loss, avg_accuracy, perfect_task_rate)
    """
    jepa_model.eval()
    pred_model.train()
    total_pred_loss = 0.0
    total_correct_tokens = 0
    total_tokens = 0
    total_perfect_tasks = 0
    total_tasks = 0
    num_batches = 0

    for batch_idx, examples in enumerate(pred_train_loader):
        # Compute prediction loss
        pred_loss, num_correct, num_total, num_perfect = compute_pred_loss_for_batch(
            jepa_model, pred_model, examples, device
        )

        # Optimize prediction model
        pred_optimizer.zero_grad()
        pred_loss.backward()
        torch.nn.utils.clip_grad_norm_(pred_model.parameters(), max_norm=1.0)
        pred_optimizer.step()

        total_pred_loss += pred_loss.item()
        total_correct_tokens += num_correct
        total_tokens += num_total
        total_perfect_tasks += num_perfect
        total_tasks += len(examples)
        num_batches += 1

    avg_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    perfect_task_rate = total_perfect_tasks / total_tasks if total_tasks > 0 else 0.0

    return (
        total_pred_loss / num_batches,
        avg_accuracy,
        perfect_task_rate,
    )


def pred_test_epoch(
    jepa_model: JepaModel,
    pred_model: PredictionModel,
    pred_test_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate prediction model on test set.

    Args:
        jepa_model: The JEPA transformer model (used without gradients)
        pred_model: The prediction model
        pred_test_loader: Data loader for prediction testing
        device: Device to evaluate on

    Returns:
        Tuple of (avg_pred_loss, avg_accuracy, perfect_task_rate)
    """
    jepa_model.eval()
    pred_model.eval()
    total_pred_loss = 0.0
    total_correct_tokens = 0
    total_tokens = 0
    total_perfect_tasks = 0
    total_tasks = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, examples in enumerate(pred_test_loader):
            # Compute prediction loss
            pred_loss, num_correct, num_total, num_perfect = compute_pred_loss_for_batch(
                jepa_model, pred_model, examples, device
            )

            total_pred_loss += pred_loss.item()
            total_correct_tokens += num_correct
            total_tokens += num_total
            total_perfect_tasks += num_perfect
            total_tasks += len(examples)
            num_batches += 1

    avg_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    perfect_task_rate = total_perfect_tasks / total_tasks if total_tasks > 0 else 0.0

    return (
        total_pred_loss / num_batches,
        avg_accuracy,
        perfect_task_rate,
    )


def jepa_learning_rate_test(
    jepa_model: JepaModel,
    jepa_train_loader: DataLoader,
    device: torch.device,
    lambd: float,
):
    """Test JEPA model learning rate by starting at 1e-7 and doubling every batch.

    Args:
        jepa_model: The JEPA transformer model
        jepa_train_loader: JEPA training data loader
        device: Device to train on
        lambd: Weight for sig_reg loss in JEPA loss
    """
    print("\nStarting JEPA learning rate test...")
    lr = 1e-7
    jepa_model.train()

    batch_count = 0
    for batch_idx, examples in enumerate(jepa_train_loader):
        if batch_count >= 20:
            break

        # Create optimizer with current learning rate
        jepa_opt = torch.optim.Adam(jepa_model.parameters(), lr=lr)

        # Compute JEPA loss
        jepa_loss, jepa_sim, jepa_sig_reg = compute_jepa_loss_for_batch(
            jepa_model, examples, device, lambd
        )

        # Backward pass for JEPA model
        jepa_opt.zero_grad()
        jepa_loss.backward()
        torch.nn.utils.clip_grad_norm_(jepa_model.parameters(), max_norm=1.0)
        jepa_opt.step()

        # Print learning rate and losses
        print(f"Batch {batch_count + 1}: LR = {lr:.2e}, JEPA Loss = {jepa_loss.item():.6f}, "
              f"JEPA Sim = {jepa_sim.item():.6f}, JEPA SigReg = {jepa_sig_reg.item():.6f}")

        # Double learning rate for next batch
        lr *= 2
        batch_count += 1

    print("\nJEPA learning rate test complete!")


def pred_learning_rate_test(
    jepa_model: JepaModel,
    pred_model: PredictionModel,
    pred_train_loader: DataLoader,
    device: torch.device,
):
    """Test prediction model learning rate by starting at 1e-7 and doubling every batch.

    Args:
        jepa_model: The JEPA transformer model (used to compute centers)
        pred_model: The prediction model
        pred_train_loader: Prediction training data loader
        device: Device to train on
    """
    print("\nStarting prediction model learning rate test...")
    lr = 1e-7
    jepa_model.eval()
    pred_model.train()

    batch_count = 0
    for batch_idx, examples in enumerate(pred_train_loader):
        if batch_count >= 20:
            break

        # Create optimizer with current learning rate
        pred_opt = torch.optim.Adam(pred_model.parameters(), lr=lr)

        # Compute prediction loss
        pred_loss, num_correct, num_total, num_perfect = compute_pred_loss_for_batch(
            jepa_model, pred_model, examples, device
        )

        # Backward pass for prediction model
        pred_opt.zero_grad()
        pred_loss.backward()
        torch.nn.utils.clip_grad_norm_(pred_model.parameters(), max_norm=1.0)
        pred_opt.step()

        # Print learning rate and losses
        print(f"Batch {batch_count + 1}: LR = {lr:.2e}, Pred Loss = {pred_loss.item():.6f}, "
              f"Accuracy: {num_correct}/{num_total} ({num_correct/num_total*100:.2f}%)")

        # Double learning rate for next batch
        lr *= 2
        batch_count += 1

    print("\nPrediction model learning rate test complete!")


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

    # Create datasets for different training purposes
    jepa_train_dataset = ConcatDataset([train_dataset, test_dataset])
    pred_train_dataset = train_dataset
    pred_test_dataset = test_dataset

    print(f"JEPA train dataset size: {len(jepa_train_dataset)}")
    print(f"Pred train dataset size: {len(pred_train_dataset)}")
    print(f"Pred test dataset size: {len(pred_test_dataset)}")

    # Create dataloaders
    jepa_train_loader = DataLoader(
        jepa_train_dataset,
        batch_size=config.jepa_batch_size,
        shuffle=True,
        collate_fn=arc_task_collate_fn,
    )
    pred_train_loader = DataLoader(
        pred_train_dataset,
        batch_size=config.pred_batch_size,
        shuffle=True,
        collate_fn=arc_task_collate_fn,
    )
    pred_test_loader = DataLoader(
        pred_test_dataset,
        batch_size=config.pred_batch_size,
        shuffle=True,
        collate_fn=arc_task_collate_fn,
    )

    # Create JEPA model
    jepa_model = JepaModel(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        embedding_dim=config.embedding_dim,
    ).to(device)

    # Create prediction model
    pred_model = PredictionModel(
        jepa_embedding_dim=config.embedding_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
    ).to(device)

    # Compile models for better performance (PyTorch 2.0+)
    print("Compiling models with torch.compile...")
    jepa_model = cast(JepaModel, torch.compile(jepa_model))
    pred_model = cast(PredictionModel, torch.compile(pred_model))

    # Count parameters
    jepa_params = sum(p.numel() for p in jepa_model.parameters() if p.requires_grad)
    pred_params = sum(p.numel() for p in pred_model.parameters() if p.requires_grad)
    total_params = jepa_params + pred_params
    
    print(f"JEPA model has {jepa_params:,} trainable parameters")
    print(f"Prediction model has {pred_params:,} trainable parameters")
    print(f"Total parameters: {total_params:,}")

    # Check if running learning rate test
    if config.mode == "learning_rate_test":
        jepa_learning_rate_test(jepa_model, jepa_train_loader, device, config.lambd)
        pred_learning_rate_test(jepa_model, pred_model, pred_train_loader, device)
        return

    # Create optimizers
    jepa_optimizer = torch.optim.Adam(jepa_model.parameters(), lr=config.learning_rate)
    pred_optimizer = torch.optim.Adam(pred_model.parameters(), lr=config.learning_rate)

    # Load existing models if specified
    start_epoch = 0
    if config.load_model_path:
        if Path(config.load_model_path).exists():
            print(f"\nLoading existing models from {config.load_model_path}")
            checkpoint = torch.load(config.load_model_path, map_location=device)
            jepa_model.load_state_dict(checkpoint["jepa_model_state_dict"])
            pred_model.load_state_dict(checkpoint["pred_model_state_dict"])
            jepa_optimizer.load_state_dict(checkpoint["jepa_optimizer_state_dict"])
            pred_optimizer.load_state_dict(checkpoint["pred_optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            print(f"Resumed from epoch {start_epoch}")
            print(f"Previous train JEPA loss: {checkpoint.get('train_jepa_loss', 'N/A')}")
            print(f"Previous train pred loss: {checkpoint.get('train_pred_loss', 'N/A')}")
            print(f"Previous test JEPA loss: {checkpoint.get('test_jepa_loss', 'N/A')}")
            print(f"Previous test pred loss: {checkpoint.get('test_pred_loss', 'N/A')}")
        else:
            print(
                f"\nWarning: Model path {config.load_model_path} does not exist. Starting from scratch."
            )

    # Create tensorboard writer
    Path(config.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=config.tensorboard_log_dir)

    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        epoch_start_time = time.time()

        # Train JEPA model
        train_jepa_loss, train_jepa_sim, train_jepa_sig_reg = jepa_train_epoch(
            jepa_model, jepa_train_loader, jepa_optimizer, device, config.lambd
        )

        # Train prediction model
        train_pred_loss, train_accuracy, train_perfect_rate = pred_train_epoch(
            jepa_model, pred_model, pred_train_loader, pred_optimizer, device
        )

        # Test prediction model
        test_pred_loss, test_accuracy, test_perfect_rate = pred_test_epoch(
            jepa_model, pred_model, pred_test_loader, device
        )

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Calculate model weight norms (L2 norm of all parameters)
        jepa_norm_sq = torch.tensor(0.0, device=device)
        for p in jepa_model.parameters():
            jepa_norm_sq += p.pow(2).sum()
        jepa_weight_norm = torch.sqrt(jepa_norm_sq).item()

        pred_norm_sq = torch.tensor(0.0, device=device)
        for p in pred_model.parameters():
            pred_norm_sq += p.pow(2).sum()
        pred_weight_norm = torch.sqrt(pred_norm_sq).item()

        # Calculate output projection scales
        jepa_output_proj_scale = torch.sqrt(jepa_model.output_proj.weight.pow(2).sum()).item()
        pred_output_proj_scale = torch.sqrt(pred_model.output_proj.weight.pow(2).sum()).item()

        # Calculate layer-wise mean squares for JEPA model
        jepa_token_emb_mean_sq = jepa_model.token_embedding.weight.pow(2).mean().item()
        jepa_output_proj_mean_sq = jepa_model.output_proj.weight.pow(2).mean().item()
        
        # Transformer layers for JEPA model
        jepa_transformer_weights: List[torch.Tensor] = []
        for layer_module in jepa_model.transformer_encoder.layers:
            encoder_layer = cast(nn.TransformerEncoderLayer, layer_module)
            if encoder_layer.self_attn.in_proj_weight is not None:
                jepa_transformer_weights.append(encoder_layer.self_attn.in_proj_weight.flatten())
            jepa_transformer_weights.append(encoder_layer.self_attn.out_proj.weight.flatten())
            jepa_transformer_weights.append(encoder_layer.linear1.weight.flatten())
            jepa_transformer_weights.append(encoder_layer.linear2.weight.flatten())
        jepa_transformer_mean_sq = torch.cat(jepa_transformer_weights).pow(2).mean().item()

        # Calculate layer-wise mean squares for prediction model
        pred_token_emb_mean_sq = pred_model.token_embedding.weight.pow(2).mean().item()
        pred_output_proj_mean_sq = pred_model.output_proj.weight.pow(2).mean().item()
        
        # Transformer layers for prediction model
        pred_transformer_weights: List[torch.Tensor] = []
        for layer_module in pred_model.transformer_encoder.layers:
            encoder_layer = cast(nn.TransformerEncoderLayer, layer_module)
            if encoder_layer.self_attn.in_proj_weight is not None:
                pred_transformer_weights.append(encoder_layer.self_attn.in_proj_weight.flatten())
            pred_transformer_weights.append(encoder_layer.self_attn.out_proj.weight.flatten())
            pred_transformer_weights.append(encoder_layer.linear1.weight.flatten())
            pred_transformer_weights.append(encoder_layer.linear2.weight.flatten())
        pred_transformer_mean_sq = torch.cat(pred_transformer_weights).pow(2).mean().item()

        # Log to console
        print(
            f"Epoch {epoch + 1}/{start_epoch + config.num_epochs} - "
            f"Train JEPA Loss: {train_jepa_loss:.6f}, Train Pred Loss: {train_pred_loss:.6f}, "
            f"Test Pred Loss: {test_pred_loss:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        print(
            f"  JEPA Loss Components - "
            f"Train Sim: {train_jepa_sim:.6f}, Train SigReg: {train_jepa_sig_reg:.6f}"
        )
        print(
            f"  Accuracy - "
            f"Train: {train_accuracy*100:.2f}%, Train Perfect: {train_perfect_rate*100:.2f}%, "
            f"Test: {test_accuracy*100:.2f}%, Test Perfect: {test_perfect_rate*100:.2f}%"
        )
        print(
            f"  Model Norms - "
            f"JEPA: {jepa_weight_norm:.4f}, Pred: {pred_weight_norm:.4f}, "
            f"JEPA Out Scale: {jepa_output_proj_scale:.4f}, Pred Out Scale: {pred_output_proj_scale:.4f}"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train_jepa", train_jepa_loss, epoch)
        writer.add_scalar("Loss/train_pred", train_pred_loss, epoch)
        writer.add_scalar("Loss/test_pred", test_pred_loss, epoch)
        writer.add_scalar("Loss/train_jepa_sim", train_jepa_sim, epoch)
        writer.add_scalar("Loss/train_jepa_sig_reg", train_jepa_sig_reg, epoch)
        writer.add_scalar("Accuracy/train_accuracy", train_accuracy, epoch)
        writer.add_scalar("Accuracy/train_perfect_rate", train_perfect_rate, epoch)
        writer.add_scalar("Accuracy/test_accuracy", test_accuracy, epoch)
        writer.add_scalar("Accuracy/test_perfect_rate", test_perfect_rate, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)
        writer.add_scalar("Model/jepa_weight_norm", jepa_weight_norm, epoch)
        writer.add_scalar("Model/pred_weight_norm", pred_weight_norm, epoch)
        writer.add_scalar("Model/jepa_output_proj_scale", jepa_output_proj_scale, epoch)
        writer.add_scalar("Model/pred_output_proj_scale", pred_output_proj_scale, epoch)
        
        # Log layer-wise mean squares for JEPA model
        writer.add_scalar("LayerMeanSquare/jepa_token_embedding", jepa_token_emb_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/jepa_output_proj", jepa_output_proj_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/jepa_transformer_avg", jepa_transformer_mean_sq, epoch)
        
        # Log layer-wise mean squares for prediction model
        writer.add_scalar("LayerMeanSquare/pred_token_embedding", pred_token_emb_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/pred_output_proj", pred_output_proj_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/pred_transformer_avg", pred_transformer_mean_sq, epoch)

        # Save checkpoint every N epochs (configurable)
        if (epoch + 1) % config.checkpoint_save_interval == 0:
            checkpoint_path = f"{config.checkpoint_dir}/{config.timestamp}_epoch_{epoch + 1}_checkpoint.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "jepa_model_state_dict": jepa_model.state_dict(),
                    "pred_model_state_dict": pred_model.state_dict(),
                    "jepa_optimizer_state_dict": jepa_optimizer.state_dict(),
                    "pred_optimizer_state_dict": pred_optimizer.state_dict(),
                    "train_jepa_loss": train_jepa_loss,
                    "train_pred_loss": train_pred_loss,
                    "test_pred_loss": test_pred_loss,
                    "config": {
                        "d_model": config.d_model,
                        "nhead": config.nhead,
                        "num_layers": config.num_layers,
                        "dim_feedforward": config.dim_feedforward,
                        "vocab_size": config.vocab_size,
                        "dropout": config.dropout,
                        "embedding_dim": config.embedding_dim,
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

    # Save final models
    Path(config.model_save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "jepa_model_state_dict": jepa_model.state_dict(),
            "pred_model_state_dict": pred_model.state_dict(),
            "jepa_optimizer_state_dict": jepa_optimizer.state_dict(),
            "pred_optimizer_state_dict": pred_optimizer.state_dict(),
            "config": {
                "d_model": config.d_model,
                "nhead": config.nhead,
                "num_layers": config.num_layers,
                "dim_feedforward": config.dim_feedforward,
                "vocab_size": config.vocab_size,
                "dropout": config.dropout,
                "embedding_dim": config.embedding_dim,
            },
        },
        config.model_save_path,
    )
    print(f"Models saved to {config.model_save_path}")


def main():
    # Create configuration
    config = Config(
        # Dataset creation parameters
        data_dir=Path("data/MINI-ARC"),
        output_dir=Path("output/mini_arc_jepa"),
        test_ratio=0.2,
        random_seed=42,
        max_augmentations=500,
        # Model parameters
        d_model=128,
        nhead=8,
        num_layers=8,
        dim_feedforward=512,
        dropout=0.1,
        embedding_dim=512,
        # Data parameters
        vocab_size=10,
        # Training parameters
        num_epochs=150,
        jepa_batch_size=128,
        pred_batch_size=32,
        learning_rate=1e-4,
        lambd=0.05,
        mode="train",
        checkpoint_save_interval=25,
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

    # Train model
    train(config)

    # Shut down Google Colab runtime if running in Colab (only in train mode)
    if config.mode == "train":
        try:
            from google.colab import runtime # type: ignore
            print("\nShutting down Google Colab runtime...")
            runtime.unassign()
        except ImportError:
            # Not running in Colab, skip shutdown
            pass


if __name__ == "__main__":
    main()
