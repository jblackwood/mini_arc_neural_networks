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
    batch_size: int
    num_epochs: int
    learning_rate: float
    lambd: float  # Weight for loss_sig_reg in loss calculation
    mode: Literal["train", "learning_rate_test"]
    checkpoint_save_interval: int
    eval_interval: int  # Number of epochs between eval steps
    eval_optimization_steps: int  # Number of optimization steps per test example
    eval_learning_rate: float  # Learning rate for eval optimization

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


class ARCTaskExample(TypedDict):
    """Typed dict for a single ARC example with token sequences."""
    input_grid: torch.Tensor  # Shape: (25,) - token values
    output_grid: torch.Tensor  # Shape: (25,) - token values
    example_type: Literal["train", "test"]
    task_id: str


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

    # Filter to only tasks with 3 train and 1 test example
    filtered_files = []
    for json_file in json_files:
        try:
            task = parse_arc_json(json_file)
            if len(task.train) == 3 and len(task.test) == 1:
                filtered_files.append(json_file)
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue
    
    print(f"Filtered to {len(filtered_files)} tasks with 3 train and 1 test examples")
    assert len(filtered_files) == 82, f"Expected 82 tasks with 3 train and 1 test, but found {len(filtered_files)}"

    # Shuffle and split into train and test
    random.shuffle(filtered_files)
    num_test = int(len(filtered_files) * test_ratio)
    test_files = filtered_files[:num_test]
    train_files = filtered_files[num_test:]

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

    def __init__(self, folder_path: str, vocab_size: int = 10, grids: Literal["all", "train", "test"] = "all"):
        """Initialize the dataset.

        Args:
            folder_path: Path to folder containing task JSON files
            vocab_size: Size of vocabulary (default 10: 0-9 for colors)
            grids: Which grids to return - 'all' for both train and test, 'train' for train only, 'test' for test only
        """
        self.folder_path = Path(folder_path)
        self.task_files = sorted(self.folder_path.glob("*.json"))
        self.vocab_size = vocab_size
        self.grids = grids

    def __len__(self) -> int:
        return len(self.task_files)

    def __getitem__(self, idx: int) -> List[ARCTaskExample]:
        """Get a task as a list of example dicts.

        Args:
            idx: Index of the task

        Returns:
            List of ARCTaskExample dicts, each containing input_grid, output_grid, and example_type.
        """
        task_file = self.task_files[idx]
        task_data = parse_arc_json(task_file)
        
        # Extract task_id from filename (without .json extension)
        task_id = task_file.stem

        # Collect examples based on grids parameter
        examples_to_process: List[Tuple[ARCExample, Literal["train", "test"]]] = []
        if self.grids == "all":
            # Add train examples
            for ex in task_data.train:
                examples_to_process.append((ex, "train"))
            # Add test examples
            for ex in task_data.test:
                examples_to_process.append((ex, "test"))
        elif self.grids == "train":
            for ex in task_data.train:
                examples_to_process.append((ex, "train"))
        elif self.grids == "test":
            for ex in task_data.test:
                examples_to_process.append((ex, "test"))

        result: List[ARCTaskExample] = []
        for example, example_type in examples_to_process:
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

            # Convert to tensors of token values (not one-hot)
            input_tokens = torch.tensor(input_cells, dtype=torch.long)
            output_tokens = torch.tensor(output_cells, dtype=torch.long)

            result.append(ARCTaskExample(
                input_grid=input_tokens,
                output_grid=output_tokens,
                example_type=example_type,
                task_id=task_id,
            ))

        return result


def arc_collate_fn(batch: List[List[ARCTaskExample]]) -> List[ARCTaskExample]:
    """Collate function to combine lists of examples from multiple tasks into a single list.
    
    Args:
        batch: List of lists, where each inner list contains examples from one task
    
    Returns:
        Flattened list of all examples from all tasks in the batch
    """
    result: List[ARCTaskExample] = []
    for task_examples in batch:
        result.extend(task_examples)
    return result


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


class TransformerModel(nn.Module):
    """Non-causal transformer encoder for ARC tasks."""

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

        # Token embedding layer (vocab_size -> d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Initialize token embeddings with mean 0 and std 0.01
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.01)

        # Learnable position embeddings for grid types
        self.input_grid_embedding = nn.Parameter(torch.randn(d_model))
        self.output_grid_embedding = nn.Parameter(torch.randn(d_model))

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

        # Output projection to embedding_dim
        self.output_proj = nn.Linear(d_model, embedding_dim)

    def _forward_from_grid_embeddings(self, grid_embeddings: torch.Tensor) -> torch.Tensor:
        """Process grid embeddings through the model architecture.
        
        Args:
            grid_embeddings: Tensor of shape (batch_size, 50, d_model) containing
                            concatenated input and output grid embeddings
        
        Returns:
            Tensor of shape (batch_size, embedding_dim) with final embeddings
        """
        # Split into input and output grids
        input_grid = grid_embeddings[:, :25, :]  # (batch_size, 25, d_model)
        output_grid = grid_embeddings[:, 25:, :]  # (batch_size, 25, d_model)
        
        # Add grid type embeddings
        input_grid = input_grid + self.input_grid_embedding  # (batch_size, 25, d_model)
        output_grid = output_grid + self.output_grid_embedding  # (batch_size, 25, d_model)
        
        # Apply 2D rotary position embeddings separately to each grid
        input_grid = self.rope(input_grid)  # (batch_size, 25, d_model)
        output_grid = self.rope(output_grid)  # (batch_size, 25, d_model)
        
        # Concatenate: input grid with RoPE, output grid with RoPE
        x = torch.cat([input_grid, output_grid], dim=1)  # (batch_size, 50, d_model)

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 50, d_model)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Project to embedding_dim
        x = self.output_proj(x)  # (batch_size, embedding_dim)
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 50) - tokens for input and output grids

        Returns:
            Tensor of shape (batch_size, embedding_dim) with embeddings
        """
        # Apply token embedding to grid tokens
        grid_embeddings = self.token_embedding(x)  # (batch_size, 50, d_model)
        
        # Process through the rest of the model
        return self._forward_from_grid_embeddings(grid_embeddings)


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


def compute_loss_for_batch(
    model: TransformerModel,
    examples: List[ARCTaskExample],
    device: torch.device,
    lambd: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute LeJEPA loss for a single batch.

    Args:
        model: The transformer model
        examples: List of ARCTaskExample dicts with task_id, input_grid, output_grid
        device: Device to compute on
        lambd: Weight for sig_reg loss

    Returns:
        Tuple of (total_loss, sim_loss, sig_reg_loss) tensors (all scalars)
    """
    # Group examples by task_id
    task_groups = defaultdict(list)
    for example in examples:
        task_id = example["task_id"]
        # Concatenate input and output grids
        concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)  # (50,)
        task_groups[task_id].append(concatenated)
    
    # Ensure all tasks have exactly 4 views (duplicate last example if needed)
    num_views = 4
    all_views = []
    for task_id in sorted(task_groups.keys()):  # Sort for determinism
        task_examples = task_groups[task_id]
        while len(task_examples) < num_views:
            # Duplicate the last example to pad to 4 views
            task_examples.append(task_examples[-1])
        # Only take first 4 if there are more
        task_examples = task_examples[:num_views]
        all_views.extend(task_examples)
    
    # Stack all views into single batch
    batch = torch.stack(all_views, dim=0).to(device)  # (num_tasks * num_views, 50)
    
    # Forward pass - model returns embeddings
    embeddings = model(batch)  # (num_tasks * num_views, embedding_dim)
    
    bs = len(task_groups)  # number of tasks
    K = embeddings.size(1)  # embedding_dim
    
    # Reshape to (num_views, bs, K)
    embeddings_reshaped = embeddings.view(num_views, bs, K)
    
    # Centers: Mean representation across all views per task
    centers = embeddings_reshaped.mean(0)  # (bs, K)
    
    # Similarity term: MSE between centers and all view embeddings
    sim = (centers.unsqueeze(0) - embeddings_reshaped).square().mean()
    
    # Regularization term: SIG-Reg applied to each view independently
    sig_reg_vals = []
    for view_idx in range(num_views):
        view_emb = embeddings_reshaped[view_idx]  # (bs, K)
        sig_reg_val = sig_reg(view_emb)  # (num_slices,)
        sig_reg_vals.append(sig_reg_val.mean())
    
    sig_reg_val = torch.stack(sig_reg_vals).mean()
    
    # Final weighted loss
    loss = (1 - lambd) * sim + lambd * sig_reg_val
    
    return loss, sim, sig_reg_val


def eval_step(
    model: TransformerModel,
    test_loader: DataLoader,
    device: torch.device,
    eval_optimization_steps: int,
    eval_learning_rate: float,
    epoch: int,
    writer: SummaryWriter,
) -> Tuple[float, float]:
    """Evaluate model by optimizing output grids to match task centers.
    
    Args:
        model: The transformer model
        test_loader: Test data loader
        device: Device to evaluate on
        eval_optimization_steps: Number of optimization steps per test example
        eval_learning_rate: Learning rate for optimization
        epoch: Current epoch number
        writer: Tensorboard writer
    
    Returns:
        Tuple of (average_accuracy, perfect_task_rate)
    """
    model.eval()
    
    total_correct_tokens = 0
    total_tokens = 0
    perfect_tasks = 0
    total_tasks = 0
    
    # Process one batch at a time
    for batch_examples in test_loader:
        # Group examples in this batch by task_id
        task_groups = defaultdict(list)
        for example in batch_examples:
            task_id = example["task_id"]
            task_groups[task_id].append(example)
        
        # Collect all tasks and assert they have exactly 3 train and 1 test example
        valid_tasks = []
        for task_id, task_examples in task_groups.items():
            train_examples = [ex for ex in task_examples if ex["example_type"] == "train"]
            test_examples = [ex for ex in task_examples if ex["example_type"] == "test"]
            
            assert len(train_examples) == 3, f"Task {task_id} has {len(train_examples)} train examples, expected 3"
            assert len(test_examples) == 1, f"Task {task_id} has {len(test_examples)} test examples, expected 1"
            
            valid_tasks.append({
                "task_id": task_id,
                "train": train_examples,
                "test": test_examples[0]
            })
        
        if len(valid_tasks) == 0:
            continue
        
        num_tasks = len(valid_tasks)
        
        # Stack all train examples from all tasks
        # Shape: (num_tasks * 3, 50)
        all_train_grids = []
        for task in valid_tasks:
            for ex in task["train"]:
                concatenated = torch.cat([ex["input_grid"], ex["output_grid"]], dim=0)
                all_train_grids.append(concatenated)
        
        train_batch = torch.stack(all_train_grids, dim=0).to(device)  # (num_tasks * 3, 50)
        
        # Compute embeddings for all train examples
        with torch.no_grad():
            train_embeddings = model(train_batch)  # (num_tasks * 3, embedding_dim)
            
            # Reshape to (num_tasks, 3, embedding_dim) and compute centers
            train_embeddings = train_embeddings.view(num_tasks, 3, -1)  # (num_tasks, 3, embedding_dim)
            centers = train_embeddings.mean(dim=1)  # (num_tasks, embedding_dim)
        
        # Stack all test input grids and true output grids
        test_input_grids = torch.stack([task["test"]["input_grid"] for task in valid_tasks], dim=0).to(device)  # (num_tasks, 25)
        true_output_grids = torch.stack([task["test"]["output_grid"] for task in valid_tasks], dim=0).to(device)  # (num_tasks, 25)
        
        # Initialize optimizable output grid parameters for all tasks
        # Shape: (num_tasks, 25, d_model)
        output_grid_params = nn.Parameter(
            torch.randn(num_tasks, 25, model.token_embedding.embedding_dim, device=device) * 0.01
        )
        
        # Create optimizer for all output grid parameters
        optimizer = torch.optim.Adam([output_grid_params], lr=eval_learning_rate)
        
        # Vectorized optimization loop
        for step in range(eval_optimization_steps):
            optimizer.zero_grad()
            
            # Build embedding sequences for all tasks
            # Input grids: use token embeddings (frozen)
            input_embeddings = model.token_embedding(test_input_grids)  # (num_tasks, 25, d_model)
            
            # Concatenate input embeddings with optimizable output parameters
            grid_embeddings = torch.cat([input_embeddings, output_grid_params], dim=1)  # (num_tasks, 50, d_model)
            
            # Process through the model (frozen)
            embeddings = model._forward_from_grid_embeddings(grid_embeddings)  # (num_tasks, embedding_dim)
            
            # Compute loss for all tasks
            loss = (embeddings - centers).square().mean()
            
            # Backprop
            loss.backward()
            optimizer.step()
        
        # After optimization, convert output_grid_params to tokens for all tasks
        with torch.no_grad():
            # For each position, find nearest token embedding
            token_embeddings = model.token_embedding.weight  # (vocab_size, d_model)
            
            # Vectorized nearest neighbor search
            # output_grid_params: (num_tasks, 25, d_model)
            # token_embeddings: (vocab_size, d_model)
            # Compute distances: (num_tasks, 25, vocab_size)
            distances = torch.cdist(output_grid_params, token_embeddings.unsqueeze(0).expand(num_tasks, -1, -1))
            
            # Find nearest token for each position
            optimized_output_grids = distances.argmin(dim=2)  # (num_tasks, 25)
            
            # Calculate accuracy for all tasks
            correct_per_task = (optimized_output_grids == true_output_grids).sum(dim=1)  # (num_tasks,)
            
            total_correct_tokens += correct_per_task.sum().item()
            total_tokens += num_tasks * 25
            
            # Count perfect tasks
            perfect_tasks += (correct_per_task == 25).sum().item()
            total_tasks += num_tasks
    
    # Calculate metrics
    avg_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0.0
    perfect_task_rate = perfect_tasks / total_tasks if total_tasks > 0 else 0.0
    
    # Log to console
    print(f"  Eval: Avg Accuracy: {avg_accuracy*100:.2f}% ({total_correct_tokens}/{total_tokens}), Perfect Tasks: {perfect_task_rate*100:.2f}% ({perfect_tasks}/{total_tasks})")
    
    # Log to tensorboard
    writer.add_scalar("Eval/avg_accuracy", avg_accuracy, epoch)
    writer.add_scalar("Eval/perfect_task_rate", perfect_task_rate, epoch)
    
    return avg_accuracy, perfect_task_rate


def train_epoch(
    model: TransformerModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambd: float,
) -> Tuple[float, float, float]:
    """Train for one epoch.

    Args:
        model: The transformer model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        lambd: Weight for sig_reg loss

    Returns:
        Tuple of (average_total_loss, average_sim_loss, average_sig_reg_loss)
    """
    model.train()
    total_loss = 0.0
    total_sim = 0.0
    total_sig_reg = 0.0
    num_batches = 0

    for batch_idx, examples in enumerate(train_loader):
        # Compute loss with raw examples
        loss, sim, sig_reg_val = compute_loss_for_batch(model, examples, device, lambd)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_sim += sim.item()
        total_sig_reg += sig_reg_val.item()
        num_batches += 1

    return total_loss / num_batches, total_sim / num_batches, total_sig_reg / num_batches


def test_epoch(
    model: TransformerModel,
    test_loader: DataLoader,
    device: torch.device,
    lambd: float,
) -> Tuple[float, float, float]:
    """Evaluate on test set.

    Args:
        model: The transformer model
        test_loader: Test data loader
        device: Device to evaluate on
        lambd: Weight for sig_reg loss

    Returns:
        Tuple of (average_total_loss, average_sim_loss, average_sig_reg_loss)
    """
    model.eval()
    total_loss = 0.0
    total_sim = 0.0
    total_sig_reg = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, examples in enumerate(test_loader):
            # Compute loss with raw examples
            loss, sim, sig_reg_val = compute_loss_for_batch(model, examples, device, lambd)

            total_loss += loss.item()
            total_sim += sim.item()
            total_sig_reg += sig_reg_val.item()
            num_batches += 1

    return total_loss / num_batches, total_sim / num_batches, total_sig_reg / num_batches


def learning_rate_test(
    model: TransformerModel,
    train_loader: DataLoader,
    device: torch.device,
    lambd: float,
):
    """Test learning rate by starting at 1e-7 and doubling every batch.

    Args:
        model: The transformer model
        train_loader: Training data loader
        device: Device to train on
        lambd: Weight for sig_reg loss
    """
    # Start learning rate test
    print("\nStarting learning rate test...")
    lr = 1e-7
    model.train()

    batch_count = 0
    for batch_idx, examples in enumerate(train_loader):
        if batch_count >= 20:
            break

        # Create optimizer with current learning rate
        opt = torch.optim.Adam(
            model.parameters(), lr=lr
        )

        # Compute loss with raw examples
        loss, sim, sig_reg_val = compute_loss_for_batch(model, examples, device, lambd)

        # Backward pass
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # Print learning rate and loss
        print(f"Batch {batch_count + 1}: LR = {lr:.2e}, Loss = {loss.item():.6f}, Sim = {sim.item():.6f}, SigReg = {sig_reg_val.item():.6f}")

        # Double learning rate for next batch
        lr *= 2
        batch_count += 1

    print("\nLearning rate test complete!")


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
    # Train dataset uses all grids (train + test from each task)
    train_dataset_all = ARCTaskDataset(config.train_data_dir, vocab_size=config.vocab_size, grids="all")
    # Test dataset train portion for training
    test_dataset_train = ARCTaskDataset(config.test_data_dir, vocab_size=config.vocab_size, grids="train")
    
    # Concatenate train_dataset_all with test_dataset_train for training
    train_dataset = ConcatDataset([train_dataset_all, test_dataset_train])
    
    # Use all examples from test set for evaluation
    test_dataset = ARCTaskDataset(config.test_data_dir, vocab_size=config.vocab_size, grids="all")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=arc_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=arc_collate_fn,
    )

    # Create model
    model = TransformerModel(
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        vocab_size=config.vocab_size,
        dropout=config.dropout,
        embedding_dim=config.embedding_dim,
    ).to(device)

    # Compile model for better performance (PyTorch 2.0+)
    print("Compiling model with torch.compile...")
    model = cast(TransformerModel, torch.compile(model))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model has {total_params:,} trainable parameters")

    # Check if running learning rate test
    if config.mode == "learning_rate_test":
        learning_rate_test(model, train_loader, device, config.lambd)
        return

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

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
        train_loss, train_sim, train_sig_reg = train_epoch(model, train_loader, optimizer, device, config.lambd)

        # Test
        test_loss, test_sim, test_sig_reg = test_epoch(model, test_loader, device, config.lambd)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Calculate model weight norm (L2 norm of all parameters)
        total_norm_sq = torch.tensor(0.0, device=device)
        for p in model.parameters():
            total_norm_sq += p.pow(2).sum()
        model_weight_norm = torch.sqrt(total_norm_sq).item()

        # Calculate output projection scale (norm of output projection layer weights)
        output_proj_scale = torch.sqrt(
            model.output_proj.weight.pow(2).sum()
        ).item()

        # Calculate layer-wise mean square using vectorized operations
        # Embedding layers
        token_emb_mean_sq = model.token_embedding.weight.pow(2).mean().item()
        
        # Output projection
        output_proj_mean_sq = model.output_proj.weight.pow(2).mean().item()
        
        # Transformer layers (vectorized - concatenate all weights and compute mean)
        transformer_weights: List[torch.Tensor] = []
        for layer_module in model.transformer_encoder.layers:
            encoder_layer = cast(nn.TransformerEncoderLayer, layer_module)
            # Self-attention weights
            if encoder_layer.self_attn.in_proj_weight is not None:
                transformer_weights.append(encoder_layer.self_attn.in_proj_weight.flatten())
            transformer_weights.append(encoder_layer.self_attn.out_proj.weight.flatten())
            # Feedforward weights
            transformer_weights.append(encoder_layer.linear1.weight.flatten())
            transformer_weights.append(encoder_layer.linear2.weight.flatten())
        transformer_mean_sq = torch.cat(transformer_weights).pow(2).mean().item()

        # Log to console
        print(
            f"Epoch {epoch + 1}/{start_epoch + config.num_epochs} - "
            f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
            f"Time: {epoch_time:.2f}s, "
            f"Weight Norm: {model_weight_norm:.4f}, "
            f"Output Scale: {output_proj_scale:.4f}"
        )
        print(
            f"  Loss Components - "
            f"Train Sim: {train_sim:.6f}, Train SigReg: {train_sig_reg:.6f}, "
            f"Test Sim: {test_sim:.6f}, Test SigReg: {test_sig_reg:.6f}"
        )
        print(
            f"  Layer Mean Squares - "
            f"Token Emb: {token_emb_mean_sq:.6f}, "
            f"Out Proj: {output_proj_mean_sq:.6f}, "
            f"Transformer: {transformer_mean_sq:.6f}"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Loss/train_sim", train_sim, epoch)
        writer.add_scalar("Loss/train_sig_reg", train_sig_reg, epoch)
        writer.add_scalar("Loss/test_sim", test_sim, epoch)
        writer.add_scalar("Loss/test_sig_reg", test_sig_reg, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)
        writer.add_scalar("Model/weight_norm", model_weight_norm, epoch)
        writer.add_scalar("Model/output_proj_scale", output_proj_scale, epoch)
        
        # Log layer-wise mean squares
        writer.add_scalar("LayerMeanSquare/token_embedding", token_emb_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/output_proj", output_proj_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/transformer_avg", transformer_mean_sq, epoch)

        # Run eval step every eval_interval epochs
        if (epoch + 1) % config.eval_interval == 0:
            eval_start_time = time.time()
            eval_step(
                model,
                test_loader,
                device,
                config.eval_optimization_steps,
                config.eval_learning_rate,
                epoch,
                writer,
            )
            eval_time = time.time() - eval_start_time
            print(f"  Eval Time: {eval_time:.2f}s")

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
        output_dir=Path("output/mini_arc_jepa"),
        test_ratio=0.2,
        random_seed=42,
        max_augmentations=500,
        # Model parameters
        d_model=256,
        nhead=8,
        num_layers=8,
        dim_feedforward=1024,
        dropout=0.1,
        embedding_dim=1024,
        # Data parameters
        vocab_size=10,
        # Training parameters
        num_epochs=150,
        batch_size=128,
        learning_rate=1e-4,
        lambd=0.00001,
        mode="train",
        checkpoint_save_interval=50,
        eval_interval=1,
        eval_optimization_steps=100,
        eval_learning_rate=1e-2,
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
