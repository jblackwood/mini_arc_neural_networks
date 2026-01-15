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
    l1_weight: float

    # Training parameters
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    mode: Literal["train", "learning_rate_test", "weight_decay_test", "eval"]
    checkpoint_save_interval: int

    # Data parameters
    vocab_size: int

    # Denoising evaluation parameters
    eval_denoise_epoch_interval: int
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


class ARCTaskExample(TypedDict):
    """Typed dict for a single ARC example with token sequences."""
    input_grid: torch.Tensor  # Shape: (25,) - token values
    output_grid: torch.Tensor  # Shape: (25,) - token values
    example_type: Literal["train", "test"]
    task_id: str


@dataclass
class DenoisingResult:
    """Result from denoising evaluation.

    Attributes:
        accuracies: Optional tensor of shape (batch_size,) with accuracy for each task
        predicted_grids: Optional tensor of shape (batch_size, 5, 5) with predicted output grids
        optimized_output_tokens: Optional tensor of shape (batch_size, 25) with optimized output token values
        best_grad_norm: Optional tensor of shape (batch_size,) with number of changed tokens for each task
        best_iteration: Optional tensor of shape (batch_size,) with iteration of best grad norm for each task
    """

    accuracies: Optional[torch.Tensor] = None
    predicted_grids: Optional[torch.Tensor] = None
    optimized_output_tokens: Optional[torch.Tensor] = None
    best_grad_norm: Optional[torch.Tensor] = None
    best_iteration: Optional[torch.Tensor] = None


@dataclass
class LossResult:
    """Result from loss computation.

    Attributes:
        loss: Total loss (cross-entropy + L1 regularization)
        ce_loss: Cross-entropy loss component
        l1_loss: L1 regularization loss component
    """

    loss: torch.Tensor
    ce_loss: torch.Tensor
    l1_loss: torch.Tensor


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
    """PyTorch dataset that loads ARC tasks and returns list of example dicts.

    Each task returns a list of dicts, where each dict contains:
    - input_grid: (25,) token values for the flattened input grid
    - output_grid: (25,) token values for the flattened output grid
    - example_type: 'train' or 'test' indicating which set the example came from
    """

    def __init__(self, folder_path: str, vocab_size: int = 11, grids: Literal["all", "train", "test"] = "all"):
        """Initialize the dataset.

        Args:
            folder_path: Path to folder containing task JSON files
            vocab_size: Size of vocabulary (default 11: 0-9 for colors, 10 for mask token)
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
        num_tasks: int,
    ):
        """Initialize the transformer model.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            vocab_size: Number of possible cell values (11 for ARC: 0-9 colors + mask token)
            dropout: Dropout rate
            num_tasks: Number of unique tasks for task embedding
        """
        super().__init__()

        # Store vocab_size and d_model for later use
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Sparse task embedding layer (size d_model * 5) with max_norm constraint
        self.task_embedding = nn.Embedding(num_tasks, d_model * 5)
        
        # Dictionary matrix to project sparse embeddings to d_model
        # Columns of the dictionary (rows of weight.T) will be normalized in forward pass
        self.dictionary = nn.Linear(d_model * 5, d_model, bias=False)
        self.dictionary_max_norm = 1.0

        # Token embedding layer (vocab_size -> d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model, max_norm=1.0)

        # Learnable position embeddings for task token and grid types
        self.task_embedding_position = nn.Parameter(torch.randn(d_model))
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

        # Two linear output projection layers with transpose
        # First: (batch_size, 51, d_model) -> (batch_size, 51, 25)
        # Transpose: (batch_size, 51, 25) -> (batch_size, 25, 51)
        # Second: (batch_size, 25, 51) -> (batch_size, 25, vocab_size+1)
        self.output_proj_1 = nn.Linear(d_model, 25)
        self.output_proj_2 = nn.Linear(51, vocab_size + 1)  # +1 for "no change" token

    def forward(self, x: torch.Tensor, task_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 50) - tokens for input and output grids
            task_indices: Task indices of shape (batch_size,) - integer indices for task embeddings

        Returns:
            Tensor of shape (batch_size, 25, vocab_size+1) with logits for token change prediction
        """
        
        # Get sparse task embeddings and project through dictionary
        task_emb_sparse = torch.sigmoid(self.task_embedding(task_indices))  # (batch_size, d_model*5)
        
        # Normalize dictionary columns (rows of weight) to have max_norm
        with torch.no_grad():
            # Dictionary weight shape: (d_model, d_model*5)
            # Normalize each row (dictionary atom) to have max_norm
            norms = torch.norm(self.dictionary.weight, p=2, dim=1, keepdim=True)
            desired_norms = torch.clamp(norms, max=self.dictionary_max_norm)
            self.dictionary.weight.mul_(desired_norms / (norms + 1e-8))
        
        task_emb = self.dictionary(task_emb_sparse)  # (batch_size, d_model)
        task_emb = task_emb + self.task_embedding_position  # Add task position embedding
        task_emb = task_emb.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Apply token embedding to grid tokens
        x = self.token_embedding(x)  # (batch_size, 50, d_model)
        
        # Split into input and output grids
        input_grid = x[:, :25, :]  # (batch_size, 25, d_model)
        output_grid = x[:, 25:, :]  # (batch_size, 25, d_model)
        
        # Add grid type embeddings
        input_grid = input_grid + self.input_grid_embedding  # (batch_size, 25, d_model)
        output_grid = output_grid + self.output_grid_embedding  # (batch_size, 25, d_model)
        
        # Apply 2D rotary position embeddings separately to each grid
        input_grid = self.rope(input_grid)  # (batch_size, 25, d_model)
        output_grid = self.rope(output_grid)  # (batch_size, 25, d_model)
        
        # Concatenate: task token, input grid with RoPE, output grid with RoPE
        x = torch.cat([task_emb, input_grid, output_grid], dim=1)  # (batch_size, 51, d_model)

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 51, d_model)

        # Apply first output projection and transpose
        x = self.output_proj_1(x)  # (batch_size, 51, 25)
        x = x.transpose(1, 2)  # (batch_size, 25, 51)
        
        # Second output projection
        logits = self.output_proj_2(x)  # (batch_size, 25, vocab_size+1)
        
        # Return raw logits for training (cross-entropy expects logits)
        return logits


def optimize_output_grid(
    model: TransformerModel,
    x_input: torch.Tensor,
    task_indices: torch.Tensor,
    num_iterations: int,
) -> DenoisingResult:
    """Optimize the output grid using iterative token prediction.

    Args:
        model: The transformer model to use for computing token predictions
        x_input: Input tensor of shape (batch_size, 50) - tokens
        task_indices: Task indices of shape (batch_size,)
        num_iterations: Number of optimization iterations

    Returns:
        DenoisingResult with optimized_output_tokens and best_grad_norm fields populated
    """
    batch_size = x_input.shape[0]
    vocab_size = model.vocab_size
    no_change_token = vocab_size  # The "no change" token

    with torch.no_grad():
        x = x_input.clone()

        # Track best grid and number of changes per sample in batch
        best_grad_norm = torch.full((batch_size,), float("inf"), device=x_input.device)
        best_iteration = torch.zeros((batch_size,), dtype=torch.long, device=x_input.device)
        best_x = x.clone()

        for iteration in range(num_iterations):
            # Get logits from model
            logits = model(x, task_indices)  # (batch_size, 25, vocab_size+1)
            
            # Get the most likely token for each position
            token_change_predictions = torch.argmax(logits, dim=2)  # (batch_size, 25)
            
            # Update x: where token_change_predictions is no_change_token, keep x's token unchanged
            # Where it's not no_change_token, replace x's token with the predicted token
            is_no_change = (token_change_predictions == no_change_token)  # (batch_size, 25)
            
            # Get current output tokens
            current_output = x[:, -25:]  # (batch_size, 25)
            
            # New output: keep current where no_change, otherwise use prediction
            new_output = torch.where(is_no_change, current_output, token_change_predictions)
            
            # Update x with new output tokens
            x[:, -25:] = new_output
            
            # Calculate grad_norm_per_sample as count of tokens that are not no_change_token
            grad_norm_per_sample = (~is_no_change).sum(dim=1).float()  # (batch_size,)

            # Update best grid for each sample if current "grad norm" (change count) is lower
            improved_mask = grad_norm_per_sample < best_grad_norm  # (batch_size,)
            
            # Update best_grad_norm for improved samples
            best_grad_norm = torch.where(improved_mask, grad_norm_per_sample, best_grad_norm)
            
            # Update best_iteration for improved samples
            best_iteration = torch.where(improved_mask, torch.tensor(iteration, device=x_input.device), best_iteration)
            
            # Update best_x for improved samples
            # Expand mask to match x dimensions: (batch_size, 50)
            improved_mask_expanded = improved_mask.view(batch_size, 1).expand_as(x)
            best_x = torch.where(improved_mask_expanded, x, best_x)

        # Use the best x (with lowest change count) as final result
        x = best_x

    # Return DenoisingResult with optimized tokens and best change count per sample
    return DenoisingResult(
        optimized_output_tokens=x[:, -25:],
        best_grad_norm=best_grad_norm,
        best_iteration=best_iteration,
    )


def decode_grids(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Decode grid tokens to 5x5 grid format.

    Args:
        tokens: Tensor of shape (batch_size, num_tokens) containing one-hot encoded token values,
                where num_tokens should be a multiple of 25 (for 5x5 grids)
        vocab_size: Number of possible cell values (unused, kept for API compatibility)

    Returns:
        Tensor of shape (batch_size, num_grids, 5, 5) with token values reshaped as grids
        where num_grids = num_tokens // 25
    """
    batch_size = tokens.shape[0]
    num_tokens = tokens.shape[1]

    assert (
        num_tokens % 25 == 0
    ), f"num_tokens must be a multiple of 25, got {num_tokens}"

    num_grids = num_tokens // 25

    # Reshape to (batch_size, num_grids, 5, 5)
    decoded_grids = tokens.view(batch_size, num_grids, 5, 5)
    
    return decoded_grids


def evaluate_denoising_accuracy(
    model: TransformerModel,
    x_clean: torch.Tensor,
    task_indices: torch.Tensor,
    num_iterations: int,
) -> DenoisingResult:
    """Evaluate denoising accuracy by corrupting and denoising output grids.

    Args:
        model: The transformer model to use for computing gradients
        x_clean: Clean input tensor of shape (batch_size, 50) - tokens
        task_indices: Task indices of shape (batch_size,)
        num_iterations: Number of optimization iterations

    Returns:
        DenoisingResult containing accuracies and predicted grids
    """
    vocab_size = model.vocab_size

    # Create noised input by replacing all last 25 tokens with random noise
    x_i = x_clean.clone()
    batch_size = x_i.shape[0]
    # Generate random tokens (0-10, where 10 is mask token) for all last 25 positions
    random_tokens = torch.randint(0, 11, (batch_size, 25), device=x_clean.device)
    # Set all last 25 tokens to random values
    x_i[:, -25:] = random_tokens

    # Perform optimization to denoise
    opt_result = optimize_output_grid(
        model=model,
        x_input=x_i,
        task_indices=task_indices,
        num_iterations=num_iterations,
    )

    # Decode the optimized output grids
    assert opt_result.optimized_output_tokens is not None
    predicted_grids = decode_grids(
        opt_result.optimized_output_tokens, vocab_size
    )  # Shape: (batch_size, 1, 5, 5)
    predicted_grids = predicted_grids[:, 0, :, :]  # Shape: (batch_size, 5, 5)

    # Decode the true output grids (last 25 tokens)
    true_output_tokens = x_clean[:, -25:]  # Shape: (batch_size, 25)
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


def noise_last_25_tokens(batch: torch.Tensor, device: torch.device, vocab_size: int) -> torch.Tensor:
    """Noise the last 25 tokens by randomly replacing 0-25 of them with random values.
    
    Args:
        batch: Batch tensor of shape (batch_size, 50) - tokens
        device: Device to perform operations on
        vocab_size: Size of vocabulary for random token generation
    
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
            # Select random positions in last 25 tokens (indices 25-49)
            positions = torch.randperm(25, device=device)[:n_random] + 25
            
            # Replace with random tokens (0 to vocab_size-1)
            random_tokens = torch.randint(0, vocab_size, (int(n_random),), device=device)
            noised_batch[i, positions] = random_tokens
    
    return noised_batch


def compute_loss_for_batch(
    model: TransformerModel,
    batch: torch.Tensor,
    task_indices: torch.Tensor,
    device: torch.device,
    l1_weight: float,
) -> LossResult:
    """Compute loss for a single batch.

    Args:
        model: The transformer model
        batch: Batch of tokens, shape (batch_size, 50)
        task_indices: Task indices of shape (batch_size,)
        device: Device to compute on
        l1_weight: Weight for L1 regularization on sparse task embeddings

    Returns:
        LossResult with total loss, cross-entropy loss, and L1 loss
    """
    x = batch.to(device)  # (batch_size, 50)
    task_indices = task_indices.to(device)  # (batch_size,)
    vocab_size = model.vocab_size
    
    # no_change_token is the largest token value (vocab_size)
    no_change_token = vocab_size

    # Create noisy input by noising last 25 tokens
    xg = noise_last_25_tokens(x, device, vocab_size)

    # Create target: vocab_size (no_change_token) if xg and x have same token, else x's token
    # Shape: (batch_size, 25)
    x_output = x[:, -25:]  # (batch_size, 25) - clean output tokens
    xg_output = xg[:, -25:]  # (batch_size, 25) - noised output tokens
    
    # Where tokens are the same, target is no_change_token; otherwise target is x's token
    same_mask = (x_output == xg_output)  # (batch_size, 25)
    target = torch.where(same_mask, no_change_token, x_output)  # (batch_size, 25)

    # Forward pass - model returns logits (batch_size, 25, vocab_size+1)
    logits = model(xg, task_indices)

    # Compute cross-entropy loss (expects logits)
    ce_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size + 1), 
        target.view(-1)
    )
    
    # Compute L1 regularization on sparse task embeddings used in this batch (after sigmoid)
    task_emb_sparse = torch.sigmoid(model.task_embedding(task_indices))  # (batch_size, d_model*5)
    l1_loss = torch.abs(task_emb_sparse).mean()  # L1 norm averaged over batch and features
    
    # Combine losses
    loss = ce_loss + l1_weight * l1_loss

    return LossResult(loss=loss, ce_loss=ce_loss, l1_loss=l1_loss)


def train_epoch(
    model: TransformerModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_id_to_index: Dict[str, int],
    l1_weight: float,
) -> LossResult:
    """Train for one epoch.

    Args:
        model: The transformer model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        task_id_to_index: Mapping from task_id strings to integer indices
        l1_weight: Weight for L1 regularization on sparse task embeddings

    Returns:
        LossResult with average losses over the epoch
    """
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_l1_loss = 0.0
    num_batches = 0

    for examples in train_loader:
        # Concat input and output grids into (50,) token tensors
        batch_tensors = []
        task_ids = []
        for example in examples:
            concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)  # (50,)
            batch_tensors.append(concatenated)
            task_ids.append(example["task_id"])
        
        # Stack into batch
        batch = torch.stack(batch_tensors, dim=0)  # (batch_size, 50)
        
        # Convert task_ids to indices
        task_indices = torch.tensor([task_id_to_index[tid] for tid in task_ids], dtype=torch.long)
        
        # Compute loss
        loss_result = compute_loss_for_batch(model, batch, task_indices, device, l1_weight)

        # Backward pass
        optimizer.zero_grad()
        loss_result.loss.backward()
        optimizer.step()

        total_loss += loss_result.loss.item()
        total_ce_loss += loss_result.ce_loss.item()
        total_l1_loss += loss_result.l1_loss.item()
        num_batches += 1

    return LossResult(
        loss=torch.tensor(total_loss / num_batches),
        ce_loss=torch.tensor(total_ce_loss / num_batches),
        l1_loss=torch.tensor(total_l1_loss / num_batches)
    )


def test_epoch(
    model: TransformerModel,
    test_loader: DataLoader,
    device: torch.device,
    task_id_to_index: Dict[str, int],
    l1_weight: float,
) -> LossResult:
    """Evaluate on test set.

    Args:
        model: The transformer model
        test_loader: Test data loader
        device: Device to evaluate on
        task_id_to_index: Mapping from task_id strings to integer indices
        l1_weight: Weight for L1 regularization on sparse task embeddings

    Returns:
        LossResult with average losses over the test set
    """
    model.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_l1_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for examples in test_loader:
            # Concat input and output grids into (50,) token tensors
            batch_tensors = []
            task_ids = []
            for example in examples:
                concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)  # (50,)
                batch_tensors.append(concatenated)
                task_ids.append(example["task_id"])
            
            # Stack into batch
            batch = torch.stack(batch_tensors, dim=0)  # (batch_size, 50)
            
            # Convert task_ids to indices
            task_indices = torch.tensor([task_id_to_index[tid] for tid in task_ids], dtype=torch.long)
            
            # Compute loss
            loss_result = compute_loss_for_batch(model, batch, task_indices, device, l1_weight)

            total_loss += loss_result.loss.item()
            total_ce_loss += loss_result.ce_loss.item()
            total_l1_loss += loss_result.l1_loss.item()
            num_batches += 1

    return LossResult(
        loss=torch.tensor(total_loss / num_batches),
        ce_loss=torch.tensor(total_ce_loss / num_batches),
        l1_loss=torch.tensor(total_l1_loss / num_batches)
    )


def learning_rate_test(
    model: TransformerModel,
    train_loader: DataLoader,
    device: torch.device,
    weight_decay: float,
    task_id_to_index: Dict[str, int],
    l1_weight: float,
):
    """Test learning rate by starting at 1e-7 and doubling every batch.

    Args:
        model: The transformer model
        train_loader: Training data loader
        device: Device to train on
        weight_decay: Weight decay parameter
        task_id_to_index: Mapping from task_id strings to integer indices
        l1_weight: Weight for L1 regularization on sparse task embeddings
    """
    # Start learning rate test
    print("\nStarting learning rate test...")
    lr = 1e-7
    model.train()

    batch_count = 0
    for examples in train_loader:
        if batch_count >= 20:
            break

        # Create optimizer with current learning rate
        opt = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Prepare batch
        batch_tensors = []
        task_ids = []
        for example in examples:
            concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)  # (50,)
            batch_tensors.append(concatenated)
            task_ids.append(example["task_id"])
        
        batch = torch.stack(batch_tensors, dim=0)  # (batch_size, 50)
        task_indices = torch.tensor([task_id_to_index[tid] for tid in task_ids], dtype=torch.long)

        # Compute loss
        loss_result = compute_loss_for_batch(model, batch, task_indices, device, l1_weight)

        # Backward pass
        opt.zero_grad()
        loss_result.loss.backward()
        opt.step()

        # Print learning rate and loss
        print(f"Batch {batch_count + 1}: LR = {lr:.2e}, Loss = {loss_result.loss.item():.6f}")

        # Double learning rate for next batch
        lr *= 2
        batch_count += 1

    print("\nLearning rate test complete!")


def weight_decay_test(
    model: TransformerModel,
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    task_id_to_index: Dict[str, int],
    l1_weight: float,
):
    """Test weight decay by starting at 1e-7 and doubling every batch.

    Args:
        model: The transformer model
        train_loader: Training data loader
        device: Device to train on
        learning_rate: Learning rate parameter
        task_id_to_index: Mapping from task_id strings to integer indices
        l1_weight: Weight for L1 regularization on sparse task embeddings
    """
    # Start weight decay test
    print("\nStarting weight decay test...")
    wd = 1e-7
    model.train()

    batch_count = 0
    for examples in train_loader:
        if batch_count >= 30:
            break

        # Create optimizer with current weight decay
        opt = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=wd
        )

        # Prepare batch
        batch_tensors = []
        task_ids = []
        for example in examples:
            concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)  # (50,)
            batch_tensors.append(concatenated)
            task_ids.append(example["task_id"])
        
        batch = torch.stack(batch_tensors, dim=0)  # (batch_size, 50)
        task_indices = torch.tensor([task_id_to_index[tid] for tid in task_ids], dtype=torch.long)

        # Compute loss
        loss_result = compute_loss_for_batch(model, batch, task_indices, device, l1_weight)

        # Backward pass
        opt.zero_grad()
        loss_result.loss.backward()
        opt.step()

        # Print weight decay and loss
        print(f"Batch {batch_count + 1}: WD = {wd:.2e}, Loss = {loss_result.loss.item():.6f}")

        # Double weight decay for next batch
        wd *= 2
        batch_count += 1

    print("\nWeight decay test complete!")


def evaluate_denoising(
    model: TransformerModel,
    train_dataset: ARCTaskDataset,
    test_dataset: ARCTaskDataset,
    device: torch.device,
    eval_denoise_num_iterations: int,
    task_id_to_index: Dict[str, int],
    writer: Optional[SummaryWriter] = None,
    epoch: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate denoising accuracy on original tasks.

    Args:
        model: The transformer model
        train_dataset: Training dataset
        test_dataset: Test dataset
        device: Device to evaluate on
        eval_denoise_num_iterations: Number of optimization iterations
        task_id_to_index: Mapping from task_id strings to integer indices
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
        # Load all train tasks and concat input/output grids
        train_tensors = []
        train_task_ids = []
        for task_idx, _ in train_original_tasks:
            examples = train_dataset[task_idx]
            for example in examples:
                # Concat input and output grids into (50,) token tensor
                concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)
                train_tensors.append(concatenated)
                train_task_ids.append(example["task_id"])
        
        train_batch = torch.stack(train_tensors, dim=0).to(device)  # (num_examples, 50)
        train_task_indices = torch.tensor([task_id_to_index[tid] for tid in train_task_ids], dtype=torch.long, device=device)

        train_result = evaluate_denoising_accuracy(
            model=model,
            x_clean=train_batch,
            task_indices=train_task_indices,
            num_iterations=eval_denoise_num_iterations,
        )

        assert train_result.accuracies is not None
        train_accuracies = train_result.accuracies.cpu().numpy()

        # Load all test tasks and concat input/output grids
        test_tensors = []
        test_task_ids = []
        for task_idx, _ in test_original_tasks:
            examples = test_dataset[task_idx]
            for example in examples:
                # Concat input and output grids into (50,) token tensor
                concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)
                test_tensors.append(concatenated)
                test_task_ids.append(example["task_id"])
        
        test_batch = torch.stack(test_tensors, dim=0).to(device)  # (num_examples, 50)
        test_task_indices = torch.tensor([task_id_to_index[tid] for tid in test_task_ids], dtype=torch.long, device=device)

        test_result = evaluate_denoising_accuracy(
            model=model,
            x_clean=test_batch,
            task_indices=test_task_indices,
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
    # Train dataset uses all grids (train + test from each task)
    train_dataset_all = ARCTaskDataset(config.train_data_dir, vocab_size=config.vocab_size, grids="all")
    # Test dataset split into train and test portions
    test_dataset_train = ARCTaskDataset(config.test_data_dir, vocab_size=config.vocab_size, grids="train")
    test_dataset_test = ARCTaskDataset(config.test_data_dir, vocab_size=config.vocab_size, grids="test")
    
    # Concatenate train_dataset_all with test_dataset_train for training
    train_dataset = ConcatDataset([train_dataset_all, test_dataset_train])
    
    # Use test_dataset_test for evaluation during training
    test_dataset = test_dataset_test

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Build task_id to task_index mapping
    print("\nBuilding task_id to task_index mapping...")
    all_task_ids = set()
    
    # Collect task_ids from train_dataset_all
    for idx in range(len(train_dataset_all)):
        examples = train_dataset_all[idx]
        for example in examples:
            all_task_ids.add(example["task_id"])
    
    # Collect task_ids from test_dataset_train
    for idx in range(len(test_dataset_train)):
        examples = test_dataset_train[idx]
        for example in examples:
            all_task_ids.add(example["task_id"])
    
    # Create mapping (sorted for determinism)
    sorted_task_ids = sorted(all_task_ids)
    task_id_to_index = {task_id: idx for idx, task_id in enumerate(sorted_task_ids)}
    num_tasks = len(task_id_to_index)
    
    print(f"Found {num_tasks} unique tasks")
    print(f"Sample task_ids: {list(sorted_task_ids)[:5]}")

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
        num_tasks=num_tasks,
    ).to(device)

    # Compile model for better performance (PyTorch 2.0+)
    print("Compiling model with torch.compile...")
    model = cast(TransformerModel, torch.compile(model))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    embedding_params = sum(p.numel() for p in model.task_embedding.parameters() if p.requires_grad)
    embedding_params += sum(p.numel() for p in model.rope.parameters() if p.requires_grad)
    
    other_params = total_params - embedding_params
    
    print(f"Model has {total_params:,} trainable parameters")
    print(f"  Embedding parameters: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
    print(f"  Other parameters: {other_params:,} ({other_params/total_params*100:.1f}%)")

    # Check if running learning rate test
    if config.mode == "learning_rate_test":
        learning_rate_test(model, train_loader, device, config.weight_decay, task_id_to_index, config.l1_weight)
        return

    # Check if running weight decay test
    if config.mode == "weight_decay_test":
        weight_decay_test(model, train_loader, device, config.learning_rate, task_id_to_index, config.l1_weight)
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
        # Create evaluation datasets (test grids only)
        train_dataset_eval = ARCTaskDataset(config.train_data_dir, vocab_size=config.vocab_size, grids="test")
        test_dataset_eval = ARCTaskDataset(config.test_data_dir, vocab_size=config.vocab_size, grids="test")
        
        evaluate_denoising(
            model=model,
            train_dataset=train_dataset_eval,
            test_dataset=test_dataset_eval,
            device=device,
            eval_denoise_num_iterations=config.eval_denoise_num_iterations,
            task_id_to_index=task_id_to_index,
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
        train_result = train_epoch(model, train_loader, optimizer, device, task_id_to_index, config.l1_weight)

        # Test
        test_result = test_epoch(model, test_loader, device, task_id_to_index, config.l1_weight)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Calculate model weight norm (L2 norm of all parameters)
        total_norm_sq = torch.tensor(0.0, device=device)
        for p in model.parameters():
            total_norm_sq += p.pow(2).sum()
        model_weight_norm = torch.sqrt(total_norm_sq).item()

        # Calculate logit scale (norm of output projection layer weights)
        logit_scale = torch.sqrt(
            model.output_proj_2.weight.pow(2).sum()
        ).item()

        # Calculate layer-wise mean square using vectorized operations
        # Embedding layers
        task_emb_mean_sq = model.task_embedding.weight.pow(2).mean().item()
        token_emb_mean_sq = model.token_embedding.weight.pow(2).mean().item()
        
        # Linear layers
        output_proj_1_mean_sq = model.output_proj_1.weight.pow(2).mean().item()
        output_proj_2_mean_sq = model.output_proj_2.weight.pow(2).mean().item()
        
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

        # Calculate task embedding sparsity (percentage of near-zero values)
        task_emb_weights = model.task_embedding.weight.abs()
        # Consider values < 1e-3 as effectively zero
        sparsity_threshold = 1e-3
        task_emb_sparsity = (task_emb_weights < sparsity_threshold).float().mean().item() * 100

        # Log to console
        print(
            f"Epoch {epoch + 1}/{start_epoch + config.num_epochs} - "
            f"Train Loss: {train_result.loss.item():.6f}, Test Loss: {test_result.loss.item():.6f}, "
            f"Time: {epoch_time:.2f}s, "
            f"Weight Norm: {model_weight_norm:.4f}, "
            f"Logit Scale: {logit_scale:.4f}"
        )
        print(
            f"  Layer Mean Squares - "
            f"Task Emb: {task_emb_mean_sq:.6f}, Token Emb: {token_emb_mean_sq:.6f}, "
            f"Out Proj 1: {output_proj_1_mean_sq:.6f}, Out Proj 2: {output_proj_2_mean_sq:.6f}, "
            f"Transformer: {transformer_mean_sq:.6f}"
        )
        print(
            f"  Task Embedding Sparsity: {task_emb_sparsity:.2f}%"
        )
        print(
            f"  Loss Components - Train CE: {train_result.ce_loss.item():.6f}, Train L1: {train_result.l1_loss.item():.6f}, "
            f"Test CE: {test_result.ce_loss.item():.6f}, Test L1: {test_result.l1_loss.item():.6f}"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_result.loss.item(), epoch)
        writer.add_scalar("Loss/test", test_result.loss.item(), epoch)
        writer.add_scalar("Loss/train_ce", train_result.ce_loss.item(), epoch)
        writer.add_scalar("Loss/test_ce", test_result.ce_loss.item(), epoch)
        writer.add_scalar("Loss/train_l1", train_result.l1_loss.item(), epoch)
        writer.add_scalar("Loss/test_l1", test_result.l1_loss.item(), epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)
        writer.add_scalar("Model/weight_norm", model_weight_norm, epoch)
        writer.add_scalar("Model/logit_scale", logit_scale, epoch)
        
        # Log layer-wise mean squares
        writer.add_scalar("LayerMeanSquare/task_embedding", task_emb_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/token_embedding", token_emb_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/output_proj_1", output_proj_1_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/output_proj_2", output_proj_2_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/transformer_avg", transformer_mean_sq, epoch)
        
        # Log task embedding sparsity
        writer.add_scalar("Sparsity/task_embedding", task_emb_sparsity, epoch)

        # Evaluate denoising accuracy periodically
        if (epoch + 1) % config.eval_denoise_epoch_interval == 0:
            print(f"\nEvaluating denoising accuracy at epoch {epoch + 1}...")
            
            # Create evaluation datasets (test grids only)
            train_dataset_eval = ARCTaskDataset(config.train_data_dir, vocab_size=config.vocab_size, grids="test")
            test_dataset_eval = ARCTaskDataset(config.test_data_dir, vocab_size=config.vocab_size, grids="test")

            evaluate_denoising(
                model=model,
                train_dataset=train_dataset_eval,
                test_dataset=test_dataset_eval,
                device=device,
                eval_denoise_num_iterations=config.eval_denoise_num_iterations,
                task_id_to_index=task_id_to_index,
                writer=writer,
                epoch=epoch,
            )

            print()  # Empty line for readability

        # Save checkpoint every N epochs (configurable)
        if (epoch + 1) % config.checkpoint_save_interval == 0:
            checkpoint_path = Path(config.checkpoint_dir) / f"{config.timestamp}_epoch_{epoch + 1}_checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "train_loss": train_result.loss.item(),
                    "test_loss": test_result.loss.item(),
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
            print(f"  Checkpoint saved to {checkpoint_path}")
            
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
        l1_weight=100,
        # Data parameters
        vocab_size=11,
        # Denoising evaluation parameters
        eval_denoise_epoch_interval=1,
        eval_denoise_num_iterations=10,
        # Training parameters
        num_epochs=150,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.0,
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
