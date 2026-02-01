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

    # Tucker decomposition core tensor dimensions
    core_dim_subject: int
    core_dim_relation: int
    core_dim_object: int
    
    # Task embedding 3D reshape dimensions
    task_embedding_3d_dim1: int
    task_embedding_3d_dim2: int
    task_embedding_3d_dim3: int

    # Training parameters
    batch_size: int
    num_epochs: int
    learning_rate: float
    task_embedding_lr: float
    weight_decay: float
    task_embedding_weight_decay: float
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
        best_change: Optional tensor of shape (batch_size,) with MSE change at best iteration
        best_iteration: Optional tensor of shape (batch_size,) with iteration of least change for each task
    """

    accuracies: Optional[torch.Tensor] = None
    predicted_grids: Optional[torch.Tensor] = None
    optimized_output_tokens: Optional[torch.Tensor] = None
    best_change: Optional[torch.Tensor] = None
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


@dataclass
class KnowledgeGraphDimensions:
    """Dimensions for the knowledge graph tensor representation.
    
    Attributes:
        num_cells: Number of cells (25 input + 25 output = 50)
        num_rows_cols: Number of row/column subjects (0-4)
        num_subjects: Total subjects = num_cells + num_rows_cols
        num_relations: Number of relation types
        num_row_col_objects: Number of row/column/integer objects (0-4)
        num_color_objects: Number of color objects (0-9)
        num_grid_type_objects: Number of grid type objects (input_grid, output_grid)
        num_objects: Total objects
    """
    num_cells: int
    num_rows_cols: int
    num_subjects: int
    num_relations: int
    num_row_col_objects: int
    num_color_objects: int
    num_grid_type_objects: int
    num_objects: int


# Subject indices:
# - Cells 0-24: input grid cells (cell_0 to cell_24)
# - Cells 25-49: output grid cells (cell_25 to cell_49)
# - Row/Col integers 50-54: integer values 0-4 (shared with row/col objects)

# Relation indices:
# 0: row
# 1: column
# 2: cell_value
# 3: grid_type
# 4: above
# 5: identity

# Object indices (organized into groups):
# Row/Column/Integer objects (0-4): shared for rows, columns, and integer relations
# Color objects (5-14): values 0-9 for cell_value relation
# Grid type objects (15-16): input_grid, output_grid

RELATION_ROW = 0
RELATION_COLUMN = 1
RELATION_CELL_VALUE = 2
RELATION_GRID_TYPE = 3
RELATION_ABOVE = 4
RELATION_IDENTITY = 5

NUM_RELATIONS = 6

# Object group offsets
OBJECT_ROW_COL_START = 0
OBJECT_COLOR_START = 5
OBJECT_GRID_TYPE_START = 15

OBJECT_INPUT_GRID = OBJECT_GRID_TYPE_START + 0
OBJECT_OUTPUT_GRID = OBJECT_GRID_TYPE_START + 1

NUM_OBJECTS = 17  # 5 (row/col/int) + 10 (colors) + 2 (grid types)


def get_knowledge_graph_dimensions() -> KnowledgeGraphDimensions:
    """Get the dimensions for the knowledge graph tensor.
    
    Returns:
        KnowledgeGraphDimensions with all dimension values
    """
    num_cells = 50  # 25 input + 25 output
    num_rows_cols = 5  # 0-4 (shared with row/col objects)
    num_subjects = num_cells + num_rows_cols  # 55
    num_relations = NUM_RELATIONS  # 6
    num_row_col_objects = 5  # 0-4 (also used for integer relations)
    num_color_objects = 10  # 0-9
    num_grid_type_objects = 2  # input, output
    num_objects = NUM_OBJECTS  # 17
    
    return KnowledgeGraphDimensions(
        num_cells=num_cells,
        num_rows_cols=num_rows_cols,
        num_subjects=num_subjects,
        num_relations=num_relations,
        num_row_col_objects=num_row_col_objects,
        num_color_objects=num_color_objects,
        num_grid_type_objects=num_grid_type_objects,
        num_objects=num_objects,
    )


def encode_arc_to_knowledge_graph(batch: torch.Tensor) -> torch.Tensor:
    """Transform ARC task tokens into a knowledge graph binary tensor.
    
    Args:
        batch: Tensor of shape (batch_size, 50) containing token values 0-9
               First 25 are input grid, last 25 are output grid
    
    Returns:
        Binary tensor of shape (batch_size, num_subjects, num_relations, num_objects)
        where:
        - num_subjects = 60 (50 cells + 10 integers)
        - num_relations = 6 (row, column, cell_value, grid_type, above, identity)
        - num_objects = 27 (5 row/col + 10 colors + 2 grid_types + 10 integers)
    """
    batch_size = batch.shape[0]
    device = batch.device
    dims = get_knowledge_graph_dimensions()
    
    # Initialize binary tensor
    kg_tensor = torch.zeros(
        batch_size, dims.num_subjects, dims.num_relations, dims.num_objects,
        device=device, dtype=torch.float32
    )
    
    # Process input grid cells (indices 0-24)
    for cell_idx in range(25):
        subject_idx = cell_idx
        row = cell_idx // 5
        col = cell_idx % 5
        
        # Row relation
        kg_tensor[:, subject_idx, RELATION_ROW, OBJECT_ROW_COL_START + row] = 1.0
        
        # Column relation
        kg_tensor[:, subject_idx, RELATION_COLUMN, OBJECT_ROW_COL_START + col] = 1.0
        
        # Cell value relation - depends on batch values
        cell_values = batch[:, cell_idx]  # (batch_size,)
        for b in range(batch_size):
            color_obj = OBJECT_COLOR_START + cell_values[b].item()
            kg_tensor[b, subject_idx, RELATION_CELL_VALUE, int(color_obj)] = 1.0
        
        # Grid type relation - input grid
        kg_tensor[:, subject_idx, RELATION_GRID_TYPE, OBJECT_INPUT_GRID] = 1.0
    
    # Process output grid cells (indices 25-49)
    for cell_idx in range(25, 50):
        subject_idx = cell_idx
        row = (cell_idx - 25) // 5
        col = (cell_idx - 25) % 5
        
        # Row relation
        kg_tensor[:, subject_idx, RELATION_ROW, OBJECT_ROW_COL_START + row] = 1.0
        
        # Column relation
        kg_tensor[:, subject_idx, RELATION_COLUMN, OBJECT_ROW_COL_START + col] = 1.0
        
        # Cell value relation - depends on batch values
        cell_values = batch[:, cell_idx]  # (batch_size,)
        for b in range(batch_size):
            color_obj = OBJECT_COLOR_START + cell_values[b].item()
            kg_tensor[b, subject_idx, RELATION_CELL_VALUE, int(color_obj)] = 1.0
        
        # Grid type relation - output grid
        kg_tensor[:, subject_idx, RELATION_GRID_TYPE, OBJECT_OUTPUT_GRID] = 1.0
    
    # Add integer subjects with above relations
    # (1, above, 0), (2, above, 1), ..., (4, above, 3)
    for i in range(1, 5):
        subject_idx = 50 + i  # Integer subject
        kg_tensor[:, subject_idx, RELATION_ABOVE, OBJECT_ROW_COL_START + (i - 1)] = 1.0
    
    # Add integer subjects with identity relations
    # (0, identity, 0), (1, identity, 1), ..., (4, identity, 4)
    for i in range(5):
        subject_idx = 50 + i  # Integer subject
        kg_tensor[:, subject_idx, RELATION_IDENTITY, OBJECT_ROW_COL_START + i] = 1.0
    
    return kg_tensor


def encode_arc_to_knowledge_graph_vectorized(batch: torch.Tensor) -> torch.Tensor:
    """Vectorized version: Transform ARC task tokens into a knowledge graph binary tensor.
    
    Args:
        batch: Tensor of shape (batch_size, 50) containing token values 0-9
               First 25 are input grid, last 25 are output grid
    
    Returns:
        Binary tensor of shape (batch_size, num_subjects, num_relations, num_objects)
    """
    batch_size = batch.shape[0]
    device = batch.device
    dims = get_knowledge_graph_dimensions()
    
    # Initialize binary tensor
    kg_tensor = torch.zeros(
        batch_size, dims.num_subjects, dims.num_relations, dims.num_objects,
        device=device, dtype=torch.float32
    )
    
    # Create cell indices for input and output grids
    input_cell_indices = torch.arange(25, device=device)
    output_cell_indices = torch.arange(25, 50, device=device)
    
    # Compute rows and columns for input grid (5x5)
    input_rows = input_cell_indices // 5
    input_cols = input_cell_indices % 5
    
    # Compute rows and columns for output grid (5x5)
    output_rows = (output_cell_indices - 25) // 5
    output_cols = (output_cell_indices - 25) % 5
    
    # Set row relations for all cells
    kg_tensor[:, input_cell_indices, RELATION_ROW, OBJECT_ROW_COL_START + input_rows] = 1.0
    kg_tensor[:, output_cell_indices, RELATION_ROW, OBJECT_ROW_COL_START + output_rows] = 1.0
    
    # Set column relations for all cells
    kg_tensor[:, input_cell_indices, RELATION_COLUMN, OBJECT_ROW_COL_START + input_cols] = 1.0
    kg_tensor[:, output_cell_indices, RELATION_COLUMN, OBJECT_ROW_COL_START + output_cols] = 1.0
    
    # Set cell_value relations - use scatter for batch-wise setting
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, 50)  # (batch_size, 50)
    cell_indices = torch.arange(50, device=device).unsqueeze(0).expand(batch_size, -1)  # (batch_size, 50)
    color_objects = OBJECT_COLOR_START + batch  # (batch_size, 50)
    
    kg_tensor[batch_indices, cell_indices, RELATION_CELL_VALUE, color_objects] = 1.0
    
    # Set grid type relations
    kg_tensor[:, :25, RELATION_GRID_TYPE, OBJECT_INPUT_GRID] = 1.0
    kg_tensor[:, 25:50, RELATION_GRID_TYPE, OBJECT_OUTPUT_GRID] = 1.0
    
    # Add integer subjects with above relations
    # (1, above, 0), (2, above, 1), ..., (4, above, 3)
    integer_above_subjects = torch.arange(51, 55, device=device)  # Integers 1-4
    integer_above_objects = torch.arange(OBJECT_ROW_COL_START, OBJECT_ROW_COL_START + 4, device=device)
    kg_tensor[:, integer_above_subjects, RELATION_ABOVE, integer_above_objects] = 1.0
    
    # Add integer subjects with identity relations
    # (0, identity, 0), (1, identity, 1), ..., (4, identity, 4)
    integer_identity_subjects = torch.arange(50, 55, device=device)  # Integers 0-4
    integer_identity_objects = torch.arange(OBJECT_ROW_COL_START, OBJECT_ROW_COL_START + 5, device=device)
    kg_tensor[:, integer_identity_subjects, RELATION_IDENTITY, integer_identity_objects] = 1.0
    
    return kg_tensor


def decode_knowledge_graph_to_triples(kg_tensor: torch.Tensor) -> List[List[Tuple[str, str, str]]]:
    """Decode a knowledge graph tensor back to (subject, relation, object) triples.
    
    Args:
        kg_tensor: Binary tensor of shape (batch_size, num_subjects, num_relations, num_objects)
    
    Returns:
        List of lists of tuples, one list per batch element, each tuple is (subject, relation, object)
    """
    batch_size = kg_tensor.shape[0]
    dims = get_knowledge_graph_dimensions()
    
    # Relation names
    relation_names = ["row", "column", "cell_value", "grid_type", "above", "identity"]
    
    result = []
    for b in range(batch_size):
        triples = []
        
        # Find all non-zero entries
        non_zero = torch.nonzero(kg_tensor[b] > 0.5, as_tuple=False)
        
        for entry in non_zero:
            subject_idx = entry[0].item()
            relation_idx = entry[1].item()
            object_idx = entry[2].item()
            
            # Decode subject
            if subject_idx < 50:
                subject = f"cell_{subject_idx}"
            else:
                subject = f"int_{subject_idx - 50}"
            
            # Decode relation
            relation = relation_names[int(relation_idx)]
            
            # Decode object based on relation type
            if relation_idx in [RELATION_ROW, RELATION_COLUMN, RELATION_ABOVE, RELATION_IDENTITY]:
                obj = str(object_idx - OBJECT_ROW_COL_START)
            elif relation_idx == RELATION_CELL_VALUE:
                obj = f"color_{object_idx - OBJECT_COLOR_START}"
            elif relation_idx == RELATION_GRID_TYPE:
                if object_idx == OBJECT_INPUT_GRID:
                    obj = "input_grid"
                else:
                    obj = "output_grid"
            else:
                obj = f"obj_{object_idx}"
            
            triples.append((subject, relation, obj))
        
        result.append(triples)
    
    return result


def mask_output_grid_relations(
    kg_tensor: torch.Tensor,
    cell_mask: torch.Tensor,
) -> torch.Tensor:
    """Mask all relations for specified output grid cells.
    
    Args:
        kg_tensor: Binary tensor of shape (batch_size, num_subjects, num_relations, num_objects)
        cell_mask: Boolean tensor of shape (batch_size, 25) indicating which output cells to mask
    
    Returns:
        Masked tensor with specified output cell relations set to 0
    """
    masked = kg_tensor.clone()
    batch_size = kg_tensor.shape[0]
    
    # Output cells are subjects 25-49
    for b in range(batch_size):
        for cell_offset in range(25):
            if cell_mask[b, cell_offset]:
                subject_idx = 25 + cell_offset  # Output grid cell index
                masked[b, subject_idx, :, :] = 0.0
    
    return masked


def mask_output_grid_relations_vectorized(
    kg_tensor: torch.Tensor,
    cell_mask: torch.Tensor,
) -> torch.Tensor:
    """Vectorized: Mask all relations for specified output grid cells.
    
    Args:
        kg_tensor: Binary tensor of shape (batch_size, num_subjects, num_relations, num_objects)
        cell_mask: Boolean tensor of shape (batch_size, 25) indicating which output cells to mask
    
    Returns:
        Masked tensor with specified output cell relations set to 0
    """
    masked = kg_tensor.clone()
    
    # Expand mask to cover all relations and objects
    # cell_mask: (batch_size, 25) -> (batch_size, 25, num_relations, num_objects)
    expanded_mask = cell_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked[:, 25:50, :, :])
    
    # Apply mask to output grid cells (subjects 25-49)
    masked[:, 25:50, :, :] = masked[:, 25:50, :, :] * (~expanded_mask).float()
    
    return masked


def get_output_cell_subject_indices() -> torch.Tensor:
    """Get subject indices for output grid cells.
    
    Returns:
        Tensor of shape (25,) with indices 25-49
    """
    return torch.arange(25, 50)


def reshape_task_embedding_to_3d(
    task_embedding: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Reshape task embedding into a small rank-3 tensor with padding.
    
    Args:
        task_embedding: Tensor of shape (batch_size, embedding_dim)
        target_shape: Tuple of (dim1, dim2, dim3) for the reshaped tensor
    
    Returns:
        Tensor of shape (batch_size, dim1, dim2, dim3) with embedding reshaped and padded
    
    Raises:
        AssertionError: If embedding would be truncated (product of target dims < embedding_dim)
    """
    batch_size = task_embedding.shape[0]
    embedding_dim = task_embedding.shape[1]
    target_size = target_shape[0] * target_shape[1] * target_shape[2]
    
    # Assert that we don't truncate
    assert target_size >= embedding_dim, (
        f"Target size {target_size} is smaller than embedding dim {embedding_dim}. "
        "Task embedding would be truncated."
    )
    
    # Pad embedding with zeros to match target size
    padded = torch.zeros(batch_size, target_size, device=task_embedding.device, dtype=task_embedding.dtype)
    padded[:, :embedding_dim] = task_embedding
    
    # Reshape to 3D
    reshaped = padded.view(batch_size, target_shape[0], target_shape[1], target_shape[2])
    
    return reshaped


def concatenate_kg_and_task_embedding(
    kg_tensor: torch.Tensor,
    task_embedding_3d: torch.Tensor,
) -> torch.Tensor:
    """Concatenate knowledge graph tensor with reshaped task embedding.
    
    The task embedding is concatenated along all three dimensions of the KG tensor.
    
    Args:
        kg_tensor: Binary tensor of shape (batch_size, num_subjects, num_relations, num_objects)
        task_embedding_3d: Tensor of shape (batch_size, te_dim1, te_dim2, te_dim3)
    
    Returns:
        Tensor of shape (batch_size, 
                        num_subjects + te_dim1, 
                        num_relations + te_dim2, 
                        num_objects + te_dim3)
    """
    batch_size = kg_tensor.shape[0]
    kg_s, kg_r, kg_o = kg_tensor.shape[1], kg_tensor.shape[2], kg_tensor.shape[3]
    te_s, te_r, te_o = task_embedding_3d.shape[1], task_embedding_3d.shape[2], task_embedding_3d.shape[3]
    
    device = kg_tensor.device
    dtype = kg_tensor.dtype
    
    # Create output tensor
    out_s = kg_s + te_s
    out_r = kg_r + te_r
    out_o = kg_o + te_o
    
    output = torch.zeros(batch_size, out_s, out_r, out_o, device=device, dtype=dtype)
    
    # Place KG tensor in the first part
    output[:, :kg_s, :kg_r, :kg_o] = kg_tensor
    
    # Place task embedding in the extended part
    output[:, kg_s:, kg_r:, kg_o:] = task_embedding_3d
    
    return output


def extract_kg_and_task_embedding(
    combined_tensor: torch.Tensor,
    kg_shape: Tuple[int, int, int],
    te_shape: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract knowledge graph tensor and task embedding from combined tensor.
    
    Args:
        combined_tensor: Tensor of shape (batch_size, kg_s + te_s, kg_r + te_r, kg_o + te_o)
        kg_shape: Tuple of (kg_subjects, kg_relations, kg_objects)
        te_shape: Tuple of (te_dim1, te_dim2, te_dim3)
    
    Returns:
        Tuple of (kg_tensor, task_embedding_3d)
    """
    kg_s, kg_r, kg_o = kg_shape
    
    kg_tensor = combined_tensor[:, :kg_s, :kg_r, :kg_o]
    task_embedding_3d = combined_tensor[:, kg_s:, kg_r:, kg_o:]
    
    return kg_tensor, task_embedding_3d


class TuckerAutoencoder(nn.Module):
    """Tucker decomposition autoencoder for ARC knowledge graph tensors."""

    def __init__(
        self,
        vocab_size: int,
        num_tasks: int,
        core_dim_subject: int,
        core_dim_relation: int,
        core_dim_object: int,
        task_embedding_3d_shape: Tuple[int, int, int],
    ):
        """Initialize the Tucker autoencoder.

        Args:
            vocab_size: Number of possible cell values (10 for ARC: 0-9 colors)
            num_tasks: Number of unique tasks for task embedding
            core_dim_subject: Dimension of core tensor along subject axis
            core_dim_relation: Dimension of core tensor along relation axis
            core_dim_object: Dimension of core tensor along object axis
            task_embedding_3d_shape: Shape for 3D task embedding (dim1, dim2, dim3)
        """
        super().__init__()

        # Store dimensions
        self.vocab_size = vocab_size
        self.core_dim_subject = core_dim_subject
        self.core_dim_relation = core_dim_relation
        self.core_dim_object = core_dim_object
        self.task_embedding_3d_shape = task_embedding_3d_shape
        
        # Derive task_embedding_dim from 3D shape
        self.task_embedding_dim = task_embedding_3d_shape[0] * task_embedding_3d_shape[1] * task_embedding_3d_shape[2]
        
        # Knowledge graph dimensions
        kg_dims = get_knowledge_graph_dimensions()
        self.kg_num_subjects = kg_dims.num_subjects
        self.kg_num_relations = kg_dims.num_relations
        self.kg_num_objects = kg_dims.num_objects
        
        # Combined dimensions (KG + task embedding 3D)
        self.total_subjects = self.kg_num_subjects + task_embedding_3d_shape[0]
        self.total_relations = self.kg_num_relations + task_embedding_3d_shape[1]
        self.total_objects = self.kg_num_objects + task_embedding_3d_shape[2]

        # Task embedding layer
        self.task_embedding = nn.Embedding(num_tasks, self.task_embedding_dim)

        # Calculate proper initialization scale for factor matrices
        # Each factor matrix operates on one dimension at a time
        subject_std = 1.0 / (self.total_subjects ** 0.5)
        relation_std = 1.0 / (self.total_relations ** 0.5)
        object_std = 1.0 / (self.total_objects ** 0.5)
        
        core_subject_std = 1.0 / (core_dim_subject ** 0.5)
        core_relation_std = 1.0 / (core_dim_relation ** 0.5)
        core_object_std = 1.0 / (core_dim_object ** 0.5)

        # Encoder factor matrices: 3 separate transformations
        # U_subject: transforms subject dimension from total_subjects to core_dim_subject
        self.encoder_U_subject = nn.Parameter(
            torch.randn(self.total_subjects, core_dim_subject) * subject_std
        )
        # U_relation: transforms relation dimension from total_relations to core_dim_relation
        self.encoder_U_relation = nn.Parameter(
            torch.randn(self.total_relations, core_dim_relation) * relation_std
        )
        # U_object: transforms object dimension from total_objects to core_dim_object
        self.encoder_U_object = nn.Parameter(
            torch.randn(self.total_objects, core_dim_object) * object_std
        )
        
        # Decoder factor matrices: 3 separate transformations (inverse of encoder)
        # V_subject: transforms subject dimension from core_dim_subject to total_subjects
        self.decoder_V_subject = nn.Parameter(
            torch.randn(core_dim_subject, self.total_subjects) * core_subject_std
        )
        # V_relation: transforms relation dimension from core_dim_relation to total_relations
        self.decoder_V_relation = nn.Parameter(
            torch.randn(core_dim_relation, self.total_relations) * core_relation_std
        )
        # V_object: transforms object dimension from core_dim_object to total_objects
        self.decoder_V_object = nn.Parameter(
            torch.randn(core_dim_object, self.total_objects) * core_object_std
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor to core tensor using Tucker decomposition.

        Args:
            x: Input tensor of shape (batch_size, total_subjects, total_relations, total_objects)

        Returns:
            Core tensor of shape (batch_size, core_dim_subject, core_dim_relation, core_dim_object)
        """
        # Apply 3 sequential factorized transformations
        # Step 1: Contract along subject dimension
        # (batch, S, R, O) x (S, core_S) -> (batch, core_S, R, O)
        z = torch.einsum('bsro,su->buro', x, self.encoder_U_subject)
        
        # Step 2: Contract along relation dimension
        # (batch, core_S, R, O) x (R, core_R) -> (batch, core_S, core_R, O)
        z = torch.einsum('buro,rv->buvo', z, self.encoder_U_relation)
        
        # Step 3: Contract along object dimension
        # (batch, core_S, core_R, O) x (O, core_O) -> (batch, core_S, core_R, core_O)
        z = torch.einsum('buvo,ow->buvw', z, self.encoder_U_object)
        
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode core tensor back to original tensor space.

        Args:
            z: Core tensor of shape (batch_size, core_dim_subject, core_dim_relation, core_dim_object)

        Returns:
            Reconstructed tensor of shape (batch_size, total_subjects, total_relations, total_objects)
        """
        # Apply 3 sequential factorized transformations (inverse of encoder)
        # Step 1: Expand along subject dimension
        # (batch, core_S, core_R, core_O) x (core_S, S) -> (batch, S, core_R, core_O)
        x = torch.einsum('buvw,us->bsvw', z, self.decoder_V_subject)
        
        # Step 2: Expand along relation dimension
        # (batch, S, core_R, core_O) x (core_R, R) -> (batch, S, R, core_O)
        x = torch.einsum('bsvw,vr->bsrw', x, self.decoder_V_relation)
        
        # Step 3: Expand along object dimension
        # (batch, S, R, core_O) x (core_O, O) -> (batch, S, R, O)
        x = torch.einsum('bsrw,wo->bsro', x, self.decoder_V_object)
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Tucker autoencoder.

        Args:
            x: Input tensor of shape (batch_size, total_subjects, total_relations, total_objects)

        Returns:
            Reconstructed tensor of shape (batch_size, total_subjects, total_relations, total_objects)
        """
        z = self.encode(x)
        output = self.decode(z)
        return output


def optimize_output_grid(
    model: TuckerAutoencoder,
    batch: torch.Tensor,
    task_indices: torch.Tensor,
    num_iterations: int,
) -> DenoisingResult:
    """Optimize the output grid by iteratively denoising with the Tucker autoencoder.

    Initializes output grid relations to zero and iteratively
    updates them based on model output.

    Args:
        model: The Tucker autoencoder to use for denoising
        batch: Input tensor of shape (batch_size, 50) - clean tokens for input and output grids
        task_indices: Task indices of shape (batch_size,)
        num_iterations: Number of optimization iterations

    Returns:
        DenoisingResult with optimized_output_tokens (decoded to 0-9 values) and best_iteration
    """
    batch_size = batch.shape[0]
    vocab_size = model.vocab_size
    device = batch.device

    with torch.no_grad():
        # Encode full batch to knowledge graph (with actual output values)
        kg_tensor_full = encode_arc_to_knowledge_graph_vectorized(batch)
        
        # Mask all output grid cells (subjects 25-49)
        all_output_mask = torch.ones(batch_size, 25, dtype=torch.bool, device=device)
        kg_tensor = mask_output_grid_relations_vectorized(kg_tensor_full, all_output_mask)
        
        # Lookup task embedding and reshape to 3D
        task_emb = model.task_embedding(task_indices)  # (batch_size, task_embedding_dim)
        task_emb_3d = reshape_task_embedding_to_3d(task_emb, model.task_embedding_3d_shape)
        
        # Combine KG and task embedding
        x = concatenate_kg_and_task_embedding(kg_tensor, task_emb_3d)
        
        # Track best result with lowest change between iterations
        best_change = torch.full((batch_size,), float("inf"), device=device)
        best_iteration = torch.zeros((batch_size,), dtype=torch.long, device=device)
        best_kg_output = kg_tensor.clone()

        for iteration in range(num_iterations):
            # Get model's denoised output
            denoised = model(x)  # (batch_size, total_S, total_R, total_O)
            
            # Extract KG part from denoised output
            denoised_kg, _ = extract_kg_and_task_embedding(
                denoised,
                kg_shape=(model.kg_num_subjects, model.kg_num_relations, model.kg_num_objects),
                te_shape=model.task_embedding_3d_shape,
            )
            
            # Calculate MSE change in output grid region (subjects 25-49)
            new_output_relations = denoised_kg[:, 25:50, :, :]
            old_output_relations = kg_tensor[:, 25:50, :, :]
            change_per_sample = (new_output_relations - old_output_relations).pow(2).mean(dim=(1, 2, 3))

            # Update best result if current change is lower (more stable)
            improved_mask = change_per_sample < best_change

            best_change = torch.where(improved_mask, change_per_sample, best_change)
            best_iteration = torch.where(
                improved_mask,
                torch.tensor(iteration, device=device),
                best_iteration,
            )

            # Update best_kg_output for improved samples
            for b in range(batch_size):
                if improved_mask[b]:
                    best_kg_output[b] = denoised_kg[b]

            # Update kg_tensor with denoised output for next iteration
            kg_tensor = denoised_kg.clone()
            
            # Keep input grid fixed (subjects 0-24 from original)
            kg_tensor[:, :25, :, :] = kg_tensor_full[:, :25, :, :]
            
            # Reconstruct combined tensor for next iteration
            x = concatenate_kg_and_task_embedding(kg_tensor, task_emb_3d)

        # Decode output grid cell values from best_kg_output
        # For each output cell (subjects 25-49), look at cell_value relation
        # and find which color object is highest
        output_cell_values = best_kg_output[:, 25:50, 2, OBJECT_COLOR_START:OBJECT_COLOR_START + 10]  # (batch_size, 25, 10)
        decoded_tokens = output_cell_values.argmax(dim=-1)  # (batch_size, 25)

    return DenoisingResult(
        optimized_output_tokens=decoded_tokens,
        best_change=best_change,
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
    model: TuckerAutoencoder,
    x_clean: torch.Tensor,
    task_indices: torch.Tensor,
    num_iterations: int,
) -> DenoisingResult:
    """Evaluate denoising accuracy by denoising output grids from zero initialization.

    Args:
        model: The Tucker autoencoder model
        x_clean: Clean input tensor of shape (batch_size, 50) - tokens
        task_indices: Task indices of shape (batch_size,)
        num_iterations: Number of optimization iterations

    Returns:
        DenoisingResult containing accuracies and predicted grids
    """
    vocab_size = model.vocab_size

    # Perform optimization to denoise (output grid initialized as zero)
    opt_result = optimize_output_grid(
        model=model,
        batch=x_clean,
        task_indices=task_indices,
        num_iterations=num_iterations,
    )

    # Decode the optimized output grids (already decoded in optimize_output_grid)
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


def compute_loss_for_batch(
    model: TuckerAutoencoder,
    batch: torch.Tensor,
    task_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute loss for a single batch using Tucker denoising autoencoder.

    Encodes ARC task to knowledge graph tensor, randomly masks 0-25 output grid cells,
    and trains the model to reconstruct the clean knowledge graph.

    Args:
        model: The Tucker autoencoder
        batch: Batch of tokens, shape (batch_size, 50)
        task_indices: Task indices of shape (batch_size,)
        device: Device to compute on

    Returns:
        Reconstruction loss (scalar tensor)
    """
    batch = batch.to(device)  # (batch_size, 50)
    task_indices = task_indices.to(device)  # (batch_size,)
    batch_size = batch.shape[0]

    # Encode to knowledge graph tensor
    kg_tensor_clean = encode_arc_to_knowledge_graph_vectorized(batch)  # (batch_size, 60, 6, 27)

    # Lookup task embeddings and reshape to 3D
    task_emb = model.task_embedding(task_indices)  # (batch_size, task_embedding_dim)
    task_emb_3d = reshape_task_embedding_to_3d(task_emb, model.task_embedding_3d_shape)
    
    # Combine KG and task embedding for clean target
    x_clean = concatenate_kg_and_task_embedding(kg_tensor_clean, task_emb_3d)

    # Randomly pick 0-25 output cells to mask
    num_cells_to_mask = torch.randint(0, 26, (batch_size,), device=device)  # (batch_size,)
    
    # Create mask for output cells (batch_size, 25)
    rand_values = torch.rand(batch_size, 25, device=device)
    cell_mask = rand_values.argsort(dim=1) < num_cells_to_mask.unsqueeze(1)
    
    # Mask output grid cells
    kg_tensor_masked = mask_output_grid_relations_vectorized(kg_tensor_clean, cell_mask)
    
    # Combine masked KG and task embedding
    x_masked = concatenate_kg_and_task_embedding(kg_tensor_masked, task_emb_3d)

    # Forward pass - model outputs denoised reconstruction
    predicted = model(x_masked)

    # Compute reconstruction MSE loss
    reconstruction_loss = (predicted - x_clean).pow(2).mean()

    return reconstruction_loss


def train_epoch(
    model: TuckerAutoencoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_id_to_index: Dict[str, int],
) -> float:
    """Train for one epoch.

    Args:
        model: The Tucker autoencoder model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        task_id_to_index: Mapping from task_id strings to integer indices

    Returns:
        Average reconstruction loss
    """
    model.train()
    loss_sum = 0.0
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
        loss = compute_loss_for_batch(model, batch, task_indices, device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.item()
        num_batches += 1

    return loss_sum / num_batches


def test_epoch(
    model: TuckerAutoencoder,
    test_loader: DataLoader,
    device: torch.device,
    task_id_to_index: Dict[str, int],
) -> float:
    """Evaluate on test set.

    Args:
        model: The Tucker autoencoder model
        test_loader: Test data loader
        device: Device to evaluate on
        task_id_to_index: Mapping from task_id strings to integer indices

    Returns:
        Average reconstruction loss
    """
    model.eval()
    loss_sum = 0.0
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
            loss = compute_loss_for_batch(model, batch, task_indices, device)

            loss_sum += loss.item()
            num_batches += 1

    return loss_sum / num_batches


def learning_rate_test(
    model: TuckerAutoencoder,
    train_loader: DataLoader,
    device: torch.device,
    weight_decay: float,
    task_id_to_index: Dict[str, int],
):
    """Test learning rate by starting at 1e-7 and doubling every batch.

    Args:
        model: The Tucker autoencoder model
        train_loader: Training data loader
        device: Device to train on
        weight_decay: Weight decay parameter
        task_id_to_index: Mapping from task_id strings to integer indices
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
        loss = compute_loss_for_batch(model, batch, task_indices, device)

        # Backward pass
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # Print learning rate and loss
        print(f"Batch {batch_count + 1}: LR = {lr:.2e}, Loss = {loss.item():.6f}")

        # Double learning rate for next batch
        lr *= 2
        batch_count += 1

    print("\nLearning rate test complete!")


def weight_decay_test(
    model: TuckerAutoencoder,
    train_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    task_id_to_index: Dict[str, int],
):
    """Test weight decay by starting at 1e-7 and doubling every batch.

    Args:
        model: The Tucker autoencoder model
        train_loader: Training data loader
        device: Device to train on
        learning_rate: Learning rate parameter
        task_id_to_index: Mapping from task_id strings to integer indices
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
        loss = compute_loss_for_batch(model, batch, task_indices, device)

        # Backward pass
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        print(f"Batch {batch_count + 1}: WD = {wd:.2e}, Loss = {loss.item():.6f}")

        # Double weight decay for next batch
        wd *= 2
        batch_count += 1

    print("\nWeight decay test complete!")


def evaluate_denoising(
    model: TuckerAutoencoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    eval_denoise_num_iterations: int,
    task_id_to_index: Dict[str, int],
    writer: Optional[SummaryWriter],
    epoch: Optional[int],
) -> Tuple[float, float]:
    """Evaluate denoising accuracy on all tasks.

    Args:
        model: The Tucker autoencoder model
        train_loader: Training dataloader
        test_loader: Test dataloader
        device: Device to evaluate on
        eval_denoise_num_iterations: Number of optimization iterations
        task_id_to_index: Mapping from task_id strings to integer indices
        writer: Optional tensorboard writer for logging
        epoch: Optional epoch number for logging

    Returns:
        Tuple of (average_train_accuracy, average_test_accuracy)
    """
    # Evaluate denoising accuracy
    eval_start_time = time.time()

    # Evaluate train tasks
    model.eval()
    with torch.no_grad():
        # Process train tasks batch by batch
        train_results = []
        for examples in train_loader:
            # Prepare batch
            batch_tensors = []
            batch_task_ids = []
            for example in examples:
                # Concat input and output grids into (50,) token tensor
                concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)
                batch_tensors.append(concatenated)
                batch_task_ids.append(example["task_id"])
            
            batch = torch.stack(batch_tensors, dim=0).to(device)  # (batch_size, 50)
            batch_task_indices = torch.tensor([task_id_to_index[tid] for tid in batch_task_ids], dtype=torch.long, device=device)

            # Evaluate this batch
            batch_result = evaluate_denoising_accuracy(
                model=model,
                x_clean=batch,
                task_indices=batch_task_indices,
                num_iterations=eval_denoise_num_iterations,
            )
            train_results.append(batch_result)

        # Process test tasks batch by batch
        test_results = []
        for examples in test_loader:
            # Prepare batch
            batch_tensors = []
            batch_task_ids = []
            for example in examples:
                # Concat input and output grids into (50,) token tensor
                concatenated = torch.cat([example["input_grid"], example["output_grid"]], dim=0)
                batch_tensors.append(concatenated)
                batch_task_ids.append(example["task_id"])
            
            batch = torch.stack(batch_tensors, dim=0).to(device)  # (batch_size, 50)
            batch_task_indices = torch.tensor([task_id_to_index[tid] for tid in batch_task_ids], dtype=torch.long, device=device)

            # Evaluate this batch
            batch_result = evaluate_denoising_accuracy(
                model=model,
                x_clean=batch,
                task_indices=batch_task_indices,
                num_iterations=eval_denoise_num_iterations,
            )
            test_results.append(batch_result)

    # Aggregate train results
    train_accuracies = np.concatenate([result.accuracies.cpu().numpy() for result in train_results])
    train_best_iterations = np.concatenate([result.best_iteration.cpu().numpy() for result in train_results])
    train_best_changes = np.concatenate([result.best_change.cpu().numpy() for result in train_results])

    # Aggregate test results
    test_accuracies = np.concatenate([result.accuracies.cpu().numpy() for result in test_results])
    test_best_iterations = np.concatenate([result.best_iteration.cpu().numpy() for result in test_results])
    test_best_changes = np.concatenate([result.best_change.cpu().numpy() for result in test_results])

    # Compute average accuracies
    avg_train_acc = np.mean(train_accuracies) if len(train_accuracies) > 0 else 0.0
    avg_test_acc = np.mean(test_accuracies) if len(test_accuracies) > 0 else 0.0

    # Compute % of grids with 100% accuracy
    train_perfect_pct = (np.sum(train_accuracies == 1.0) / len(train_accuracies) * 100) if len(train_accuracies) > 0 else 0.0
    test_perfect_pct = (np.sum(test_accuracies == 1.0) / len(test_accuracies) * 100) if len(test_accuracies) > 0 else 0.0

    # Get max iteration across all samples
    max_train_iter = int(np.max(train_best_iterations)) if len(train_best_iterations) > 0 else 0
    max_test_iter = int(np.max(test_best_iterations)) if len(test_best_iterations) > 0 else 0

    # Compute average and std of best iteration
    avg_train_iter = np.mean(train_best_iterations) if len(train_best_iterations) > 0 else 0.0
    std_train_iter = np.std(train_best_iterations) if len(train_best_iterations) > 0 else 0.0
    avg_test_iter = np.mean(test_best_iterations) if len(test_best_iterations) > 0 else 0.0
    std_test_iter = np.std(test_best_iterations) if len(test_best_iterations) > 0 else 0.0

    # Compute average best change (MSE)
    avg_train_change = np.mean(train_best_changes) if len(train_best_changes) > 0 else 0.0
    avg_test_change = np.mean(test_best_changes) if len(test_best_changes) > 0 else 0.0

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
        print(
            f"  Train Avg Change: {avg_train_change:.6f}, "
            f"Test Avg Change: {avg_test_change:.6f}"
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
        print(
            f"Train Avg Change: {avg_train_change:.6f}, "
            f"Test Avg Change: {avg_test_change:.6f}"
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
        writer.add_scalar(
            f"DenoiseBestChange/train", avg_train_change, epoch
        )
        writer.add_scalar(
            f"DenoiseBestChange/test", avg_test_change, epoch
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

    # Task embedding 3D shape for combining with knowledge graph
    task_embedding_3d_shape = (
        config.task_embedding_3d_dim1,
        config.task_embedding_3d_dim2,
        config.task_embedding_3d_dim3,
    )

    # Create model
    model = TuckerAutoencoder(
        vocab_size=config.vocab_size,
        num_tasks=num_tasks,
        core_dim_subject=config.core_dim_subject,
        core_dim_relation=config.core_dim_relation,
        core_dim_object=config.core_dim_object,
        task_embedding_3d_shape=task_embedding_3d_shape,
    ).to(device)

    # Compile model for better performance (PyTorch 2.0+)
    print("Compiling model with torch.compile...")
    model = cast(TuckerAutoencoder, torch.compile(model))

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    embedding_params = sum(p.numel() for p in model.task_embedding.parameters() if p.requires_grad)
    
    other_params = total_params - embedding_params
    
    print(f"Model has {total_params:,} trainable parameters")
    print(f"  Task embedding parameters: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
    print(f"  Other parameters: {other_params:,} ({other_params/total_params*100:.1f}%)")

    # Check if running learning rate test
    if config.mode == "learning_rate_test":
        learning_rate_test(model, train_loader, device, config.weight_decay, task_id_to_index)
        return

    # Check if running weight decay test
    if config.mode == "weight_decay_test":
        weight_decay_test(model, train_loader, device, config.learning_rate, task_id_to_index)
        return

    # Create optimizer with separate learning rate and weight decay for task embeddings
    task_embedding_params = list(model.task_embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if 'task_embedding' not in n]
    
    optimizer = torch.optim.AdamW([
        {'params': task_embedding_params, 'lr': config.task_embedding_lr, 'weight_decay': config.task_embedding_weight_decay},
        {'params': other_params, 'lr': config.learning_rate, 'weight_decay': config.weight_decay}
    ])

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
        
        # Create dataloaders
        train_loader_eval = DataLoader(
            train_dataset_eval,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=arc_collate_fn,
        )
        test_loader_eval = DataLoader(
            test_dataset_eval,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=arc_collate_fn,
        )
        
        evaluate_denoising(
            model=model,
            train_loader=train_loader_eval,
            test_loader=test_loader_eval,
            device=device,
            eval_denoise_num_iterations=config.eval_denoise_num_iterations,
            task_id_to_index=task_id_to_index,
            writer=None,
            epoch=None,
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
        train_loss = train_epoch(model, train_loader, optimizer, device, task_id_to_index)

        # Test
        test_loss = test_epoch(model, test_loader, device, task_id_to_index)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Calculate model weight norm (L2 norm of all parameters)
        total_norm_sq = torch.tensor(0.0, device=device)
        for p in model.parameters():
            total_norm_sq += p.pow(2).sum()
        model_weight_norm = torch.sqrt(total_norm_sq).item()

        # Calculate layer-wise mean square
        task_emb_mean_sq = model.task_embedding.weight.pow(2).mean().item()
        # Average mean square across all encoder factor matrices
        encoder_weight_mean_sq = (
            model.encoder_U_subject.pow(2).mean().item() +
            model.encoder_U_relation.pow(2).mean().item() +
            model.encoder_U_object.pow(2).mean().item()
        ) / 3.0
        # Average mean square across all decoder factor matrices
        decoder_weight_mean_sq = (
            model.decoder_V_subject.pow(2).mean().item() +
            model.decoder_V_relation.pow(2).mean().item() +
            model.decoder_V_object.pow(2).mean().item()
        ) / 3.0

        # Log to console
        print(
            f"Epoch {epoch + 1}/{start_epoch + config.num_epochs} - "
            f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
            f"Time: {epoch_time:.2f}s, "
            f"Weight Norm: {model_weight_norm:.4f}"
        )
        print(
            f"  Layer Mean Squares - "
            f"Task Emb: {task_emb_mean_sq:.6f}, "
            f"Encoder: {encoder_weight_mean_sq:.6f}, "
            f"Decoder: {decoder_weight_mean_sq:.6f}"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)
        writer.add_scalar("Model/weight_norm", model_weight_norm, epoch)
        
        # Log layer-wise mean squares
        writer.add_scalar("LayerMeanSquare/task_embedding", task_emb_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/encoder_weight", encoder_weight_mean_sq, epoch)
        writer.add_scalar("LayerMeanSquare/decoder_weight", decoder_weight_mean_sq, epoch)

        # Evaluate denoising accuracy periodically
        if (epoch + 1) % config.eval_denoise_epoch_interval == 0:
            print(f"\nEvaluating denoising accuracy at epoch {epoch + 1}...")
            
            # Create evaluation datasets (test grids only)
            train_dataset_eval = ARCTaskDataset(config.train_data_dir, vocab_size=config.vocab_size, grids="test")
            test_dataset_eval = ARCTaskDataset(config.test_data_dir, vocab_size=config.vocab_size, grids="test")
            
            # Create dataloaders
            train_loader_eval = DataLoader(
                train_dataset_eval,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=arc_collate_fn,
            )
            test_loader_eval = DataLoader(
                test_dataset_eval,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=arc_collate_fn,
            )

            evaluate_denoising(
                model=model,
                train_loader=train_loader_eval,
                test_loader=test_loader_eval,
                device=device,
                eval_denoise_num_iterations=config.eval_denoise_num_iterations,
                task_id_to_index=task_id_to_index,
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
                        "vocab_size": config.vocab_size,
                        "core_dim_subject": config.core_dim_subject,
                        "core_dim_relation": config.core_dim_relation,
                        "core_dim_object": config.core_dim_object,
                        "task_embedding_3d_dim1": config.task_embedding_3d_dim1,
                        "task_embedding_3d_dim2": config.task_embedding_3d_dim2,
                        "task_embedding_3d_dim3": config.task_embedding_3d_dim3,
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
                "vocab_size": config.vocab_size,
                "core_dim_subject": config.core_dim_subject,
                "core_dim_relation": config.core_dim_relation,
                "core_dim_object": config.core_dim_object,
                "task_embedding_3d_dim1": config.task_embedding_3d_dim1,
                "task_embedding_3d_dim2": config.task_embedding_3d_dim2,
                "task_embedding_3d_dim3": config.task_embedding_3d_dim3,
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
        output_dir=Path("output/mini_arc_tucker"),
        test_ratio=0.2,
        random_seed=42,
        max_augmentations=500,
        # Tucker decomposition core tensor dimensions
        core_dim_subject=10,
        core_dim_relation=10,
        core_dim_object=10,
        # Task embedding 3D reshape dimensions
        task_embedding_3d_dim1=4,
        task_embedding_3d_dim2=4,
        task_embedding_3d_dim3=4,
        # Data parameters
        vocab_size=10,
        # Denoising evaluation parameters
        eval_denoise_epoch_interval=5,
        eval_denoise_num_iterations=10,
        # Training parameters
        num_epochs=300,
        batch_size=32,
        learning_rate=1e-4,
        task_embedding_lr=1e-2,
        weight_decay=0,
        task_embedding_weight_decay=0,
        mode="train",
        checkpoint_save_interval=50,
        # Google Drive location for Colab
        google_drive_dir="/content/drive/MyDrive/sparse_arc",
        # Optional: Load existing model to continue training
        load_model_path=None,
    )

    # Print configuration
    print("Configuration:")
    pprint.pprint(asdict(config), width=100, sort_dicts=False)
    print()

    # Print knowledge graph dimensions
    kg_dims = get_knowledge_graph_dimensions()
    print("Knowledge Graph Dimensions:")
    pprint.pprint(asdict(kg_dims), width=100, sort_dicts=False)
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
