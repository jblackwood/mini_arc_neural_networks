"""Core data structures for ARC datasets."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional


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


def parse_arc_json(file_path: Path) -> ARCTask:
    """Parse an ARC JSON file into an ARCTask dataclass.

    Args:
        file_path: Path to the JSON file

    Returns:
        ARCTask object containing train and test examples
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    train_examples = [
        ARCExample(input=ex["input"], output=ex["output"]) for ex in data["train"]
    ]

    test_examples = [
        ARCExample(input=ex["input"], output=ex["output"]) for ex in data["test"]
    ]

    return ARCTask(train=train_examples, test=test_examples)


def task_to_dict(task: ARCTask) -> Dict:
    """Convert an ARCTask to a dictionary suitable for JSON serialization.

    Args:
        task: ARCTask object to convert

    Returns:
        Dictionary with train and test examples, and optional metadata
    """
    result = {
        "train": [{"input": ex.input, "output": ex.output} for ex in task.train],
        "test": [{"input": ex.input, "output": ex.output} for ex in task.test],
    }

    # Add optional metadata fields if present
    if task.task_id is not None:
        result["task_id"] = task.task_id
    if task.task_type is not None:
        result["task_type"] = task.task_type
    if task.transformation is not None:
        result["transformation"] = task.transformation
    if task.color_permutation is not None:
        result["color_permutation"] = task.color_permutation

    return result


def save_task_json(task: ARCTask, file_path: Path) -> None:
    """Save an ARCTask to a JSON file.

    Args:
        task: ARCTask object to save
        file_path: Path where the JSON file should be saved
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(task_to_dict(task), f, indent=2)
        f.write("\n")
