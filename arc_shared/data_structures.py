"""Core data structures for ARC datasets."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


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


def parse_arc_json(file_path: Path) -> ARCTask:
    """Parse an ARC JSON file into an ARCTask dataclass.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        ARCTask object containing train and test examples
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    train_examples = [
        ARCExample(input=ex['input'], output=ex['output'])
        for ex in data['train']
    ]
    
    test_examples = [
        ARCExample(input=ex['input'], output=ex['output'])
        for ex in data['test']
    ]
    
    return ARCTask(train=train_examples, test=test_examples)

