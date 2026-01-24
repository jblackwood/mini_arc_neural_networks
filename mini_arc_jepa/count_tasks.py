import csv
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List
from collections import defaultdict


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
    """Parse an ARC JSON file into an ARCTask dataclass."""
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return ARCTask(
        train=[ARCExample(**ex) for ex in data["train"]],
        test=[ARCExample(**ex) for ex in data["test"]],
    )


def main():
    # Path to the MiniARC data directory
    data_dir = Path("data/MINI-ARC/data/MiniARC")
    
    # Get all JSON files
    json_files = sorted(data_dir.glob("*.json"))
    
    print(f"Found {len(json_files)} task files")
    
    # Collect statistics - group by (num_train, num_test)
    task_counts = defaultdict(int)
    
    for json_file in json_files:
        try:
            task = parse_arc_json(json_file)
            key = (len(task.train), len(task.test))
            task_counts[key] += 1
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Convert to list of dicts for CSV output
    results = []
    for (num_train, num_test), count in sorted(task_counts.items()):
        results.append({
            "num_tasks": count,
            "num_train_examples": num_train,
            "num_test_examples": num_test,
        })
    
    # Write to CSV
    output_dir = Path("output/mini_arc_jepa")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mini_arc_task_counts.csv"
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_tasks", "num_train_examples", "num_test_examples"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nWrote statistics to {output_file}")
    print(f"Total unique (train, test) combinations: {len(results)}")


if __name__ == "__main__":
    main()
