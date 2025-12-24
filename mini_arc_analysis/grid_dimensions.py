"""Analyze grid dimensions in MiniARC dataset.

This module provides functionality to parse JSON files from the MiniARC dataset
and extract grid dimensions to create a distribution of grid sizes.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

from arc_shared import parse_arc_json


@dataclass
class GridDimension:
    """Represents a grid dimension with width, height, and count."""

    grid_width: int
    grid_height: int
    num_grids: int


def get_grid_dimensions(grid: List[List[int]]) -> tuple[int, int]:
    """Get the dimensions of a grid.

    Args:
        grid: 2D list representing the grid

    Returns:
        Tuple of (width, height)
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    return width, height


def analyze_mini_arc_dimensions(data_dir: Path) -> List[GridDimension]:
    """Analyze grid dimensions across all MiniARC tasks.

    Args:
        data_dir: Path to the MiniARC data directory

    Returns:
        List of GridDimension objects sorted by width, then height
    """
    dimension_counter = Counter()

    # Iterate through all JSON files in the directory
    for json_file in data_dir.glob("*.json"):
        try:
            task = parse_arc_json(json_file)

            # Process all grids (both input and output from train and test)
            for example in task.train + task.test:
                for grid in [example.input, example.output]:
                    width, height = get_grid_dimensions(grid)
                    dimension_counter[(width, height)] += 1

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue

    # Convert to list of GridDimension objects
    grid_dimensions = [
        GridDimension(grid_width=width, grid_height=height, num_grids=count)
        for (width, height), count in dimension_counter.items()
    ]

    # Sort by width, then height
    grid_dimensions.sort(key=lambda x: (x.grid_width, x.grid_height))

    return grid_dimensions


def save_to_csv(grid_dimensions: List[GridDimension], output_path: Path) -> None:
    """Save grid dimensions to a CSV file.

    Args:
        grid_dimensions: List of GridDimension objects
        output_path: Path to the output CSV file
    """
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Write header
        f.write("grid_width,grid_height,num_grids\n")

        # Write data rows
        for dim in grid_dimensions:
            f.write(f"{dim.grid_width},{dim.grid_height},{dim.num_grids}\n")


def main():
    """Main function to analyze MiniARC grid dimensions."""
    data_dir = Path("data/MINI-ARC/data/MiniARC")
    output_path = Path("output/mini_arc_analysis/grid_dimensions.csv")

    print(f"Analyzing grid dimensions in {data_dir}...")
    grid_dimensions = analyze_mini_arc_dimensions(data_dir)

    print(f"\nFound {len(grid_dimensions)} unique grid dimensions")
    print(f"Total grids analyzed: {sum(dim.num_grids for dim in grid_dimensions)}")

    print(f"\nSaving results to {output_path}...")
    save_to_csv(grid_dimensions, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
