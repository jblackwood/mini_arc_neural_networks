"""PyTorch dataset for ARC tasks."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from arc_shared import parse_arc_json


class GridDict(TypedDict):
    """Type definition for a grid dictionary.

    Attributes:
        task_idx: Vocabulary index of the task_id as a tensor
        input: 2D tensor representing the input grid
        output: 2D tensor representing the output grid
    """

    task_idx: torch.Tensor
    input: torch.Tensor
    output: torch.Tensor


class ARCTaskDataset(Dataset):
    """PyTorch dataset for ARC tasks.

    Args:
        folder_path: Path to folder containing task JSON files
        grid_type: Either "train" or "test" to filter which examples to return
    """

    def __init__(self, folder_path: str, grid_type: Literal["train", "test"]):
        self.folder_path = Path(folder_path)
        self.grid_type = grid_type

        # Load all task files and build vocabulary
        self.task_files = sorted(self.folder_path.glob("*.json"))
        self.task_id_to_idx: Dict[str, int] = {}

        # Build vocabulary and collect examples
        for task_file in self.task_files:
            # Parse task using ARCTask dataclass
            task_data = parse_arc_json(task_file)

            assert task_data.task_id is not None, f"Missing task_id in {task_file}"
            task_id = task_data.task_id

            # Add task_id to vocabulary if not present
            if task_id not in self.task_id_to_idx:
                self.task_id_to_idx[task_id] = len(self.task_id_to_idx)

    def __len__(self) -> int:
        return len(self.task_files)

    def __getitem__(self, idx: int) -> List[GridDict]:
        """Return a list of grid dictionaries for a task.

        Args:
            idx: Index of the task file

        Returns:
            List of GridDict, each containing 'task_idx', 'input' and 'output' grids as tensors
        """
        task_file = self.task_files[idx]
        task_data = parse_arc_json(task_file)

        # Get task_idx from vocabulary
        assert task_data.task_id is not None
        task_idx = self.task_id_to_idx[task_data.task_id]

        # Get examples based on grid_type
        examples = task_data.train if self.grid_type == "train" else task_data.test

        # Convert each example to a GridDict
        grid_dicts: List[GridDict] = []
        for example in examples:
            grid_dict: GridDict = {
                "task_idx": torch.tensor(task_idx, dtype=torch.long),
                "input": torch.tensor(example.input, dtype=torch.long),
                "output": torch.tensor(example.output, dtype=torch.long),
            }
            grid_dicts.append(grid_dict)

        return grid_dicts

    @property
    def vocab_size(self) -> int:
        """Return the size of the task vocabulary."""
        return len(self.task_id_to_idx)


def flatten_collate(batch: List[List[GridDict]]) -> List[GridDict]:
    """Custom collate function that flattens the batch.

    Flattens the nested list structure into a single list of GridDict objects.

    Args:
        batch: List of items from __getitem__, where each item is a List[GridDict]

    Returns:
        Flattened list of GridDict objects
    """
    flattened = []
    for task_examples in batch:
        flattened.extend(task_examples)
    return flattened


class ARCTransformer(nn.Module):
    """Transformer model for ARC tasks using BERT-style masking.

    Args:
        num_tasks: Number of unique tasks in vocabulary
        num_colors: Number of unique colors (0-9, so 10)
        d_model: Dimension of embeddings and transformer
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        max_seq_len: Maximum sequence length for positional embeddings
    """

    def __init__(
        self,
        num_tasks: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        max_seq_len: int = 51,
        num_colors: int = 10,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_colors = num_colors

        # MASK token is added as an extra embedding (index = num_colors)
        self.mask_token_idx = num_colors

        # Embedding layers
        self.task_embedding = nn.Embedding(num_tasks, d_model)
        # Grid embedding now includes the MASK token (num_colors + 1 total)
        self.grid_embedding = nn.Embedding(num_colors + 1, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection to predict colors
        self.output_proj = nn.Linear(d_model, num_colors)

    def forward(
        self,
        task_idx: torch.Tensor,
        input_grid: torch.Tensor,
    ):
        """Forward pass using BERT-style masking.

        Args:
            task_idx: (batch_size,) task indices
            input_grid: (batch_size, H, W) input grids

        Returns:
            logits: (batch_size, H*W, num_colors) predictions for output grid
        """
        batch_size = task_idx.shape[0]
        H, W = input_grid.shape[1], input_grid.shape[2]

        # Flatten input grid
        input_flat = input_grid.view(batch_size, -1)  # (batch_size, H*W)

        # Get embeddings
        task_emb = self.task_embedding(task_idx).unsqueeze(
            1
        )  # (batch_size, 1, d_model)
        input_emb = self.grid_embedding(input_flat)  # (batch_size, H*W, d_model)

        # Create mask tokens for all output positions (same size as input)
        mask_tokens = torch.full(
            (batch_size, H * W),
            self.mask_token_idx,
            dtype=torch.long,
            device=input_grid.device,
        )
        output_emb = self.grid_embedding(mask_tokens)  # (batch_size, H*W, d_model)

        # Concatenate: [task, input, masked_output]
        seq = torch.cat(
            [task_emb, input_emb, output_emb], dim=1
        )  # (batch_size, 1+2*H*W, d_model)
        seq_len = seq.shape[1]

        # Add positional embeddings
        positions = (
            torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        )
        pos_emb = self.pos_embedding(positions)
        seq = seq + pos_emb

        # No attention mask needed - BERT style allows all positions to attend to each other
        encoded = self.transformer(seq)  # (batch_size, seq_len, d_model)

        # Extract output portion and project
        output_start = 1 + H * W
        output_encoded = encoded[:, output_start:, :]  # (batch_size, H*W, d_model)
        logits = self.output_proj(output_encoded)  # (batch_size, H*W, num_colors)

        return logits


def train_epoch(
    model: ARCTransformer, dataloader: DataLoader, optimizer, criterion, device
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Stack batch items
        task_indices = torch.stack([item["task_idx"] for item in batch]).to(device)
        input_grids = torch.stack([item["input"] for item in batch]).to(device)
        output_grids = torch.stack([item["output"] for item in batch]).to(device)

        # Forward pass (model only needs task_idx and input_grid)
        logits = model(task_indices, input_grids)

        # Compute loss
        _, _, num_colors = logits.shape
        logits_flat = logits.view(-1, num_colors)
        targets_flat = output_grids.view(-1)
        loss = criterion(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # print(
        #     f"  Train batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}"
        # )

    return total_loss / num_batches


def test_epoch(model: ARCTransformer, dataloader: DataLoader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Stack batch items
            task_indices = torch.stack([item["task_idx"] for item in batch]).to(device)
            input_grids = torch.stack([item["input"] for item in batch]).to(device)
            output_grids = torch.stack([item["output"] for item in batch]).to(device)

            # Forward pass (model only needs task_idx and input_grid)
            logits = model(task_indices, input_grids)

            # Compute loss
            _, _, num_colors = logits.shape
            logits_flat = logits.view(-1, num_colors)
            targets_flat = output_grids.view(-1)
            loss = criterion(logits_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

            # print(
            #     f"  Test batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}"
            # )

    return total_loss / num_batches


def main():
    """Train and test the ARC transformer model."""
    # Hyperparameters
    folder_path = "output/mini_arc_analysis/train"
    batch_size = 512
    num_epochs = 100
    learning_rate = 1e-4

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create datasets
    train_dataset = ARCTaskDataset(folder_path=folder_path, grid_type="train")
    test_dataset = ARCTaskDataset(folder_path=folder_path, grid_type="test")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=flatten_collate,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=flatten_collate,
    )

    # Create model
    model = ARCTransformer(
        num_tasks=train_dataset.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        max_seq_len=51,
        num_colors=10,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = (
        sum(p.numel() for p in model.task_embedding.parameters())
        + sum(p.numel() for p in model.grid_embedding.parameters())
        + sum(p.numel() for p in model.pos_embedding.parameters())
    )
    non_embedding_params = total_params - embedding_params

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device}")
    print(f"Total model parameters: {total_params:,}")
    print(f"Non-embedding parameters: {non_embedding_params:,}")

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = test_epoch(model, test_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    # Save model with timestamp
    output_dir = Path("output/mini_arc_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"model_{timestamp}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "vocab_size": train_dataset.vocab_size,
            "model_config": {
                "d_model": 128,
                "nhead": 4,
                "num_layers": 3,
                "dim_feedforward": 512,
                "max_seq_len": 51,
                "num_colors": 10,
            },
        },
        model_path,
    )

    print(f"\nModel saved to: {model_path}")

    return model, train_loader, test_loader


if __name__ == "__main__":
    main()
