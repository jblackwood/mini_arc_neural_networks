"""PyTorch dataset for ARC tasks."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from arc_shared import parse_arc_json


class ARCTaskDataset(Dataset):
    """PyTorch dataset that loads ARC tasks and returns concatenated cell values.

    Each task returns a 200-length tensor of cell values:
    - 4 examples (train + test combined)
    - Each example has input and output grids (2 grids per example)
    - Each grid is 5x5 = 25 cells
    - Total: 4 examples * 2 grids * 25 cells = 200 cells
    - Each cell contains a value from 0-9
    """

    def __init__(self, folder_path: str):
        """Initialize the dataset.

        Args:
            folder_path: Path to folder containing task JSON files
        """
        self.folder_path = Path(folder_path)
        self.task_files = sorted(self.folder_path.glob("*.json"))

    def __len__(self) -> int:
        return len(self.task_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a task as a 200-length tensor of cell values.

        Args:
            idx: Index of the task

        Returns:
            Tensor of shape (200,) containing cell values (0-9)
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

        # Convert to tensor
        # all_cells should have 200 elements (4 examples * 2 grids * 25 cells)
        cells_tensor = torch.tensor(all_cells, dtype=torch.long)

        return cells_tensor


class TransformerModel(nn.Module):
    """Non-causal transformer encoder for ARC tasks."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        seq_len: int,
        dropout: float = 0.1,
    ):
        """Initialize the transformer model.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            seq_len: Sequence length (200 positions)
            dropout: Dropout rate
        """
        super().__init__()

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

        # Output projection to d_model (will be converted back to embeddings)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) - already embedded

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Add positional embedding
        x = x + self.pos_embedding.unsqueeze(0)  # (batch_size, seq_len, d_model)

        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Project output
        x = self.output_proj(x)  # (batch_size, seq_len, d_model)

        return x


def compute_loss_for_batch(
    model: nn.Module,
    embedding: nn.Embedding,
    batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute loss for a single batch.

    Args:
        model: The transformer model
        embedding: Embedding layer for cell values
        batch: Batch of cell values, shape (batch_size, 200)
        device: Device to compute on

    Returns:
        Loss tensor (scalar)
    """
    cell_values = batch.to(device)  # (batch_size, 200)

    # Convert to embeddings
    x = embedding(cell_values)  # (batch_size, 200, d_model)

    # Create noisy input - only corrupt the last 25 tokens (output grid)
    xg = x.clone()

    # Create random gaussian noise for the last 25 tokens
    eps = torch.randn_like(xg[:, -25:, :])

    # Sample random gamma uniformly between 0 and 1
    # Shape: (batch_size, 1, 1) - broadcasts across the 25 positions and d_model
    gamma = torch.rand(x.size(0), 1, 1, device=device)

    # Create noisy input for the last 25 tokens
    xg[:, -25:, :] = (1 - gamma) * eps + gamma * x[:, -25:, :]

    # Create target with conditional scaling for the last 25 tokens
    # c(gamma) = 1 if gamma < 0.8, else (1-gamma)/(1-0.8)
    c_gamma = torch.where(gamma < 0.8, torch.ones_like(gamma), (1 - gamma) / 0.2)
    target_last_25 = (eps - x[:, -25:, :]) * c_gamma

    # Forward pass
    output = model(xg)

    # Compute loss only on the last 25 tokens
    loss = ((output[:, -25:, :] - target_last_25) ** 2).mean()

    return loss


def train_epoch(
    model: nn.Module,
    embedding: nn.Embedding,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The transformer model
        embedding: Embedding layer for cell values
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    embedding.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        # Compute loss
        loss = compute_loss_for_batch(model, embedding, batch, device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def test_epoch(
    model: nn.Module,
    embedding: nn.Embedding,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate on test set.

    Args:
        model: The transformer model
        embedding: Embedding layer for cell values
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Average test loss
    """
    model.eval()
    embedding.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            # Compute loss
            loss = compute_loss_for_batch(model, embedding, batch, device)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    """Train transformer model on ARC tasks."""
    # ========== Configuration ==========
    # Model parameters
    d_model = 128  # Model dimension
    nhead = 4  # Number of attention heads
    num_layers = 3  # Number of transformer layers
    dim_feedforward = 512  # Feedforward dimension
    dropout = 0.1

    # Training parameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-3

    # Data parameters
    seq_len = 200  # Sequence length
    num_cell_values = 10  # Cell values 0-9

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log_dir = f"output/mini_arc_eqm/runs/{timestamp}_model"
    model_save_dir = "output/mini_arc_eqm/models"
    model_save_path = f"{model_save_dir}/{timestamp}_model.pt"
    # ===================================

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = ARCTaskDataset("output/mini_arc_eqm/train")
    test_dataset = ARCTaskDataset("output/mini_arc_eqm/test")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Create embedding layer
    embedding = nn.Embedding(num_cell_values, d_model).to(device)

    # Create model
    model = TransformerModel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        seq_len=seq_len,
        dropout=dropout,
    ).to(device)

    # Count parameters
    embedding_params = sum(p.numel() for p in embedding.parameters() if p.requires_grad)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = embedding_params + model_params
    print(f"Embedding has {embedding_params:,} trainable parameters")
    print(f"Model has {model_params:,} trainable parameters")
    print(f"Total: {total_params:,} trainable parameters")

    # Create optimizer (include both embedding and model parameters)
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + list(model.parameters()), lr=learning_rate
    )

    # Create tensorboard writer
    Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, embedding, train_loader, optimizer, device)

        # Test
        test_loss = test_epoch(model, embedding, test_loader, device)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Log to console
        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )

        # Log to tensorboard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Time/epoch", epoch_time, epoch)

    writer.close()
    print("\nTraining complete!")

    # Save model
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_state_dict": embedding.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "seq_len": seq_len,
                "num_cell_values": num_cell_values,
                "dropout": dropout,
            },
        },
        model_save_path,
    )
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
