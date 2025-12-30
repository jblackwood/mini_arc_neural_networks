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
    """PyTorch dataset that loads ARC tasks and returns centered one-hot encoded cell values.

    Each task returns a (200, d_model) tensor:
    - 4 examples (train + test combined)
    - Each example has input and output grids (2 grids per example)
    - Each grid is 5x5 = 25 cells
    - Total: 4 examples * 2 grids * 25 cells = 200 cells
    - Each cell is one-hot encoded with 10 values (0-9) padded to d_model
    - One-hot vectors are centered to have mean 0 (subtract 1/d_model from all dims)
    """

    def __init__(self, folder_path: str, d_model: int):
        """Initialize the dataset.

        Args:
            folder_path: Path to folder containing task JSON files
            d_model: Model dimension for one-hot encoding (must be >= 10)
        """
        if d_model < 10:
            raise ValueError(f"d_model must be >= 10, got {d_model}")
        self.folder_path = Path(folder_path)
        self.task_files = sorted(self.folder_path.glob("*.json"))
        self.d_model = d_model

    def __len__(self) -> int:
        return len(self.task_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a task as a (200, d_model) tensor of centered one-hot encoded cell values.

        Args:
            idx: Index of the task

        Returns:
            Tensor of shape (200, d_model) containing centered one-hot encoded cell values.
            Each vector has mean 0 across all d_model dimensions.
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

        # Convert to one-hot encoding
        # all_cells should have 200 elements (4 examples * 2 grids * 25 cells)
        one_hot = torch.zeros(200, self.d_model, dtype=torch.float32)
        for i, cell_value in enumerate(all_cells):
            one_hot[i, cell_value] = 1.0

        # Center the one-hot vectors to have mean 0
        # Each vector has one 1.0 and (d_model-1) 0.0s, so mean is 1/d_model
        # We subtract 1/d_model from all dimensions to center around 0
        one_hot -= 1.0 / self.d_model

        return one_hot


class TransformerModel(nn.Module):
    """Non-causal transformer encoder for ARC tasks."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        seq_len: int,
        dropout: float,
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

        # Linear output projection (d_model -> d_model)
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

        # Apply output projection
        x = self.output_proj(x)  # (batch_size, seq_len, d_model)

        return x


def compute_loss_for_batch(
    model: nn.Module,
    batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute loss for a single batch.

    Args:
        model: The transformer model
        batch: Batch of one-hot encoded cell values, shape (batch_size, 200, d_model)
        device: Device to compute on

    Returns:
        Loss tensor (scalar)
    """
    x = batch.to(device)  # (batch_size, 200, d_model)

    # Create noisy input - corrupt all 200 tokens
    xg = x.clone()

    # Create random gaussian noise for all tokens
    eps = torch.randn_like(xg)

    # Sample random gamma uniformly between 0 and 1
    # Shape: (batch_size, 1, 1) - broadcasts across all 200 positions and d_model
    gamma = torch.rand(x.size(0), 1, 1, device=device)

    # Create noisy input for all tokens
    xg = (1 - gamma) * eps + gamma * x

    # Create target with conditional scaling for all tokens
    # c(gamma) = 1 if gamma < 0.8, else (1-gamma)/(1-0.8)
    c_gamma = torch.where(gamma < 0.8, torch.ones_like(gamma), (1 - gamma) / 0.2)
    target = (eps - x) * c_gamma

    # Forward pass
    output = model(xg)

    # Compute loss on all tokens
    loss = ((output - target) ** 2).mean()

    return loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Args:
        model: The transformer model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        # Compute loss
        loss = compute_loss_for_batch(model, batch, device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def test_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate on test set.

    Args:
        model: The transformer model
        test_loader: Test data loader
        device: Device to evaluate on

    Returns:
        Average test loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            # Compute loss
            loss = compute_loss_for_batch(model, batch, device)

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
    num_epochs = 40
    learning_rate = 1e-3

    # Data parameters
    seq_len = 200  # Sequence length
    num_cell_values = 10  # Cell values 0-9

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_log_dir = f"output/mini_arc_eqm/runs/{timestamp}_model"
    model_save_dir = "output/mini_arc_eqm/models"
    model_save_path = f"{model_save_dir}/{timestamp}_model.pt"
    checkpoint_dir = "output/mini_arc_eqm/checkpoints"
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
    train_dataset = ARCTaskDataset("output/mini_arc_eqm/train", d_model=d_model)
    test_dataset = ARCTaskDataset("output/mini_arc_eqm/test", d_model=d_model)

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
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {model_params:,} trainable parameters")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create tensorboard writer
    Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Test
        test_loss = test_epoch(model, test_loader, device)

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

        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint_path = (
                f"{checkpoint_dir}/{timestamp}_epoch_{epoch + 1}_checkpoint.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
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
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

    writer.close()
    print("\nTraining complete!")

    # Save model
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
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
