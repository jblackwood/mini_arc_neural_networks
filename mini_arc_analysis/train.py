"""PyTorch dataset for ARC tasks."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, TypedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from arc_shared import parse_arc_json


@dataclass
class LossComponents:
    """Container for all 9 loss components from 3 masking patterns plus KL loss.

    Attributes:
        output_grid_mask_task_loss: Task loss when output grid is masked
        output_grid_mask_input_loss: Input loss when output grid is masked
        output_grid_mask_output_loss: Output loss when output grid is masked
        input_grid_mask_task_loss: Task loss when input grid is masked
        input_grid_mask_input_loss: Input loss when input grid is masked
        input_grid_mask_output_loss: Output loss when input grid is masked
        task_mask_task_loss: Task loss when task is masked
        task_mask_input_loss: Input loss when task is masked
        task_mask_output_loss: Output loss when task is masked
        kl_loss: KL divergence loss encouraging task embeddings to follow N(0,1)
        total_loss: Sum of all 9 losses plus KL loss
    """

    output_grid_mask_task_loss: float
    output_grid_mask_input_loss: float
    output_grid_mask_output_loss: float
    input_grid_mask_task_loss: float
    input_grid_mask_input_loss: float
    input_grid_mask_output_loss: float
    task_mask_task_loss: float
    task_mask_input_loss: float
    task_mask_output_loss: float
    kl_loss: float
    total_loss: float


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
        task_embedding_num_tokens: Number of tokens to use for task embedding
        d_model: Dimension of embeddings and transformer
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        max_seq_len: Maximum sequence length for positional embeddings
        num_colors: Number of unique colors (0-9, so 10)
    """

    def __init__(
        self,
        num_tasks: int,
        task_embedding_num_tokens: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_seq_len: int,
        num_colors: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_colors = num_colors
        self.task_embedding_num_tokens = task_embedding_num_tokens

        # MASK token is added as an extra embedding (index = num_colors)
        self.mask_token_idx = num_colors

        # Embedding layers
        # Task embedding is larger: num_tokens * d_model
        self.task_embedding = nn.Embedding(
            num_tasks, d_model * task_embedding_num_tokens
        )
        # Initialize task embedding weights to small values near 0
        # Scale down the default initialization to encourage N(0,1) distribution
        with torch.no_grad():
            self.task_embedding.weight.data *= 0.01

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

        # Store num_tasks for weight tying
        self.num_tasks = num_tasks

        # No separate output projection layers - we'll use weight tying with embeddings

    def extract_logits(self, encoded: torch.Tensor, batch_size: int, H: int, W: int):
        """Extract task, input, and output logits from encoded sequence.

        Args:
            encoded: (batch_size, seq_len, d_model) encoded sequence from transformer
            batch_size: Batch size
            H: Height of grids
            W: Width of grids

        Returns:
            Tuple of (task_logits, input_logits, output_logits)
        """
        # Split encoded sequence into task, input, and output portions
        task_encoded = encoded[:, : self.task_embedding_num_tokens, :]
        input_start = self.task_embedding_num_tokens
        input_end = input_start + H * W
        input_encoded = encoded[:, input_start:input_end, :]
        output_encoded = encoded[:, input_end:, :]

        # Task logits
        task_encoded_flat = task_encoded.view(
            batch_size, self.task_embedding_num_tokens * self.d_model
        )
        task_logits = torch.matmul(task_encoded_flat, self.task_embedding.weight.t())

        # Grid logits
        grid_weight = self.grid_embedding.weight[: self.num_colors]
        input_logits = torch.matmul(input_encoded, grid_weight.t())
        output_logits = torch.matmul(output_encoded, grid_weight.t())

        return task_logits, input_logits, output_logits

    def forward(
        self,
        task_idx: torch.Tensor,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor,
    ):
        """Forward pass predicting the entire sequence using weight tying.

        Creates three different masked sequences and returns predictions for each:
        1. output_grid_mask: Mask output grid (current behavior)
        2. input_grid_mask: Mask input grid
        3. task_mask: Mask task embedding

        Args:
            task_idx: (batch_size,) task indices
            input_grid: (batch_size, H, W) input grids
            output_grid: (batch_size, H, W) output grids (not used currently)

        Returns:
            Dictionary containing 9 logits (3 mask types × 3 prediction types):
                - output_grid_mask_task_logits: (batch_size, num_tasks)
                - output_grid_mask_input_logits: (batch_size, H*W, num_colors)
                - output_grid_mask_output_logits: (batch_size, H*W, num_colors)
                - input_grid_mask_task_logits: (batch_size, num_tasks)
                - input_grid_mask_input_logits: (batch_size, H*W, num_colors)
                - input_grid_mask_output_logits: (batch_size, H*W, num_colors)
                - task_mask_task_logits: (batch_size, num_tasks)
                - task_mask_input_logits: (batch_size, H*W, num_colors)
                - task_mask_output_logits: (batch_size, H*W, num_colors)
        """
        batch_size = task_idx.shape[0]
        H, W = input_grid.shape[1], input_grid.shape[2]

        # Flatten grids
        input_flat = input_grid.view(batch_size, -1)  # (batch_size, H*W)
        output_flat = output_grid.view(batch_size, -1)  # (batch_size, H*W)

        # Get task embedding and reshape into multiple tokens
        task_emb_flat = self.task_embedding(
            task_idx
        )  # (batch_size, d_model * num_tokens)
        task_emb = task_emb_flat.view(
            batch_size, self.task_embedding_num_tokens, self.d_model
        )  # (batch_size, num_tokens, d_model)

        # Get grid embeddings
        input_emb = self.grid_embedding(input_flat)  # (batch_size, H*W, d_model)
        output_emb = self.grid_embedding(output_flat)  # (batch_size, H*W, d_model)

        # Create mask token embeddings
        mask_tokens = torch.full(
            (batch_size, H * W),
            self.mask_token_idx,
            dtype=torch.long,
            device=input_grid.device,
        )
        mask_emb = self.grid_embedding(mask_tokens)  # (batch_size, H*W, d_model)

        # Create masked task embedding (all mask tokens)
        task_mask_tokens = torch.full(
            (batch_size, self.task_embedding_num_tokens, self.d_model),
            0.0,  # Use zeros for masked task tokens
            dtype=task_emb.dtype,
            device=task_emb.device,
        )

        # Create three different sequences with different masking patterns
        # 1. Mask output grid (current behavior)
        seq_output_mask = torch.cat(
            [task_emb, input_emb, mask_emb], dim=1
        )  # (batch_size, num_tokens+2*H*W, d_model)

        # 2. Mask input grid
        seq_input_mask = torch.cat(
            [task_emb, mask_emb, output_emb], dim=1
        )  # (batch_size, num_tokens+2*H*W, d_model)

        # 3. Mask task embedding
        seq_task_mask = torch.cat(
            [task_mask_tokens, input_emb, output_emb], dim=1
        )  # (batch_size, num_tokens+2*H*W, d_model)

        seq_len = seq_output_mask.shape[1]

        # Add positional embeddings to all three sequences
        positions = (
            torch.arange(seq_len, device=seq_output_mask.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_emb = self.pos_embedding(positions)

        seq_output_mask = seq_output_mask + pos_emb
        seq_input_mask = seq_input_mask + pos_emb
        seq_task_mask = seq_task_mask + pos_emb

        # Pass all three sequences through transformer
        encoded_output_mask = self.transformer(seq_output_mask)
        encoded_input_mask = self.transformer(seq_input_mask)
        encoded_task_mask = self.transformer(seq_task_mask)

        # Extract logits for all three masked sequences
        (
            output_grid_mask_task_logits,
            output_grid_mask_input_logits,
            output_grid_mask_output_logits,
        ) = self.extract_logits(encoded_output_mask, batch_size, H, W)

        (
            input_grid_mask_task_logits,
            input_grid_mask_input_logits,
            input_grid_mask_output_logits,
        ) = self.extract_logits(encoded_input_mask, batch_size, H, W)

        (
            task_mask_task_logits,
            task_mask_input_logits,
            task_mask_output_logits,
        ) = self.extract_logits(encoded_task_mask, batch_size, H, W)

        return {
            "output_grid_mask_task_logits": output_grid_mask_task_logits,
            "output_grid_mask_input_logits": output_grid_mask_input_logits,
            "output_grid_mask_output_logits": output_grid_mask_output_logits,
            "input_grid_mask_task_logits": input_grid_mask_task_logits,
            "input_grid_mask_input_logits": input_grid_mask_input_logits,
            "input_grid_mask_output_logits": input_grid_mask_output_logits,
            "task_mask_task_logits": task_mask_task_logits,
            "task_mask_input_logits": task_mask_input_logits,
            "task_mask_output_logits": task_mask_output_logits,
        }


def train_epoch(
    model: ARCTransformer,
    dataloader: DataLoader,
    optimizer,
    criterion,
    device,
    kl_weight: float,
) -> LossComponents:
    """Train for one epoch.

    Args:
        model: The ARCTransformer model
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss criterion
        device: Device to train on
        kl_weight: Weight for KL divergence loss encouraging task embeddings to follow N(0,1)

    Returns:
        LossComponents containing average losses for the epoch
    """
    model.train()

    # Initialize accumulators for all 9 losses plus KL loss
    total_output_grid_mask_task_loss = 0
    total_output_grid_mask_input_loss = 0
    total_output_grid_mask_output_loss = 0
    total_input_grid_mask_task_loss = 0
    total_input_grid_mask_input_loss = 0
    total_input_grid_mask_output_loss = 0
    total_task_mask_task_loss = 0
    total_task_mask_input_loss = 0
    total_task_mask_output_loss = 0
    total_kl_loss = 0
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Stack batch items
        task_indices = torch.stack([item["task_idx"] for item in batch]).to(device)
        input_grids = torch.stack([item["input"] for item in batch]).to(device)
        output_grids = torch.stack([item["output"] for item in batch]).to(device)

        # Forward pass
        predictions = model(task_indices, input_grids, output_grids)

        # Prepare targets
        input_targets = input_grids.view(input_grids.shape[0], -1)
        output_targets = output_grids.view(output_grids.shape[0], -1)

        # Compute all 9 losses
        # Output grid mask losses
        output_grid_mask_task_loss = criterion(
            predictions["output_grid_mask_task_logits"], task_indices
        )
        output_grid_mask_input_loss = criterion(
            predictions["output_grid_mask_input_logits"].view(
                -1, predictions["output_grid_mask_input_logits"].shape[-1]
            ),
            input_targets.view(-1),
        )
        output_grid_mask_output_loss = criterion(
            predictions["output_grid_mask_output_logits"].view(
                -1, predictions["output_grid_mask_output_logits"].shape[-1]
            ),
            output_targets.view(-1),
        )

        # Input grid mask losses
        input_grid_mask_task_loss = criterion(
            predictions["input_grid_mask_task_logits"], task_indices
        )
        input_grid_mask_input_loss = criterion(
            predictions["input_grid_mask_input_logits"].view(
                -1, predictions["input_grid_mask_input_logits"].shape[-1]
            ),
            input_targets.view(-1),
        )
        input_grid_mask_output_loss = criterion(
            predictions["input_grid_mask_output_logits"].view(
                -1, predictions["input_grid_mask_output_logits"].shape[-1]
            ),
            output_targets.view(-1),
        )

        # Task mask losses
        task_mask_task_loss = criterion(
            predictions["task_mask_task_logits"], task_indices
        )
        task_mask_input_loss = criterion(
            predictions["task_mask_input_logits"].view(
                -1, predictions["task_mask_input_logits"].shape[-1]
            ),
            input_targets.view(-1),
        )
        task_mask_output_loss = criterion(
            predictions["task_mask_output_logits"].view(
                -1, predictions["task_mask_output_logits"].shape[-1]
            ),
            output_targets.view(-1),
        )

        # Compute KL divergence loss to encourage task embeddings to follow N(0,1)
        # Task embeddings shape: (num_tasks, d_model * task_embedding_num_tokens)
        # Reshape to: (num_tasks, task_embedding_num_tokens, d_model)
        task_emb_reshaped = model.task_embedding.weight.view(
            -1, model.task_embedding_num_tokens, model.d_model
        )
        # Flatten to (num_tasks * task_embedding_num_tokens, d_model)
        # All task tokens share the same latent space
        task_emb_flat = task_emb_reshaped.view(-1, model.d_model)
        # Compute mean and variance across ALL tasks and ALL token positions
        # This ensures all tokens are part of the same N(0,1) latent space
        mean = task_emb_flat.mean(dim=0, keepdim=True)  # (1, d_model)
        var = task_emb_flat.var(dim=0, unbiased=False, keepdim=True)  # (1, d_model)
        # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
        kl_loss = kl_weight * 0.5 * torch.sum(var + mean**2 - 1 - torch.log(var + 1e-8))

        # Total loss is sum of all 9 losses plus KL loss
        loss = (
            output_grid_mask_task_loss
            + output_grid_mask_input_loss
            + output_grid_mask_output_loss
            + input_grid_mask_task_loss
            + input_grid_mask_input_loss
            + input_grid_mask_output_loss
            + task_mask_task_loss
            + task_mask_input_loss
            + task_mask_output_loss
            + kl_loss
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses
        total_output_grid_mask_task_loss += output_grid_mask_task_loss.item()
        total_output_grid_mask_input_loss += output_grid_mask_input_loss.item()
        total_output_grid_mask_output_loss += output_grid_mask_output_loss.item()
        total_input_grid_mask_task_loss += input_grid_mask_task_loss.item()
        total_input_grid_mask_input_loss += input_grid_mask_input_loss.item()
        total_input_grid_mask_output_loss += input_grid_mask_output_loss.item()
        total_task_mask_task_loss += task_mask_task_loss.item()
        total_task_mask_input_loss += task_mask_input_loss.item()
        total_task_mask_output_loss += task_mask_output_loss.item()
        total_kl_loss += kl_loss.item()
        total_loss += loss.item()
        num_batches += 1

        # # Debug: print batch progress
        # print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    return LossComponents(
        output_grid_mask_task_loss=total_output_grid_mask_task_loss / num_batches,
        output_grid_mask_input_loss=total_output_grid_mask_input_loss / num_batches,
        output_grid_mask_output_loss=total_output_grid_mask_output_loss / num_batches,
        input_grid_mask_task_loss=total_input_grid_mask_task_loss / num_batches,
        input_grid_mask_input_loss=total_input_grid_mask_input_loss / num_batches,
        input_grid_mask_output_loss=total_input_grid_mask_output_loss / num_batches,
        task_mask_task_loss=total_task_mask_task_loss / num_batches,
        task_mask_input_loss=total_task_mask_input_loss / num_batches,
        task_mask_output_loss=total_task_mask_output_loss / num_batches,
        kl_loss=total_kl_loss / num_batches,
        total_loss=total_loss / num_batches,
    )


def test_epoch(
    model: ARCTransformer,
    dataloader: DataLoader,
    criterion,
    device,
    kl_weight: float,
) -> LossComponents:
    """Evaluate on test set.

    Args:
        model: The ARCTransformer model
        dataloader: DataLoader for test data
        criterion: Loss criterion
        device: Device to evaluate on
        kl_weight: Weight for KL divergence loss encouraging task embeddings to follow N(0,1)

    Returns:
        LossComponents containing average losses for the epoch
    """
    model.eval()

    # Initialize accumulators for all 9 losses plus KL loss
    total_output_grid_mask_task_loss = 0
    total_output_grid_mask_input_loss = 0
    total_output_grid_mask_output_loss = 0
    total_input_grid_mask_task_loss = 0
    total_input_grid_mask_input_loss = 0
    total_input_grid_mask_output_loss = 0
    total_task_mask_task_loss = 0
    total_task_mask_input_loss = 0
    total_task_mask_output_loss = 0
    total_kl_loss = 0
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Stack batch items
            task_indices = torch.stack([item["task_idx"] for item in batch]).to(device)
            input_grids = torch.stack([item["input"] for item in batch]).to(device)
            output_grids = torch.stack([item["output"] for item in batch]).to(device)

            # Forward pass
            predictions = model(task_indices, input_grids, output_grids)

            # Prepare targets
            input_targets = input_grids.view(input_grids.shape[0], -1)
            output_targets = output_grids.view(output_grids.shape[0], -1)

            # Compute all 9 losses
            # Output grid mask losses
            output_grid_mask_task_loss = criterion(
                predictions["output_grid_mask_task_logits"], task_indices
            )
            output_grid_mask_input_loss = criterion(
                predictions["output_grid_mask_input_logits"].view(
                    -1, predictions["output_grid_mask_input_logits"].shape[-1]
                ),
                input_targets.view(-1),
            )
            output_grid_mask_output_loss = criterion(
                predictions["output_grid_mask_output_logits"].view(
                    -1, predictions["output_grid_mask_output_logits"].shape[-1]
                ),
                output_targets.view(-1),
            )

            # Input grid mask losses
            input_grid_mask_task_loss = criterion(
                predictions["input_grid_mask_task_logits"], task_indices
            )
            input_grid_mask_input_loss = criterion(
                predictions["input_grid_mask_input_logits"].view(
                    -1, predictions["input_grid_mask_input_logits"].shape[-1]
                ),
                input_targets.view(-1),
            )
            input_grid_mask_output_loss = criterion(
                predictions["input_grid_mask_output_logits"].view(
                    -1, predictions["input_grid_mask_output_logits"].shape[-1]
                ),
                output_targets.view(-1),
            )

            # Task mask losses
            task_mask_task_loss = criterion(
                predictions["task_mask_task_logits"], task_indices
            )
            task_mask_input_loss = criterion(
                predictions["task_mask_input_logits"].view(
                    -1, predictions["task_mask_input_logits"].shape[-1]
                ),
                input_targets.view(-1),
            )
            task_mask_output_loss = criterion(
                predictions["task_mask_output_logits"].view(
                    -1, predictions["task_mask_output_logits"].shape[-1]
                ),
                output_targets.view(-1),
            )

            # Compute KL divergence loss to encourage task embeddings to follow N(0,1)
            # Task embeddings shape: (num_tasks, d_model * task_embedding_num_tokens)
            # Reshape to: (num_tasks, task_embedding_num_tokens, d_model)
            task_emb_reshaped = model.task_embedding.weight.view(
                -1, model.task_embedding_num_tokens, model.d_model
            )
            # Flatten to (num_tasks * task_embedding_num_tokens, d_model)
            # All task tokens share the same latent space
            task_emb_flat = task_emb_reshaped.view(-1, model.d_model)
            # Compute mean and variance across ALL tasks and ALL token positions
            # This ensures all tokens are part of the same N(0,1) latent space
            mean = task_emb_flat.mean(dim=0, keepdim=True)  # (1, d_model)
            var = task_emb_flat.var(dim=0, unbiased=False, keepdim=True)  # (1, d_model)
            # KL(N(μ, σ²) || N(0, 1)) = 0.5 * (σ² + μ² - 1 - log(σ²))
            kl_loss = (
                kl_weight * 0.5 * torch.sum(var + mean**2 - 1 - torch.log(var + 1e-8))
            )

            # Total loss is sum of all 9 losses plus KL loss
            loss = (
                output_grid_mask_task_loss
                + output_grid_mask_input_loss
                + output_grid_mask_output_loss
                + input_grid_mask_task_loss
                + input_grid_mask_input_loss
                + input_grid_mask_output_loss
                + task_mask_task_loss
                + task_mask_input_loss
                + task_mask_output_loss
                + kl_loss
            )

            # Accumulate losses
            total_output_grid_mask_task_loss += output_grid_mask_task_loss.item()
            total_output_grid_mask_input_loss += output_grid_mask_input_loss.item()
            total_output_grid_mask_output_loss += output_grid_mask_output_loss.item()
            total_input_grid_mask_task_loss += input_grid_mask_task_loss.item()
            total_input_grid_mask_input_loss += input_grid_mask_input_loss.item()
            total_input_grid_mask_output_loss += input_grid_mask_output_loss.item()
            total_task_mask_task_loss += task_mask_task_loss.item()
            total_task_mask_input_loss += task_mask_input_loss.item()
            total_task_mask_output_loss += task_mask_output_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
            num_batches += 1

    return LossComponents(
        output_grid_mask_task_loss=total_output_grid_mask_task_loss / num_batches,
        output_grid_mask_input_loss=total_output_grid_mask_input_loss / num_batches,
        output_grid_mask_output_loss=total_output_grid_mask_output_loss / num_batches,
        input_grid_mask_task_loss=total_input_grid_mask_task_loss / num_batches,
        input_grid_mask_input_loss=total_input_grid_mask_input_loss / num_batches,
        input_grid_mask_output_loss=total_input_grid_mask_output_loss / num_batches,
        task_mask_task_loss=total_task_mask_task_loss / num_batches,
        task_mask_input_loss=total_task_mask_input_loss / num_batches,
        task_mask_output_loss=total_task_mask_output_loss / num_batches,
        kl_loss=total_kl_loss / num_batches,
        total_loss=total_loss / num_batches,
    )


def main():
    """Train and test the ARC transformer model."""
    # Checkpoint configuration
    # Set to None if not loading from a checkpoint, or provide path to checkpoint file
    LOAD_CHECKPOINT_PATH = None
    assert LOAD_CHECKPOINT_PATH is None or isinstance(
        LOAD_CHECKPOINT_PATH, str
    ), "LOAD_CHECKPOINT_PATH must be explicitly set to None or a string path"

    # Save checkpoint every N epochs (set to 0 to disable checkpoint saving)
    CHECKPOINT_SAVE_INTERVAL = 10

    # Hyperparameters
    folder_path = "output/mini_arc_analysis/train"
    batch_size = 256
    num_epochs = 100
    learning_rate = 1e-4
    task_embedding_learning_rate = 1e-2
    kl_weight = 1

    # Model architecture hyperparameters
    TASK_EMBEDDING_NUM_TOKENS = 5
    D_MODEL = 128
    NHEAD = 4
    NUM_LAYERS = 3
    DIM_FEEDFORWARD = 512
    MAX_SEQ_LEN = 55
    NUM_COLORS = 10

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
        task_embedding_num_tokens=TASK_EMBEDDING_NUM_TOKENS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        max_seq_len=MAX_SEQ_LEN,
        num_colors=NUM_COLORS,
    ).to(device)

    # Optimizer and loss
    # Separate task embedding parameters from other parameters
    task_embedding_params = list(model.task_embedding.parameters())
    task_embedding_param_ids = {id(p) for p in task_embedding_params}
    other_params = [
        p for p in model.parameters() if id(p) not in task_embedding_param_ids
    ]

    # Create optimizer with different learning rates for different parameter groups
    optimizer = torch.optim.Adam(
        [
            {"params": task_embedding_params, "lr": task_embedding_learning_rate},
            {"params": other_params, "lr": learning_rate},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    # Load from checkpoint if specified
    start_epoch = 0
    if LOAD_CHECKPOINT_PATH is not None:
        print(f"Loading checkpoint from: {LOAD_CHECKPOINT_PATH}")
        checkpoint = torch.load(LOAD_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = (
        sum(p.numel() for p in model.task_embedding.parameters())
        + sum(p.numel() for p in model.grid_embedding.parameters())
        + sum(p.numel() for p in model.pos_embedding.parameters())
    )
    non_embedding_params = total_params - embedding_params

    # Create timestamp for this training run (used for both logs and model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # TensorBoard writer
    log_dir = Path(f"output/mini_arc_analysis/runs/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"Training on {device}")
    print(f"Total model parameters: {total_params:,}")
    print(f"Non-embedding parameters: {non_embedding_params:,}")
    print(f"Learning rate (default): {learning_rate}")
    print(f"Learning rate (task embedding): {task_embedding_learning_rate}")
    print(f"KL weight: {kl_weight}")
    print(f"TensorBoard logs: {log_dir}")

    # Create checkpoint directory
    checkpoint_dir = Path("output/mini_arc_analysis/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"model_{timestamp}_checkpoint.pt"

    if CHECKPOINT_SAVE_INTERVAL > 0:
        print(
            f"Checkpoints will be saved every {CHECKPOINT_SAVE_INTERVAL} epochs to: {checkpoint_path}"
        )

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        train_losses = train_epoch(
            model, train_loader, optimizer, criterion, device, kl_weight
        )
        test_losses = test_epoch(model, test_loader, criterion, device, kl_weight)

        epoch_time = time.time() - epoch_start_time

        print(f"\nEpoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(
            f"Total Loss - Train: {train_losses.total_loss:.4f} | Test: {test_losses.total_loss:.4f}"
        )
        print(
            f"KL Loss - Train: {train_losses.kl_loss:.4f} | Test: {test_losses.kl_loss:.4f}"
        )
        print("Output Grid Mask:")
        print(
            f"  Task:   Train: {train_losses.output_grid_mask_task_loss:.4f} | Test: {test_losses.output_grid_mask_task_loss:.4f}"
        )
        print(
            f"  Input:  Train: {train_losses.output_grid_mask_input_loss:.4f} | Test: {test_losses.output_grid_mask_input_loss:.4f}"
        )
        print(
            f"  Output: Train: {train_losses.output_grid_mask_output_loss:.4f} | Test: {test_losses.output_grid_mask_output_loss:.4f}"
        )
        print("Input Grid Mask:")
        print(
            f"  Task:   Train: {train_losses.input_grid_mask_task_loss:.4f} | Test: {test_losses.input_grid_mask_task_loss:.4f}"
        )
        print(
            f"  Input:  Train: {train_losses.input_grid_mask_input_loss:.4f} | Test: {test_losses.input_grid_mask_input_loss:.4f}"
        )
        print(
            f"  Output: Train: {train_losses.input_grid_mask_output_loss:.4f} | Test: {test_losses.input_grid_mask_output_loss:.4f}"
        )
        print("Task Mask:")
        print(
            f"  Task:   Train: {train_losses.task_mask_task_loss:.4f} | Test: {test_losses.task_mask_task_loss:.4f}"
        )
        print(
            f"  Input:  Train: {train_losses.task_mask_input_loss:.4f} | Test: {test_losses.task_mask_input_loss:.4f}"
        )
        print(
            f"  Output: Train: {train_losses.task_mask_output_loss:.4f} | Test: {test_losses.task_mask_output_loss:.4f}"
        )

        # Log to TensorBoard
        # Total loss
        writer.add_scalars(
            "Loss/Total",
            {"Train": train_losses.total_loss, "Test": test_losses.total_loss},
            epoch,
        )

        # KL divergence loss
        writer.add_scalars(
            "Loss/KL",
            {"Train": train_losses.kl_loss, "Test": test_losses.kl_loss},
            epoch,
        )

        # Output grid mask losses
        writer.add_scalars(
            "Loss/OutputGridMask/Task",
            {
                "Train": train_losses.output_grid_mask_task_loss,
                "Test": test_losses.output_grid_mask_task_loss,
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/OutputGridMask/Input",
            {
                "Train": train_losses.output_grid_mask_input_loss,
                "Test": test_losses.output_grid_mask_input_loss,
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/OutputGridMask/Output",
            {
                "Train": train_losses.output_grid_mask_output_loss,
                "Test": test_losses.output_grid_mask_output_loss,
            },
            epoch,
        )

        # Input grid mask losses
        writer.add_scalars(
            "Loss/InputGridMask/Task",
            {
                "Train": train_losses.input_grid_mask_task_loss,
                "Test": test_losses.input_grid_mask_task_loss,
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/InputGridMask/Input",
            {
                "Train": train_losses.input_grid_mask_input_loss,
                "Test": test_losses.input_grid_mask_input_loss,
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/InputGridMask/Output",
            {
                "Train": train_losses.input_grid_mask_output_loss,
                "Test": test_losses.input_grid_mask_output_loss,
            },
            epoch,
        )

        # Task mask losses
        writer.add_scalars(
            "Loss/TaskMask/Task",
            {
                "Train": train_losses.task_mask_task_loss,
                "Test": test_losses.task_mask_task_loss,
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/TaskMask/Input",
            {
                "Train": train_losses.task_mask_input_loss,
                "Test": test_losses.task_mask_input_loss,
            },
            epoch,
        )
        writer.add_scalars(
            "Loss/TaskMask/Output",
            {
                "Train": train_losses.task_mask_output_loss,
                "Test": test_losses.task_mask_output_loss,
            },
            epoch,
        )

        # Log epoch time
        writer.add_scalar("Time/EpochTime", epoch_time, epoch)

        # Save checkpoint at specified intervals
        if CHECKPOINT_SAVE_INTERVAL > 0 and (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "train_loss": train_losses.total_loss,
                    "test_loss": test_losses.total_loss,
                    "vocab_size": train_dataset.vocab_size,
                    "model_config": {
                        "task_embedding_num_tokens": TASK_EMBEDDING_NUM_TOKENS,
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "num_layers": NUM_LAYERS,
                        "dim_feedforward": DIM_FEEDFORWARD,
                        "max_seq_len": MAX_SEQ_LEN,
                        "num_colors": NUM_COLORS,
                    },
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to: {checkpoint_path}")

    # Close TensorBoard writer
    writer.close()

    # Save model with same timestamp as logs
    output_dir = Path("output/mini_arc_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"model_{timestamp}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": num_epochs,
            "train_loss": train_losses.total_loss,
            "test_loss": test_losses.total_loss,
            "vocab_size": train_dataset.vocab_size,
            "model_config": {
                "task_embedding_num_tokens": TASK_EMBEDDING_NUM_TOKENS,
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "num_layers": NUM_LAYERS,
                "dim_feedforward": DIM_FEEDFORWARD,
                "max_seq_len": MAX_SEQ_LEN,
                "num_colors": NUM_COLORS,
            },
        },
        model_path,
    )

    print(f"\nModel saved to: {model_path}")

    return model, train_loader, test_loader


if __name__ == "__main__":
    main()
