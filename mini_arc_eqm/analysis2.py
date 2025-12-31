"""Load and analyze ARC model checkpoint."""
import torch
from pathlib import Path
from nb import TransformerModel

# Load the checkpoint
checkpoint_path = Path("output/mini_arc_eqm2/checkpoints/20251231_093942_epoch_20_checkpoint.pt")
print(f"Loading checkpoint from: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Extract model configuration
config = checkpoint['config']
print(f"\nModel configuration:")
print(f"  d_model: {config['d_model']}")
print(f"  nhead: {config['nhead']}")
print(f"  num_layers: {config['num_layers']}")
print(f"  dim_feedforward: {config['dim_feedforward']}")
print(f"  seq_len: {config['seq_len']}")
print(f"  vocab_size: {config['vocab_size']}")
print(f"  dropout: {config['dropout']}")

# Create model and load state dict
model = TransformerModel(
    d_model=config['d_model'],
    nhead=config['nhead'],
    num_layers=config['num_layers'],
    dim_feedforward=config['dim_feedforward'],
    seq_len=config['seq_len'],
    dropout=config['dropout'],
    vocab_size=config['vocab_size'],
)

model.load_state_dict(checkpoint['model_state_dict'])
print(f"\nModel loaded successfully!")

# Get color embedding matrix
color_embeddings = model.color_embedding.weight.detach().cpu()
print(f"\nColor Embedding Matrix (shape: {color_embeddings.shape}):")
print(f"Each row represents the embedding for one color (0-9):\n")

# Print the matrix with color indices
for color_idx in range(color_embeddings.shape[0]):
    embedding = color_embeddings[color_idx]
    print(f"Color {color_idx}: {embedding.numpy()}")
