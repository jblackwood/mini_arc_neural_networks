#!/usr/bin/env python3
"""
Script to recreate TensorBoard events from JEPA checkpoint markdown files.

This script parses the result.md file from mini_arc_jepa and writes the metrics to TensorBoard logs,
allowing you to recreate the events when the original tf events have been lost.

Usage:
    Set the md_file and log_dir variables at the top of main() and run:
    python create_tensor_board_events.py

Example:
    md_file = "mini_arc_jepa/results/result.md"
    log_dir = "output/mini_arc_jepa/runs"
"""

import re
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_epoch_line(line: str):
    """Parse an epoch summary line for JEPA model.
    
    Example format:
    "Epoch 1/150 - Train JEPA Loss: 0.515991, Train Pred Loss: 1.175712, Test Pred Loss: 1.174907, Time: 306.76s"
    
    Returns:
        dict with keys: epoch, train_jepa_loss, train_pred_loss, test_pred_loss, time
    """
    pattern = r"Epoch (\d+)/\d+ - Train JEPA Loss: ([\d.]+), Train Pred Loss: ([\d.]+), Test Pred Loss: ([\d.]+), Time: ([\d.]+)s"
    match = re.match(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'train_jepa_loss': float(match.group(2)),
            'train_pred_loss': float(match.group(3)),
            'test_pred_loss': float(match.group(4)),
            'time': float(match.group(5))
        }
    return None


def parse_jepa_components_line(line: str):
    """Parse JEPA loss components line.
    
    Example: "  JEPA Loss Components - Train Sim: 0.065497, Train SigReg: 9.075383"
    
    Returns:
        dict with keys: train_sim, train_sigreg (or None if no match)
    """
    pattern = r"\s+JEPA Loss Components - Train Sim: ([\d.]+), Train SigReg: ([\d.]+)"
    match = re.match(pattern, line)
    if match:
        return {
            'train_sim': float(match.group(1)),
            'train_sigreg': float(match.group(2))
        }
    return None


def parse_accuracy_line(line: str):
    """Parse accuracy line.
    
    Example: "  Accuracy - Train: 62.27%, Train Perfect: 0.87%, Test: 62.88%, Test Perfect: 0.50%"
    
    Returns:
        dict with keys: train_acc, train_perfect, test_acc, test_perfect (or None if no match)
    """
    pattern = r"\s+Accuracy - Train: ([\d.]+)%, Train Perfect: ([\d.]+)%, Test: ([\d.]+)%, Test Perfect: ([\d.]+)%"
    match = re.match(pattern, line)
    if match:
        return {
            'train_acc': float(match.group(1)),
            'train_perfect': float(match.group(2)),
            'test_acc': float(match.group(3)),
            'test_perfect': float(match.group(4))
        }
    return None


def parse_model_norms_line(line: str):
    """Parse model norms line.
    
    Example: "  Model Norms - JEPA: 88.1220, Pred: 84.2358, JEPA Out Scale: 13.6842, Pred Out Scale: 1.9275"
    
    Returns:
        dict with keys: jepa_norm, pred_norm, jepa_out_scale, pred_out_scale (or None if no match)
    """
    pattern = r"\s+Model Norms - JEPA: ([\d.]+), Pred: ([\d.]+), JEPA Out Scale: ([\d.]+), Pred Out Scale: ([\d.]+)"
    match = re.match(pattern, line)
    if match:
        return {
            'jepa_norm': float(match.group(1)),
            'pred_norm': float(match.group(2)),
            'jepa_out_scale': float(match.group(3)),
            'pred_out_scale': float(match.group(4))
        }
    return None


def extract_model_name(md_path: Path):
    """Extract model name from the result file.
    
    Example: "result.md" -> "result_model"
    
    Args:
        md_path: Path to the .md file
        
    Returns:
        Model name string
    """
    filename = md_path.stem  # Gets filename without extension
    return f"{filename}_model"


def parse_md_file(md_path: Path):
    """Parse the JEPA markdown file and extract all metrics.
    
    Args:
        md_path: Path to the .md checkpoint file
        
    Returns:
        List of dicts containing metrics for each epoch
    """
    if not md_path.exists():
        raise FileNotFoundError(f"File not found: {md_path}")
    
    metrics = []
    current_epoch_data = None
    
    with open(md_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            
            # Try to parse epoch line
            epoch_data = parse_epoch_line(line)
            if epoch_data:
                # Save previous epoch if exists
                if current_epoch_data is not None:
                    metrics.append(current_epoch_data)
                # Start new epoch
                current_epoch_data = epoch_data
                continue
            
            # If we have a current epoch, try to parse additional metrics
            if current_epoch_data is not None:
                # Try to parse JEPA loss components
                jepa_components_data = parse_jepa_components_line(line)
                if jepa_components_data:
                    current_epoch_data.update(jepa_components_data)
                    continue
                
                # Try to parse accuracy
                accuracy_data = parse_accuracy_line(line)
                if accuracy_data:
                    current_epoch_data.update(accuracy_data)
                    continue
                
                # Try to parse model norms
                model_norms_data = parse_model_norms_line(line)
                if model_norms_data:
                    current_epoch_data.update(model_norms_data)
                    continue
        
        # Don't forget the last epoch
        if current_epoch_data is not None:
            metrics.append(current_epoch_data)
    
    return metrics


def write_to_tensorboard(metrics, log_dir: Path, model_name: str):
    """Write JEPA metrics to TensorBoard.
    
    Args:
        metrics: List of dicts containing metrics for each epoch
        log_dir: Base directory where TensorBoard logs will be written
        model_name: Name of the model (used as subdirectory)
    """
    full_log_dir = Path(log_dir) / model_name
    full_log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(full_log_dir))
    
    for epoch_data in metrics:
        epoch = epoch_data['epoch']
        
        # Write loss metrics (always present)
        writer.add_scalar("Loss/train_jepa", epoch_data['train_jepa_loss'], epoch)
        writer.add_scalar("Loss/train_pred", epoch_data['train_pred_loss'], epoch)
        writer.add_scalar("Loss/test_pred", epoch_data['test_pred_loss'], epoch)
        writer.add_scalar("Time/epoch", epoch_data['time'], epoch)
        
        # Write JEPA loss components if available
        if 'train_sim' in epoch_data:
            writer.add_scalar("JEPAComponents/train_sim", epoch_data['train_sim'], epoch)
            writer.add_scalar("JEPAComponents/train_sigreg", epoch_data['train_sigreg'], epoch)
        
        # Write accuracy metrics if available (convert percentages to decimals)
        if 'train_acc' in epoch_data:
            writer.add_scalar("Accuracy/train", epoch_data['train_acc'] / 100.0, epoch)
            writer.add_scalar("Accuracy/test", epoch_data['test_acc'] / 100.0, epoch)
            writer.add_scalar("Perfect/train", epoch_data['train_perfect'] / 100.0, epoch)
            writer.add_scalar("Perfect/test", epoch_data['test_perfect'] / 100.0, epoch)
        
        # Write model norms if available
        if 'jepa_norm' in epoch_data:
            writer.add_scalar("ModelNorms/jepa", epoch_data['jepa_norm'], epoch)
            writer.add_scalar("ModelNorms/pred", epoch_data['pred_norm'], epoch)
            writer.add_scalar("ModelNorms/jepa_out_scale", epoch_data['jepa_out_scale'], epoch)
            writer.add_scalar("ModelNorms/pred_out_scale", epoch_data['pred_out_scale'], epoch)
    
    writer.close()
    print(f"Successfully wrote {len(metrics)} epochs to TensorBoard logs at {full_log_dir}")


def main():
    # ============ Configure these variables ============
    md_file = "mini_arc_jepa/results/Saved checkpoint to output/mini_arc_jepa/checkpoints/20260209_055632_epoch_75_checkpoint.pt.md"
    log_dir = "output/mini_arc_jepa/runs"
    # ===================================================
    
    md_path = Path(md_file)
    log_dir = Path(log_dir)
    
    model_name = extract_model_name(md_path)
    print(f"Model name: {model_name}")
    
    print(f"Parsing {md_path}...")
    metrics = parse_md_file(md_path)
    print(f"Found {len(metrics)} epochs")
    
    print(f"Writing to TensorBoard at {log_dir / model_name}...")
    write_to_tensorboard(metrics, log_dir, model_name)
    
    print("\nDone! You can now view the logs with:")
    print(f"  tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
