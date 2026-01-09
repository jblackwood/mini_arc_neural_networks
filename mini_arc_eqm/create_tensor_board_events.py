#!/usr/bin/env python3
"""
Script to recreate TensorBoard events from checkpoint markdown files.

This script parses the .md checkpoint files and writes the metrics to TensorBoard logs,
allowing you to recreate the events when the original tf events have been lost.

Usage:
    Set the md_file and log_dir variables at the top of main() and run:
    python create_tensor_board_events.py

Example:
    md_file = "mini_arc_eqm/results/20260105_010813_epoch_150_checkpoint.md"
    log_dir = "output/mini_arc_eqm5"
"""

import re
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def parse_epoch_line(line: str):
    """Parse an epoch summary line.
    
    Example: "Epoch 1/300 - Train Loss: 0.045345, Test Loss: 0.031181, Time: 154.78s"
    
    Returns:
        dict with keys: epoch, train_loss, test_loss, time (or None if no match)
    """
    pattern = r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Test Loss: ([\d.]+), Time: ([\d.]+)s"
    match = re.match(pattern, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'train_loss': float(match.group(2)),
            'test_loss': float(match.group(3)),
            'time': float(match.group(4))
        }
    return None


def parse_eval_line(line: str):
    """Parse an evaluation line.
    
    Example: "  Train Accuracy: 57.80% (100% acc: 0.8%), Test Accuracy: 59.45% (100% acc: 0.0%)"
    
    Returns:
        dict with keys: train_acc, train_perfect, test_acc, test_perfect (or None if no match)
    """
    pattern = r"\s+Train Accuracy: ([\d.]+)% \(100% acc: ([\d.]+)%\), Test Accuracy: ([\d.]+)% \(100% acc: ([\d.]+)%\)"
    match = re.match(pattern, line)
    if match:
        return {
            'train_acc': float(match.group(1)),
            'train_perfect': float(match.group(2)),
            'test_acc': float(match.group(3)),
            'test_perfect': float(match.group(4))
        }
    return None


def parse_best_iter_line(line: str):
    """Parse a best iteration line.
    
    Example: "  Train Best Iter: 0.5±2.6 (max: 26), Test Best Iter: 0.3±1.0 (max: 5), Time: 24.18s"
    
    Returns:
        dict with keys: train_mean, train_std, test_mean, test_std (or None if no match)
    """
    pattern = r"\s+Train Best Iter: ([\d.]+)±([\d.]+) \(max: \d+\), Test Best Iter: ([\d.]+)±([\d.]+) \(max: \d+\), Time: [\d.]+s"
    match = re.match(pattern, line)
    if match:
        return {
            'train_mean': float(match.group(1)),
            'train_std': float(match.group(2)),
            'test_mean': float(match.group(3)),
            'test_std': float(match.group(4))
        }
    return None


def extract_model_name(md_path: Path):
    """Extract model name from the checkpoint filename.
    
    Example: "20260105_010813_epoch_150_checkpoint.md" -> "20260105_010813_model"
    
    Args:
        md_path: Path to the .md checkpoint file
        
    Returns:
        Model name string
    """
    filename = md_path.stem  # Gets filename without extension
    # Extract timestamp (part before _epoch_)
    match = re.match(r'(\d{8}_\d{6})_epoch_\d+_checkpoint', filename)
    if match:
        return f"{match.group(1)}_model"
    # Fallback: just use the filename stem
    return filename


def parse_md_file(md_path: Path):
    """Parse the markdown file and extract all metrics.
    
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
                # Try to parse evaluation accuracy
                eval_data = parse_eval_line(line)
                if eval_data:
                    current_epoch_data.update(eval_data)
                    continue
                
                # Try to parse best iteration
                best_iter_data = parse_best_iter_line(line)
                if best_iter_data:
                    current_epoch_data.update(best_iter_data)
                    continue
        
        # Don't forget the last epoch
        if current_epoch_data is not None:
            metrics.append(current_epoch_data)
    
    return metrics


def write_to_tensorboard(metrics, log_dir: Path, model_name: str):
    """Write metrics to TensorBoard.
    
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
        
        # Write loss and time metrics (always present)
        writer.add_scalar("Loss/train", epoch_data['train_loss'], epoch)
        writer.add_scalar("Loss/test", epoch_data['test_loss'], epoch)
        writer.add_scalar("Time/epoch", epoch_data['time'], epoch)
        
        # Write denoising metrics if available (convert accuracy percentages to decimals)
        if 'train_acc' in epoch_data:
            writer.add_scalar("DenoiseAccuracy/train", epoch_data['train_acc'] / 100.0, epoch)
            writer.add_scalar("DenoiseAccuracy/test", epoch_data['test_acc'] / 100.0, epoch)
            writer.add_scalar("DenoisePerfect/train", epoch_data['train_perfect'], epoch)
            writer.add_scalar("DenoisePerfect/test", epoch_data['test_perfect'], epoch)
        
        # Write best iteration metrics if available
        if 'train_mean' in epoch_data:
            writer.add_scalar("DenoiseBestIter/train_mean", epoch_data['train_mean'], epoch)
            writer.add_scalar("DenoiseBestIter/train_std", epoch_data['train_std'], epoch)
            writer.add_scalar("DenoiseBestIter/test_mean", epoch_data['test_mean'], epoch)
            writer.add_scalar("DenoiseBestIter/test_std", epoch_data['test_std'], epoch)
    
    writer.close()
    print(f"Successfully wrote {len(metrics)} epochs to TensorBoard logs at {full_log_dir}")


def main():
    # ============ Configure these variables ============
    md_file = "mini_arc_eqm/results/20260109_154034_epoch_600_checkpoint.md"
    log_dir = "output/mini_arc_eqm5/runs"
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
