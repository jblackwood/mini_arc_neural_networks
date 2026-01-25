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
    
    Example formats:
    - "Epoch 1/300 - Train Loss: 0.045345, Test Loss: 0.031181, Time: 154.78s, Weight Norm: 378.6285, Logit Scale: 8.1446"
    - "Epoch 1/150 - Train Loss: 635.124390 (CE: 1.578391, KL: 633.546021), Test Loss: 470.223816 (CE: 1.351296, KL: 468.872528), Time: 127.30s"
    
    Returns:
        dict with keys: epoch, train_loss, test_loss, time, weight_norm (optional), logit_scale (optional), 
                        train_ce (optional), train_kl (optional), test_ce (optional), test_kl (optional)
    """
    # Try the format with CE/KL breakdown
    pattern_ce_kl = r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+) \(CE: ([\d.]+), KL: ([-\d.]+)\), Test Loss: ([\d.]+) \(CE: ([\d.]+), KL: ([-\d.]+)\), Time: ([\d.]+)s"
    match = re.match(pattern_ce_kl, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'train_loss': float(match.group(2)),
            'train_ce': float(match.group(3)),
            'train_kl': float(match.group(4)),
            'test_loss': float(match.group(5)),
            'test_ce': float(match.group(6)),
            'test_kl': float(match.group(7)),
            'time': float(match.group(8))
        }
    
    # Try the format with Weight Norm and Logit Scale
    pattern_new = r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Test Loss: ([\d.]+), Time: ([\d.]+)s, Weight Norm: ([\d.]+), Logit Scale: ([\d.]+)"
    match = re.match(pattern_new, line)
    if match:
        return {
            'epoch': int(match.group(1)),
            'train_loss': float(match.group(2)),
            'test_loss': float(match.group(3)),
            'time': float(match.group(4)),
            'weight_norm': float(match.group(5)),
            'logit_scale': float(match.group(6))
        }
    
    # Try the old format without Weight Norm and Logit Scale
    pattern_old = r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+), Test Loss: ([\d.]+), Time: ([\d.]+)s"
    match = re.match(pattern_old, line)
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


def parse_layer_mean_squares_line(line: str):
    """Parse a layer mean squares line.
    
    Example: "  Layer Mean Squares - Task Emb: 0.003909, Token Emb: 0.001299, Out Proj 1: 0.001515, Out Proj 2: 0.006631, Transformer: 0.001155"
    
    Returns:
        dict with keys: task_emb, token_emb, out_proj_1, out_proj_2, transformer (or None if no match)
    """
    pattern = r"\s+Layer Mean Squares - Task Emb: ([\d.]+), Token Emb: ([\d.]+), Out Proj 1: ([\d.]+), Out Proj 2: ([\d.]+), Transformer: ([\d.]+)"
    match = re.match(pattern, line)
    if match:
        return {
            'task_emb': float(match.group(1)),
            'token_emb': float(match.group(2)),
            'out_proj_1': float(match.group(3)),
            'out_proj_2': float(match.group(4)),
            'transformer': float(match.group(5))
        }
    return None


def parse_sparsity_line(line: str):
    """Parse a sparsity line.
    
    Example: "  Task Embedding Sparsity: 0.38%"
    
    Returns:
        dict with key: task_emb_sparsity (or None if no match)
    """
    pattern = r"\s+Task Embedding Sparsity: ([\d.]+)%"
    match = re.match(pattern, line)
    if match:
        return {
            'task_emb_sparsity': float(match.group(1))
        }
    return None


def parse_loss_components_line(line: str):
    """Parse a loss components line.
    
    Example: "  Loss Components - Train CE: 1.535273, Train L1: 0.223067, Test CE: 1.325884, Test L1: 0.222456"
    
    Returns:
        dict with keys: train_ce, train_l1, test_ce, test_l1 (or None if no match)
    """
    pattern = r"\s+Loss Components - Train CE: ([\d.]+), Train L1: ([\d.]+), Test CE: ([\d.]+), Test L1: ([\d.]+)"
    match = re.match(pattern, line)
    if match:
        return {
            'train_ce': float(match.group(1)),
            'train_l1': float(match.group(2)),
            'test_ce': float(match.group(3)),
            'test_l1': float(match.group(4))
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
                
                # Try to parse layer mean squares
                layer_mean_squares_data = parse_layer_mean_squares_line(line)
                if layer_mean_squares_data:
                    current_epoch_data.update(layer_mean_squares_data)
                    continue
                
                # Try to parse sparsity
                sparsity_data = parse_sparsity_line(line)
                if sparsity_data:
                    current_epoch_data.update(sparsity_data)
                    continue
                
                # Try to parse loss components
                loss_components_data = parse_loss_components_line(line)
                if loss_components_data:
                    current_epoch_data.update(loss_components_data)
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
        
        # Write CE/KL components if available
        if 'train_ce' in epoch_data:
            writer.add_scalar("LossComponents/train_ce", epoch_data['train_ce'], epoch)
            writer.add_scalar("LossComponents/train_kl", epoch_data['train_kl'], epoch)
            writer.add_scalar("LossComponents/test_ce", epoch_data['test_ce'], epoch)
            writer.add_scalar("LossComponents/test_kl", epoch_data['test_kl'], epoch)
        
        # Write model metrics if available
        if 'weight_norm' in epoch_data:
            writer.add_scalar("Model/weight_norm", epoch_data['weight_norm'], epoch)
        if 'logit_scale' in epoch_data:
            writer.add_scalar("Model/logit_scale", epoch_data['logit_scale'], epoch)
        
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
        
        # Write layer mean squares if available
        if 'task_emb' in epoch_data:
            writer.add_scalar("LayerMeanSquare/task_embedding", epoch_data['task_emb'], epoch)
            writer.add_scalar("LayerMeanSquare/token_embedding", epoch_data['token_emb'], epoch)
            writer.add_scalar("LayerMeanSquare/output_proj_1", epoch_data['out_proj_1'], epoch)
            writer.add_scalar("LayerMeanSquare/output_proj_2", epoch_data['out_proj_2'], epoch)
            writer.add_scalar("LayerMeanSquare/transformer_avg", epoch_data['transformer'], epoch)
        
        # Write sparsity if available
        if 'task_emb_sparsity' in epoch_data:
            writer.add_scalar("Sparsity/task_embedding", epoch_data['task_emb_sparsity'], epoch)
        
        # Write L1 loss components if available
        if 'train_l1' in epoch_data:
            writer.add_scalar("LossComponents/train_l1", epoch_data['train_l1'], epoch)
            writer.add_scalar("LossComponents/test_l1", epoch_data['test_l1'], epoch)
    
    writer.close()
    print(f"Successfully wrote {len(metrics)} epochs to TensorBoard logs at {full_log_dir}")


def main():
    # ============ Configure these variables ============
    md_file = "mini_arc_eqm/results/20260125_191626_epoch_60_checkpoint.md"
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
