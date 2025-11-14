"""
src/visualize_training.py
Standalone script to visualize training history from JSON files
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_history(json_path: Path) -> dict:
    """Load training history from JSON file"""
    with open(json_path, 'r') as f:
        history = json.load(f)
    return history


def plot_training_curves(history: dict, output_path: Optional[Path] = None, title: str = "Training Curves"):
    """
    Plot training curves from history dictionary
    
    Args:
        history: Dictionary with keys like 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_f1'
        output_path: Path to save the plot (if None, displays)
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2, marker='o', markersize=3)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2, marker='s', markersize=3)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # F1 Score (if available)
    if 'val_f1' in history and history['val_f1']:
        axes[2].plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2, marker='^', markersize=3)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('F1 Score', fontsize=12)
        axes[2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
    else:
        # If no F1, show learning rate or other metric
        axes[2].text(0.5, 0.5, 'F1 Score not available', 
                    ha='center', va='center', fontsize=14, transform=axes[2].transAxes)
        axes[2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training curves to {output_path}")
    else:
        plt.show()
    
    plt.close()


def find_history_files(search_dir: Path) -> list:
    """Find all history.json files in a directory"""
    history_files = []
    
    # Search in outputs directory
    if search_dir.exists():
        for json_file in search_dir.rglob('history.json'):
            history_files.append(json_file)
        for json_file in search_dir.rglob('training_history.json'):
            history_files.append(json_file)
    
    return sorted(history_files)


def main():
    parser = argparse.ArgumentParser(description='Visualize training history from JSON files')
    parser.add_argument('--json', type=str, help='Path to history.json file')
    parser.add_argument('--output', type=str, help='Output path for plot (default: same as JSON with .png)')
    parser.add_argument('--outputs-dir', type=str, default='outputs', 
                       help='Directory to search for history files (default: outputs)')
    parser.add_argument('--list', action='store_true', 
                       help='List all available history files and exit')
    parser.add_argument('--all', action='store_true',
                       help='Generate plots for all found history files')
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        search_dir = Path(args.outputs_dir)
        history_files = find_history_files(search_dir)
        
        if not history_files:
            logger.info(f"No history files found in {search_dir}")
            return
        
        logger.info(f"\nFound {len(history_files)} history file(s):\n")
        for i, hist_file in enumerate(history_files, 1):
            logger.info(f"{i}. {hist_file}")
        return
    
    # Process all files
    if args.all:
        search_dir = Path(args.outputs_dir)
        history_files = find_history_files(search_dir)
        
        if not history_files:
            logger.info(f"No history files found in {search_dir}")
            return
        
        logger.info(f"Found {len(history_files)} history file(s), generating plots...\n")
        for hist_file in history_files:
            try:
                history = load_history(hist_file)
                output_path = hist_file.parent / 'training_curves.png'
                title = f"Training Curves - {hist_file.parent.name}"
                plot_training_curves(history, output_path, title)
                logger.info(f"✓ Generated plot for {hist_file}")
            except Exception as e:
                logger.error(f"✗ Failed to process {hist_file}: {e}")
        return
    
    # Single file mode
    if not args.json:
        logger.error("Please provide --json path, use --list to see available files, or --all to process all")
        return
    
    json_path = Path(args.json)
    if not json_path.exists():
        logger.error(f"History file not found: {json_path}")
        return
    
    # Load and plot
    try:
        history = load_history(json_path)
        logger.info(f"Loaded history from {json_path}")
        logger.info(f"  Epochs: {len(history['train_loss'])}")
        logger.info(f"  Keys: {list(history.keys())}")
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = json_path.parent / 'training_curves.png'
        
        # Generate title from directory name
        title = f"Training Curves - {json_path.parent.name}"
        
        plot_training_curves(history, output_path, title)
        logger.info(f"\n✓ Visualization complete!")
        
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        raise


if __name__ == "__main__":
    main()

