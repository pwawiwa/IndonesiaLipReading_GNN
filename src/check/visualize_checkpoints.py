"""
Script untuk visualisasi dan analisis checkpoint yang lengkap
"""
import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def format_size(size_bytes):
    """Format file size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def load_checkpoint_info(checkpoint_path):
    """Load checkpoint dan extract informasi"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        file_size = checkpoint_path.stat().st_size
        
        info = {
            'path': str(checkpoint_path),
            'name': checkpoint_path.name,
            'file_size': file_size,
            'file_size_str': format_size(file_size),
            'epoch': checkpoint.get('epoch', 'N/A'),
            'best_val_acc': checkpoint.get('best_val_acc', 0.0),
            'best_val_f1': checkpoint.get('best_val_f1', 0.0),
            'has_history': 'history' in checkpoint,
            'has_model': 'model_state_dict' in checkpoint,
            'has_optimizer': 'optimizer_state_dict' in checkpoint,
            'has_scheduler': 'scheduler_state_dict' in checkpoint,
        }
        
        # Extract history info
        if 'history' in checkpoint:
            history = checkpoint['history']
            info['num_epochs_trained'] = len(history.get('train_loss', []))
            info['final_train_loss'] = history['train_loss'][-1] if history.get('train_loss') else None
            info['final_train_acc'] = history['train_acc'][-1] if history.get('train_acc') else None
            info['final_val_loss'] = history['val_loss'][-1] if history.get('val_loss') else None
            info['final_val_acc'] = history['val_acc'][-1] if history.get('val_acc') else None
            info['final_val_f1'] = history['val_f1'][-1] if history.get('val_f1') else None
            info['max_val_acc'] = max(history['val_acc']) if history.get('val_acc') else None
            info['max_val_f1'] = max(history['val_f1']) if history.get('val_f1') else None
        else:
            info['num_epochs_trained'] = 0
        
        # Model parameter count
        if 'model_state_dict' in checkpoint:
            total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
            trainable_params = sum(p.numel() for p in checkpoint['model_state_dict'].values() if p.requires_grad)
            info['total_params'] = total_params
            info['trainable_params'] = trainable_params
        else:
            info['total_params'] = 0
            info['trainable_params'] = 0
        
        return info, checkpoint
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None, None


def analyze_checkpoints(checkpoint_dir):
    """Analyze semua checkpoint di directory"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = sorted(checkpoint_dir.glob('*.pth'))
    
    print("="*80)
    print("CHECKPOINT ANALYSIS")
    print("="*80)
    print(f"\nDirectory: {checkpoint_dir}")
    print(f"Found {len(checkpoint_files)} checkpoint files\n")
    
    all_info = []
    all_checkpoints = {}
    
    for ckpt_path in checkpoint_files:
        info, checkpoint = load_checkpoint_info(ckpt_path)
        if info:
            all_info.append(info)
            all_checkpoints[ckpt_path.name] = checkpoint
    
    if not all_info:
        print("No valid checkpoints found!")
        return None, None
    
    # Create DataFrame
    df = pd.DataFrame(all_info)
    
    # Print summary table
    print("\n" + "="*80)
    print("CHECKPOINT SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    csv_path = checkpoint_dir / 'checkpoint_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary saved to: {csv_path}")
    
    return df, all_checkpoints


def plot_checkpoint_comparison(df, output_dir):
    """Plot perbandingan checkpoint"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Checkpoint Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Filter valid data
    valid_df = df[df['epoch'] != 'N/A'].copy()
    valid_df['epoch'] = pd.to_numeric(valid_df['epoch'], errors='coerce')
    valid_df = valid_df.dropna(subset=['epoch'])
    
    if len(valid_df) == 0:
        print("No valid epoch data for comparison plots")
        return
    
    # 1. File Size vs Epoch
    ax = axes[0, 0]
    ax.scatter(valid_df['epoch'], valid_df['file_size'] / (1024**2), alpha=0.7, s=100)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('File Size (MB)')
    ax.set_title('Checkpoint File Size vs Epoch')
    ax.grid(True, alpha=0.3)
    
    # 2. Best Val Acc vs Epoch
    ax = axes[0, 1]
    valid_best = valid_df[valid_df['best_val_acc'] > 0]
    if len(valid_best) > 0:
        ax.plot(valid_best['epoch'], valid_best['best_val_acc'], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best Val Accuracy')
        ax.set_title('Best Validation Accuracy vs Epoch')
        ax.grid(True, alpha=0.3)
    
    # 3. Best Val F1 vs Epoch
    ax = axes[0, 2]
    valid_f1 = valid_df[valid_df['best_val_f1'] > 0]
    if len(valid_f1) > 0:
        ax.plot(valid_f1['epoch'], valid_f1['best_val_f1'], 'o-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Best Val F1 Score')
        ax.set_title('Best Validation F1 Score vs Epoch')
        ax.grid(True, alpha=0.3)
    
    # 4. Final Metrics Comparison
    ax = axes[1, 0]
    metrics_to_plot = ['final_train_acc', 'final_val_acc']
    for metric in metrics_to_plot:
        valid_metric = valid_df[valid_df[metric].notna()]
        if len(valid_metric) > 0:
            ax.plot(valid_metric['epoch'], valid_metric[metric], 'o-', label=metric.replace('final_', '').replace('_', ' ').title(), linewidth=2, markersize=6)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Final Train vs Val Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Model Parameters
    ax = axes[1, 1]
    valid_params = valid_df[valid_df['total_params'] > 0]
    if len(valid_params) > 0:
        ax.bar(range(len(valid_params)), valid_params['total_params'] / 1e6, alpha=0.7)
        ax.set_xlabel('Checkpoint Index')
        ax.set_ylabel('Parameters (Millions)')
        ax.set_title('Model Parameter Count')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Training Progress (Epochs Trained)
    ax = axes[1, 2]
    valid_progress = valid_df[valid_df['num_epochs_trained'] > 0]
    if len(valid_progress) > 0:
        ax.plot(valid_progress['epoch'], valid_progress['num_epochs_trained'], 'o-', linewidth=2, markersize=8, color='purple')
        ax.set_xlabel('Checkpoint Epoch')
        ax.set_ylabel('Epochs Trained (in History)')
        ax.set_title('Training Progress')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'checkpoint_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    plt.close()


def plot_training_curves_from_checkpoints(checkpoints_dict, output_dir):
    """Plot training curves dari semua checkpoint"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Training Curves from All Checkpoints', fontsize=16, fontweight='bold')
    
    all_histories = []
    checkpoint_names = []
    
    for name, checkpoint in checkpoints_dict.items():
        if 'history' in checkpoint:
            history = checkpoint['history']
            epoch = checkpoint.get('epoch', 0)
            
            # Filter out NaN values
            train_loss = [x for x in history.get('train_loss', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
            train_acc = [x for x in history.get('train_acc', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
            val_loss = [x for x in history.get('val_loss', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
            val_acc = [x for x in history.get('val_acc', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
            
            if len(train_loss) > 0:
                all_histories.append({
                    'name': name,
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                })
                checkpoint_names.append(name)
    
    if not all_histories:
        print("No valid training histories found in checkpoints")
        return
    
    # Sort by epoch
    all_histories.sort(key=lambda x: x['epoch'])
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    for hist in all_histories:
        epochs = range(1, len(hist['train_loss']) + 1)
        label = f"Epoch {hist['epoch']}" if hist['epoch'] != 'N/A' else hist['name']
        ax.plot(epochs, hist['train_loss'], alpha=0.6, linewidth=1.5, label=f"Train - {label}")
        if len(hist['val_loss']) > 0:
            val_epochs = range(1, len(hist['val_loss']) + 1)
            ax.plot(val_epochs, hist['val_loss'], '--', alpha=0.6, linewidth=1.5, label=f"Val - {label}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax = axes[0, 1]
    for hist in all_histories:
        epochs = range(1, len(hist['train_acc']) + 1)
        label = f"Epoch {hist['epoch']}" if hist['epoch'] != 'N/A' else hist['name']
        ax.plot(epochs, hist['train_acc'], alpha=0.6, linewidth=1.5, label=f"Train - {label}")
        if len(hist['val_acc']) > 0:
            val_epochs = range(1, len(hist['val_acc']) + 1)
            ax.plot(val_epochs, hist['val_acc'], '--', alpha=0.6, linewidth=1.5, label=f"Val - {label}")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Combined best checkpoint (latest or best)
    ax = axes[1, 0]
    if all_histories:
        latest = all_histories[-1]  # Latest checkpoint
        epochs = range(1, len(latest['train_loss']) + 1)
        ax.plot(epochs, latest['train_loss'], 'b-', linewidth=2, label='Train Loss')
        if len(latest['val_loss']) > 0:
            val_epochs = range(1, len(latest['val_loss']) + 1)
            ax.plot(val_epochs, latest['val_loss'], 'r-', linewidth=2, label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Latest Checkpoint Training Curves (Epoch {latest["epoch"]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Combined accuracy
    ax = axes[1, 1]
    if all_histories:
        latest = all_histories[-1]
        epochs = range(1, len(latest['train_acc']) + 1)
        ax.plot(epochs, latest['train_acc'], 'b-', linewidth=2, label='Train Acc')
        if len(latest['val_acc']) > 0:
            val_epochs = range(1, len(latest['val_acc']) + 1)
            ax.plot(val_epochs, latest['val_acc'], 'r-', linewidth=2, label='Val Acc')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Latest Checkpoint Accuracy Curves (Epoch {latest["epoch"]})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'training_curves_all.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves plot saved to: {output_path}")
    plt.close()


def plot_detailed_metrics(checkpoints_dict, output_dir):
    """Plot detailed metrics analysis"""
    # Find best checkpoint
    best_checkpoint = None
    best_val_acc = 0
    
    for name, checkpoint in checkpoints_dict.items():
        if 'history' in checkpoint:
            history = checkpoint['history']
            if history.get('val_acc'):
                max_acc = max([x for x in history['val_acc'] if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))])
                if max_acc > best_val_acc:
                    best_val_acc = max_acc
                    best_checkpoint = (name, checkpoint)
    
    if not best_checkpoint:
        print("No valid checkpoint found for detailed metrics")
        return
    
    name, checkpoint = best_checkpoint
    history = checkpoint['history']
    
    # Filter NaN values
    train_loss = [x for x in history.get('train_loss', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    train_acc = [x for x in history.get('train_acc', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    val_loss = [x for x in history.get('val_loss', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    val_acc = [x for x in history.get('val_acc', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    val_f1 = [x for x in history.get('val_f1', []) if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'Detailed Training Metrics - Best Checkpoint: {name} (Max Val Acc: {best_val_acc:.4f})', 
                 fontsize=16, fontweight='bold')
    
    epochs = range(1, len(train_loss) + 1)
    
    # Loss plot
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    if len(val_loss) > 0:
        ax.plot(epochs[:len(val_loss)], val_loss, 'r-', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Acc')
    if len(val_acc) > 0:
        ax.plot(epochs[:len(val_acc)], val_acc, 'r-', linewidth=2, label='Val Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score plot
    ax = axes[1, 0]
    if len(val_f1) > 0:
        ax.plot(epochs[:len(val_f1)], val_f1, 'g-', linewidth=2, label='Val F1')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Generalization gap
    ax = axes[1, 1]
    if len(val_loss) > 0 and len(train_loss) > 0:
        min_len = min(len(val_loss), len(train_loss))
        gap = [v - t for v, t in zip(val_loss[:min_len], train_loss[:min_len])]
        gap_epochs = range(1, min_len + 1)
        ax.plot(gap_epochs, gap, 'purple', linewidth=2, label='Val Loss - Train Loss')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Generalization Gap')
        ax.set_title('Generalization Gap (Val Loss - Train Loss)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'detailed_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Detailed metrics plot saved to: {output_path}")
    plt.close()


def analyze_model_parameters(checkpoints_dict, output_dir):
    """Analyze model parameters dari checkpoints"""
    param_info = []
    
    for name, checkpoint in checkpoints_dict.items():
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Count parameters by layer
            layer_params = {}
            total = 0
            for key, param in state_dict.items():
                num_params = param.numel()
                layer_name = key.split('.')[0]  # First part of key
                layer_params[layer_name] = layer_params.get(layer_name, 0) + num_params
                total += num_params
            
            param_info.append({
                'checkpoint': name,
                'total_params': total,
                'layers': layer_params
            })
    
    if not param_info:
        print("No model parameters found in checkpoints")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Parameter Analysis', fontsize=16, fontweight='bold')
    
    # Total parameters comparison
    ax = axes[0]
    names = [info['checkpoint'] for info in param_info]
    totals = [info['total_params'] / 1e6 for info in param_info]  # Convert to millions
    ax.barh(range(len(names)), totals, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[:30] + '...' if len(n) > 30 else n for n in names], fontsize=8)
    ax.set_xlabel('Total Parameters (Millions)')
    ax.set_title('Total Parameters per Checkpoint')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Layer-wise breakdown (for first checkpoint)
    if param_info:
        ax = axes[1]
        first_info = param_info[0]
        layers = list(first_info['layers'].keys())
        counts = [first_info['layers'][layer] / 1e6 for layer in layers]
        ax.barh(range(len(layers)), counts, alpha=0.7)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=8)
        ax.set_xlabel('Parameters (Millions)')
        ax.set_title(f'Layer-wise Parameter Breakdown\n({first_info["checkpoint"]})')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / 'model_parameters.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Model parameters plot saved to: {output_path}")
    plt.close()


def generate_report(df, checkpoints_dict, output_dir):
    """Generate text report"""
    report_path = output_dir / 'checkpoint_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHECKPOINT ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Checkpoints: {len(df)}\n")
        f.write(f"Total Size: {format_size(df['file_size'].sum())}\n\n")
        
        # Best checkpoint
        valid_df = df[df['best_val_acc'] > 0]
        if len(valid_df) > 0:
            best = valid_df.loc[valid_df['best_val_acc'].idxmax()]
            f.write("BEST CHECKPOINT\n")
            f.write("-"*80 + "\n")
            f.write(f"Name: {best['name']}\n")
            f.write(f"Epoch: {best['epoch']}\n")
            f.write(f"Best Val Acc: {best['best_val_acc']:.4f}\n")
            f.write(f"Best Val F1: {best['best_val_f1']:.4f}\n")
            f.write(f"File Size: {best['file_size_str']}\n\n")
        
        # Latest checkpoint
        valid_epoch = df[df['epoch'] != 'N/A'].copy()
        valid_epoch['epoch'] = pd.to_numeric(valid_epoch['epoch'], errors='coerce')
        if len(valid_epoch) > 0:
            latest = valid_epoch.loc[valid_epoch['epoch'].idxmax()]
            f.write("LATEST CHECKPOINT\n")
            f.write("-"*80 + "\n")
            f.write(f"Name: {latest['name']}\n")
            f.write(f"Epoch: {latest['epoch']}\n")
            f.write(f"Epochs Trained: {latest['num_epochs_trained']}\n")
            if latest['final_val_acc'] is not None:
                f.write(f"Final Val Acc: {latest['final_val_acc']:.4f}\n")
            if latest['final_val_f1'] is not None:
                f.write(f"Final Val F1: {latest['final_val_f1']:.4f}\n")
            f.write(f"File Size: {latest['file_size_str']}\n\n")
        
        # All checkpoints
        f.write("ALL CHECKPOINTS\n")
        f.write("-"*80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Report saved to: {report_path}")


def main():
    """Main function"""
    checkpoint_dir = Path('outputs/combined/checkpoints')
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Create output directory for visualizations
    viz_dir = checkpoint_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("CHECKPOINT VISUALIZATION TOOL")
    print("="*80 + "\n")
    
    # Analyze checkpoints
    df, checkpoints_dict = analyze_checkpoints(checkpoint_dir)
    
    if df is None:
        return
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_checkpoint_comparison(df, viz_dir)
    plot_training_curves_from_checkpoints(checkpoints_dict, viz_dir)
    plot_detailed_metrics(checkpoints_dict, viz_dir)
    analyze_model_parameters(checkpoints_dict, viz_dir)
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80 + "\n")
    
    generate_report(df, checkpoints_dict, viz_dir)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {viz_dir}")
    print("\nGenerated files:")
    print("  - checkpoint_summary.csv")
    print("  - checkpoint_comparison.png")
    print("  - training_curves_all.png")
    print("  - detailed_metrics.png")
    print("  - model_parameters.png")
    print("  - checkpoint_report.txt")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

