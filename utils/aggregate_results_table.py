#!/usr/bin/env python3
"""
Generate summary table with all configs and results.
"""
import argparse
import torch
from pathlib import Path
import pandas as pd
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config


def collect_results(results_dir, partition):
    """Collect all results from the results directory."""
    results_dir = Path(results_dir)
    partition_dir = results_dir / partition
    
    if not partition_dir.exists():
        print(f"Partition directory not found: {partition_dir}")
        return []
    
    results = []
    
    # Iterate through feature levels
    for fe_level_dir in sorted(partition_dir.iterdir()):
        if not fe_level_dir.is_dir():
            continue
        
        fe_level = fe_level_dir.name
        
        # Iterate through models
        for model_dir in sorted(fe_level_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            # Look for seed directories
            for seed_dir in sorted(model_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                
                run_meta_file = seed_dir / 'run_meta.pt'
                if not run_meta_file.exists():
                    continue
                
                try:
                    # Load run metadata
                    run_meta = torch.load(run_meta_file)
                    
                    # Get config from run_meta (it's saved there)
                    config = run_meta.get('config', None)
                    
                    # Extract information
                    result = {
                        'partition': partition,
                        'feature_level': fe_level,
                        'model': model_dir.name,
                        'seed': run_meta.get('seed', 0),
                        'best_val_acc': run_meta.get('best_val_acc', 0.0),
                        'test_acc': run_meta.get('test_acc', 0.0),
                        'test_loss': run_meta.get('test_loss', 0.0),
                        'best_epoch': run_meta.get('best_epoch', 0),
                        'num_parameters': run_meta.get('num_parameters', 0),
                    }
                    
                    # Add config details if available
                    if config:
                        result['learning_rate'] = config['training'].get('learning_rate', None)
                        result['batch_size'] = config['training'].get('batch_size', None)
                        result['optimizer'] = config['training'].get('optimizer', None)
                        result['weight_decay'] = config['training'].get('weight_decay', None)
                        result['epochs'] = config['training'].get('epochs', None)
                        
                        # Model-specific params
                        if 'params' in config['model']:
                            model_params = config['model']['params']
                            result['hidden_dim'] = model_params.get('hidden_dim', None)
                            result['num_layers'] = model_params.get('num_layers', None)
                            result['dropout'] = model_params.get('dropout', None)
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error loading {run_meta_file}: {e}")
                    continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate summary table')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Results directory')
    parser.add_argument('--partition', type=str, required=True,
                       help='Partition name')
    parser.add_argument('--output', type=str, required=True,
                       help='Output .pt file path')
    parser.add_argument('--output-csv', type=str, required=True,
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Collect results
    print("Collecting results...")
    results = collect_results(args.results_dir, args.partition)
    
    if len(results) == 0:
        print("No results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by feature level and test accuracy
    df = df.sort_values(['feature_level', 'test_acc'], ascending=[True, False])
    
    # Save as .pt (torch format)
    output_pt = Path(args.output)
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'results': results,
        'dataframe': df.to_dict('records')
    }, output_pt)
    print(f"Saved summary to: {output_pt}")
    
    # Save as CSV
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to: {output_csv}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"\nTotal runs: {len(results)}")
    print(f"\nBest model per feature level:")
    for fe_level in sorted(df['feature_level'].unique()):
        fe_df = df[df['feature_level'] == fe_level]
        best = fe_df.loc[fe_df['test_acc'].idxmax()]
        print(f"  {fe_level}: {best['model']} - Test Acc: {best['test_acc']:.2f}%")
    
    print(f"\nOverall best: {df.loc[df['test_acc'].idxmax(), 'model']} "
          f"({df.loc[df['test_acc'].idxmax(), 'feature_level']}) - "
          f"Test Acc: {df['test_acc'].max():.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()

