#!/usr/bin/env python3
"""
Generate comprehensive Excel summary from all training runs.
Creates multiple sheets with different views of the results.
"""
import torch
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, List, Optional
import re
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_run_info(result_dir: Path) -> Optional[Dict]:
    """Extract information from a single run directory."""
    run_meta_file = result_dir / 'run_meta.pt'
    summary_file = result_dir / 'summary.txt'
    config_file = result_dir / 'config.yaml'
    
    info = {
        'result_path': str(result_dir),
        'run_meta_exists': run_meta_file.exists(),
        'summary_exists': summary_file.exists(),
        'config_exists': config_file.exists(),
    }
    
    # Try to load run_meta.pt
    if run_meta_file.exists():
        try:
            run_meta = torch.load(run_meta_file, map_location='cpu')
            
            # Extract key metrics
            info.update({
                'partition': run_meta.get('partition', 'unknown'),
                'feature_level': run_meta.get('feature_level', 'unknown'),
                'model': run_meta.get('model', 'unknown'),
                'seed': run_meta.get('seed', 0),
                'best_val_acc': run_meta.get('best_val_acc', 0.0),
                'test_acc': run_meta.get('test_acc', None),
                'test_loss': run_meta.get('test_loss', None),
                'best_epoch': run_meta.get('best_epoch', 0),
                'num_parameters': run_meta.get('num_parameters', 0),
                'git_hash': run_meta.get('git_hash', 'unknown'),
            })
            
            # Extract timestamps
            # Try to get from system_info first
            if 'system_info' in run_meta:
                sys_info = run_meta['system_info']
                # Extract timestamp (ISO format)
                if 'timestamp' in sys_info:
                    try:
                        timestamp_str = sys_info['timestamp']
                        # Parse ISO format timestamp
                        if isinstance(timestamp_str, str):
                            try:
                                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                info['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                                info['timestamp_datetime'] = dt  # For sorting
                            except:
                                info['timestamp'] = timestamp_str
                                info['timestamp_datetime'] = None
                        else:
                            info['timestamp'] = str(timestamp_str)
                            info['timestamp_datetime'] = None
                    except:
                        pass
                if 'start_time' in sys_info:
                    try:
                        if isinstance(sys_info['start_time'], str):
                            info['start_time'] = sys_info['start_time']
                        else:
                            info['start_time'] = str(sys_info['start_time'])
                    except:
                        pass
                if 'end_time' in sys_info:
                    try:
                        if isinstance(sys_info['end_time'], str):
                            info['end_time'] = sys_info['end_time']
                        else:
                            info['end_time'] = str(sys_info['end_time'])
                    except:
                        pass
            
            # Get file modification time as fallback (if timestamp not available)
            if 'timestamp_datetime' not in info or info.get('timestamp_datetime') is None:
                try:
                    mtime = os.path.getmtime(run_meta_file)
                    file_datetime = datetime.fromtimestamp(mtime)
                    if 'timestamp' not in info:
                        info['timestamp'] = file_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    info['file_modified'] = file_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    info['timestamp_datetime'] = file_datetime  # For sorting
                except:
                    if 'timestamp' not in info:
                        info['timestamp'] = None
                    info['file_modified'] = None
                    info['timestamp_datetime'] = None
            
            # Try to get from config if available
            if 'config' in run_meta and 'training' in run_meta['config']:
                training = run_meta['config']['training']
                if 'start_time' in training:
                    info['start_time'] = training['start_time']
                if 'end_time' in training:
                    info['end_time'] = training['end_time']
            
            # Extract config if available
            if 'config' in run_meta:
                config = run_meta['config']
                
                # Training config
                if 'training' in config:
                    training = config['training']
                    info.update({
                        'learning_rate': training.get('learning_rate', None),
                        'batch_size': training.get('batch_size', None),
                        'epochs': training.get('epochs', None),
                        'weight_decay': training.get('weight_decay', None),
                        'optimizer': training.get('optimizer', None),
                        'label_smoothing': training.get('label_smoothing', 0.0),
                        'early_stopping_patience': training.get('early_stopping_patience', None),
                        'gradient_clip': training.get('gradient_clip', None),
                    })
                    
                    # Scheduler config
                    if 'scheduler' in training:
                        scheduler = training['scheduler']
                        info.update({
                            'scheduler_name': scheduler.get('name', None),
                            'scheduler_step_size': scheduler.get('step_size', None),
                            'scheduler_gamma': scheduler.get('gamma', None),
                        })
                
                # Model config
                if 'model' in config:
                    model_cfg = config['model']
                    info['model_name_config'] = model_cfg.get('name', None)
                    
                    if 'params' in model_cfg:
                        params = model_cfg['params']
                        info.update({
                            'hidden_dim': params.get('hidden_dim', None),
                            'num_layers': params.get('num_layers', None),
                            'num_gin_layers': params.get('num_gin_layers', None),
                            'num_gcn_layers': params.get('num_gcn_layers', None),
                            'num_sage_layers': params.get('num_sage_layers', None),
                            'num_lstm_layers': params.get('num_lstm_layers', None),
                            'num_gru_layers': params.get('num_gru_layers', None),
                            'dropout': params.get('dropout', None),
                            'bidirectional': params.get('bidirectional', False),
                            'temporal_pool': params.get('temporal_pool', None),
                            'aggregator': params.get('aggregator', None),
                            'num_heads': params.get('num_heads', None),
                            'alpha': params.get('alpha', None),  # For AdaptiveGCN
                            'eps': params.get('eps', None),  # For GIN
                            'train_eps': params.get('train_eps', False),  # For GIN
                        })
                
                # Data config
                if 'data' in config:
                    data_cfg = config['data']
                    info.update({
                        'data_partition': data_cfg.get('partition', None),
                        'data_feature_level': data_cfg.get('feature_level', None),
                    })
                
                # Store full config as JSON string for reference
                import json
                try:
                    info['config_json'] = json.dumps(config, indent=2, default=str)
                except:
                    info['config_json'] = str(config)
        except Exception as e:
            info['error'] = str(e)
    
    # Try to load summary.txt as fallback
    elif summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
                # Extract key values using regex
                patterns = {
                    'partition': r'Partition:\s*(\w+)',
                    'feature_level': r'Feature Level:\s*(\w+)',
                    'model': r'Model:\s*(\w+)',
                    'seed': r'Seed:\s*(\d+)',
                    'best_val_acc': r'Best Val Accuracy:\s*([\d.]+)%',
                    'test_acc': r'Test Accuracy:\s*([\d.]+)%',
                    'current_train_loss': r'Current Train Loss:\s*([\d.]+)',
                    'current_val_loss': r'Current Val Loss:\s*([\d.]+)',
                    'current_train_acc': r'Current Train Acc:\s*([\d.]+)%',
                    'current_val_acc': r'Current Val Acc:\s*([\d.]+)%',
                }
                for key, pattern in patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        value = match.group(1)
                        if key in ['seed']:
                            info[key] = int(value)
                        elif key in ['best_val_acc', 'test_acc', 'current_train_loss', 'current_val_loss', 
                                     'current_train_acc', 'current_val_acc']:
                            info[key] = float(value)
                        else:
                            info[key] = value
        except Exception as e:
            info['error'] = str(e)
    
    # Extract from path if metadata not available
    if 'partition' not in info or info['partition'] == 'unknown':
        parts = result_dir.parts
        if 'results' in parts:
            idx = parts.index('results')
            if idx + 1 < len(parts):
                info['partition'] = parts[idx + 1]
            if idx + 2 < len(parts):
                info['feature_level'] = parts[idx + 2]
            if idx + 3 < len(parts):
                info['model'] = parts[idx + 3]
            if idx + 4 < len(parts):
                seed_str = parts[idx + 4]
                if seed_str.startswith('seed_'):
                    info['seed'] = int(seed_str.split('_')[1])
    
    return info if info.get('run_meta_exists') or info.get('summary_exists') else None


def collect_all_results(results_dir: Path) -> List[Dict]:
    """Collect all results from results directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory not found: {results_path}")
        return results
    
    # Find all seed_* directories
    for seed_dir in results_path.rglob('seed_*'):
        if seed_dir.is_dir():
            info = extract_run_info(seed_dir)
            if info:
                results.append(info)
    
    return results


def create_summary_sheets(results: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Create multiple summary sheets from results."""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    sheets = {}
    
    # Sheet 1: All Results (Complete) - sorted by date (newest first), then by partition/feature/model
    sort_cols = []
    if 'timestamp_datetime' in df.columns:
        sort_cols.append('timestamp_datetime')
    elif 'file_modified_timestamp' in df.columns:
        sort_cols.append('file_modified_timestamp')
    elif 'start_time' in df.columns:
        sort_cols.append('start_time')
    sort_cols.extend(['partition', 'feature_level', 'model', 'seed'])
    sort_cols = [col for col in sort_cols if col in df.columns]
    sheets['All_Results'] = df.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    
    # Sheet 2: Best by Feature Level - sorted by date
    if 'feature_level' in df.columns and 'best_val_acc' in df.columns:
        best_by_feature = df.loc[df.groupby(['partition', 'feature_level'])['best_val_acc'].idxmax()]
        sort_cols = []
        if 'timestamp_datetime' in best_by_feature.columns:
            sort_cols.append('timestamp_datetime')
        elif 'file_modified_timestamp' in best_by_feature.columns:
            sort_cols.append('file_modified_timestamp')
        sort_cols.extend(['partition', 'feature_level'])
        sort_cols = [col for col in sort_cols if col in best_by_feature.columns]
        sheets['Best_by_FeatureLevel'] = best_by_feature.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    
    # Sheet 3: Best by Model - sorted by date
    if 'model' in df.columns and 'best_val_acc' in df.columns:
        best_by_model = df.loc[df.groupby(['partition', 'model'])['best_val_acc'].idxmax()]
        sort_cols = []
        if 'timestamp_datetime' in best_by_model.columns:
            sort_cols.append('timestamp_datetime')
        elif 'file_modified_timestamp' in best_by_model.columns:
            sort_cols.append('file_modified_timestamp')
        sort_cols.extend(['partition', 'model'])
        sort_cols = [col for col in sort_cols if col in best_by_model.columns]
        sheets['Best_by_Model'] = best_by_model.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    
    # Sheet 4: Comparison Table (pivot)
    if 'feature_level' in df.columns and 'model' in df.columns and 'best_val_acc' in df.columns:
        pivot_data = df.pivot_table(
            values='best_val_acc',
            index='model',
            columns='feature_level',
            aggfunc='max'
        )
        sheets['Comparison_Matrix'] = pivot_data
    
    # Sheet 5: Hyperparameters Summary - sorted by date
    hyperparam_cols = ['learning_rate', 'batch_size', 'epochs', 'weight_decay', 'optimizer',
                       'scheduler_name', 'scheduler_step_size', 'scheduler_gamma',
                       'hidden_dim', 'dropout', 'bidirectional', 'label_smoothing', 
                       'early_stopping_patience', 'gradient_clip']
    hyperparam_cols = [col for col in hyperparam_cols if col in df.columns]
    if hyperparam_cols:
        base_cols = ['partition', 'feature_level', 'model', 'seed']
        if 'timestamp' in df.columns:
            base_cols.insert(0, 'timestamp')
        elif 'file_modified' in df.columns:
            base_cols.insert(0, 'file_modified')
        hyperparam_df = df[base_cols + hyperparam_cols].copy()
        sort_cols = []
        if 'timestamp_datetime' in hyperparam_df.columns:
            sort_cols.append('timestamp_datetime')
        elif 'file_modified_timestamp' in hyperparam_df.columns:
            sort_cols.append('file_modified_timestamp')
        sort_cols.extend(['partition', 'feature_level', 'model'])
        sort_cols = [col for col in sort_cols if col in hyperparam_df.columns]
        sheets['Hyperparameters'] = hyperparam_df.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    
    # Sheet 6: Model Architecture Details - sorted by date
    model_arch_cols = ['num_layers', 'num_gin_layers', 'num_gcn_layers', 'num_sage_layers',
                       'num_lstm_layers', 'num_gru_layers', 'hidden_dim', 'dropout',
                       'bidirectional', 'temporal_pool', 'aggregator', 'num_heads', 'alpha', 'eps', 'train_eps']
    model_arch_cols = [col for col in model_arch_cols if col in df.columns]
    if model_arch_cols:
        base_cols = ['partition', 'feature_level', 'model', 'seed']
        if 'timestamp' in df.columns:
            base_cols.insert(0, 'timestamp')
        elif 'file_modified' in df.columns:
            base_cols.insert(0, 'file_modified')
        arch_df = df[base_cols + model_arch_cols].copy()
        sort_cols = []
        if 'timestamp_datetime' in arch_df.columns:
            sort_cols.append('timestamp_datetime')
        elif 'file_modified_timestamp' in arch_df.columns:
            sort_cols.append('file_modified_timestamp')
        sort_cols.extend(['partition', 'feature_level', 'model'])
        sort_cols = [col for col in sort_cols if col in arch_df.columns]
        sheets['Model_Architecture'] = arch_df.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    
    # Sheet 7: Test Results (if available) - sorted by date
    if 'test_acc' in df.columns:
        test_df = df[df['test_acc'].notna()].copy()
        if not test_df.empty:
            sort_cols = []
            if 'timestamp_datetime' in test_df.columns:
                sort_cols.append('timestamp_datetime')
            elif 'file_modified_timestamp' in test_df.columns:
                sort_cols.append('file_modified_timestamp')
            sort_cols.extend(['partition', 'feature_level', 'model'])
            sort_cols = [col for col in sort_cols if col in test_df.columns]
            sheets['Test_Results'] = test_df.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    
    # Sheet 8: Full Config (JSON format for reference) - sorted by date
    if 'config_json' in df.columns:
        base_cols = ['partition', 'feature_level', 'model', 'seed', 'config_json']
        if 'timestamp' in df.columns:
            base_cols.insert(0, 'timestamp')
        elif 'file_modified' in df.columns:
            base_cols.insert(0, 'file_modified')
        config_df = df[base_cols].copy()
        sort_cols = []
        if 'timestamp_datetime' in config_df.columns:
            sort_cols.append('timestamp_datetime')
        elif 'file_modified_timestamp' in config_df.columns:
            sort_cols.append('file_modified_timestamp')
        sort_cols.extend(['partition', 'feature_level', 'model'])
        sort_cols = [col for col in sort_cols if col in config_df.columns]
        sheets['Full_Configs'] = config_df.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))
    
    # Sheet 9: Statistics Summary
    if 'best_val_acc' in df.columns:
        stats_data = []
        for partition in df['partition'].unique():
            for fe_level in df['feature_level'].unique():
                subset = df[(df['partition'] == partition) & (df['feature_level'] == fe_level)]
                if not subset.empty:
                    stats_data.append({
                        'partition': partition,
                        'feature_level': fe_level,
                        'num_runs': len(subset),
                        'mean_acc': subset['best_val_acc'].mean(),
                        'std_acc': subset['best_val_acc'].std(),
                        'max_acc': subset['best_val_acc'].max(),
                        'min_acc': subset['best_val_acc'].min(),
                        'best_model': subset.loc[subset['best_val_acc'].idxmax(), 'model'] if 'model' in subset.columns else 'N/A',
                    })
        if stats_data:
            sheets['Statistics'] = pd.DataFrame(stats_data)
    
    return sheets


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Excel summary from all training runs')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory path')
    parser.add_argument('--output', type=str, default='results/training_summary.xlsx',
                       help='Output Excel file path')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_file = Path(args.output)
    
    print("=" * 80)
    print("GENERATING EXCEL SUMMARY")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Collect all results
    print("Collecting results...")
    results = collect_all_results(results_dir)
    print(f"Found {len(results)} training runs")
    
    if not results:
        print("No results found!")
        return
    
    # Create summary sheets
    print("Creating summary sheets...")
    sheets = create_summary_sheets(results)
    
    if not sheets:
        print("No sheets created!")
        return
    
    # Write to Excel
    print(f"Writing to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df_sheet in sheets.items():
            # Remove internal timestamp columns before writing (keep only display columns)
            df_to_write = df_sheet.copy()
            if 'timestamp_datetime' in df_to_write.columns:
                df_to_write = df_to_write.drop(columns=['timestamp_datetime'])
            if 'file_modified_timestamp' in df_to_write.columns:
                df_to_write = df_to_write.drop(columns=['file_modified_timestamp'])
            
            print(f"  Writing sheet: {sheet_name} ({len(df_to_write)} rows)")
            df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print()
    print("=" * 80)
    print("✅ EXCEL SUMMARY GENERATED")
    print("=" * 80)
    print(f"File: {output_file}")
    print(f"Sheets: {', '.join(sheets.keys())}")
    print()
    print("Sheet Descriptions:")
    print("-" * 80)
    print("1. All_Results: Complete list of all training runs with all metrics")
    print("2. Best_by_FeatureLevel: Best model for each feature level")
    print("3. Best_by_Model: Best result for each model type")
    print("4. Comparison_Matrix: Pivot table (Model × Feature Level)")
    print("5. Hyperparameters: Training hyperparameters (LR, batch size, optimizer, etc.)")
    print("6. Model_Architecture: Model architecture details (layers, dropout, etc.)")
    print("7. Test_Results: Test set results (if available)")
    print("8. Full_Configs: Complete config JSON for each run (for reference)")
    print("9. Statistics: Statistical summary per feature level")


if __name__ == '__main__':
    main()

