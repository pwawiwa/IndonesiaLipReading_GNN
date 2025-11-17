"""
Evaluate v2 and v3 models on test set
"""
import torch
import torch.nn as nn
from pathlib import Path
from dataset.dataset import create_dataloaders
from models.combined import CombinedModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, config: dict, test_loader, device):
    """Evaluate a model on test set"""
    # Create model
    model = CombinedModel(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'],
        spatial_dim=config['spatial_dim'],
        temporal_dim=config['temporal_dim'],
        spatial_layers=config['spatial_layers'],
        temporal_layers=config['temporal_layers'],
        dropout=config['dropout']
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {Path(model_path).parent.parent.name}", leave=False):
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.squeeze().cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro
    }


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, test_loader, num_classes, label_map = create_dataloaders(
        train_pt='data/processed/train.pt',
        val_pt='data/processed/val.pt',
        test_pt='data/processed/test.pt',
        batch_size=32,
        num_workers=4
    )
    
    # Get input dim
    sample = next(iter(train_loader))
    input_dim = sample.x.shape[1]
    
    # V2 Config (from previous training - 256 dims, 3 layers)
    v2_config = {
        'input_dim': input_dim,
        'num_classes': num_classes,
        'spatial_dim': 256,
        'temporal_dim': 256,
        'spatial_layers': 3,
        'temporal_layers': 2,
        'dropout': 0.5
    }
    
    # V3 Config (current - 128 dims, 2 layers)
    v3_config = {
        'input_dim': input_dim,
        'num_classes': num_classes,
        'spatial_dim': 128,
        'temporal_dim': 128,
        'spatial_layers': 2,
        'temporal_layers': 2,
        'dropout': 0.7
    }
    
    # Model paths
    v2_model = 'outputs/v2/best_model.pth'
    v3_model = 'outputs/v3/best_model.pth'
    
    logger.info("="*60)
    logger.info("EVALUATING MODELS")
    logger.info("="*60)
    
    results = {}
    
    # Evaluate V2
    if Path(v2_model).exists():
        logger.info("\nEvaluating V2...")
        v2_results = evaluate_model(v2_model, v2_config, test_loader, device)
        
        # Count parameters
        v2_model_obj = CombinedModel(**v2_config)
        v2_params = count_parameters(v2_model_obj)
        
        results['v2'] = {
            'config': v2_config,
            'params': v2_params,
            'metrics': v2_results
        }
        logger.info(f"V2 Test Accuracy: {v2_results['accuracy']:.4f}")
    else:
        logger.warning(f"V2 model not found: {v2_model}")
    
    # Evaluate V3
    if Path(v3_model).exists():
        logger.info("\nEvaluating V3...")
        v3_results = evaluate_model(v3_model, v3_config, test_loader, device)
        
        # Count parameters
        v3_model_obj = CombinedModel(**v3_config)
        v3_params = count_parameters(v3_model_obj)
        
        results['v3'] = {
            'config': v3_config,
            'params': v3_params,
            'metrics': v3_results
        }
        logger.info(f"V3 Test Accuracy: {v3_results['accuracy']:.4f}")
    else:
        logger.warning(f"V3 model not found: {v3_model}")
    
    # Save results
    results_file = Path('outputs/model_comparison.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'V2':<25} {'V3':<25}")
    print("-"*80)
    
    if 'v2' in results and 'v3' in results:
        v2 = results['v2']
        v3 = results['v3']
        
        print(f"{'Parameters':<30} {v2['params']:<25,} {v3['params']:<25,}")
        print(f"{'Spatial Dim':<30} {v2['config']['spatial_dim']:<25} {v3['config']['spatial_dim']:<25}")
        print(f"{'Temporal Dim':<30} {v2['config']['temporal_dim']:<25} {v3['config']['temporal_dim']:<25}")
        print(f"{'Spatial Layers':<30} {v2['config']['spatial_layers']:<25} {v3['config']['spatial_layers']:<25}")
        print(f"{'Dropout':<30} {v2['config']['dropout']:<25} {v3['config']['dropout']:<25}")
        print("-"*80)
        print(f"{'Test Accuracy':<30} {v2['metrics']['accuracy']:<25.4f} {v3['metrics']['accuracy']:<25.4f}")
        print(f"{'Test F1 (Macro)':<30} {v2['metrics']['f1_macro']:<25.4f} {v3['metrics']['f1_macro']:<25.4f}")
        print(f"{'Test F1 (Weighted)':<30} {v2['metrics']['f1_weighted']:<25.4f} {v3['metrics']['f1_weighted']:<25.4f}")
        print(f"{'Test Precision (Macro)':<30} {v2['metrics']['precision_macro']:<25.4f} {v3['metrics']['precision_macro']:<25.4f}")
        print(f"{'Test Recall (Macro)':<30} {v2['metrics']['recall_macro']:<25.4f} {v3['metrics']['recall_macro']:<25.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()



