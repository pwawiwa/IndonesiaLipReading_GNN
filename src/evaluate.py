# Evaluation script for lip reading model
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from models.lip_reading_gnn import LipReadingGNN
from utils.dataset import LipReadingDataset
from torch_geometric.data import DataLoader

def load_model(checkpoint_path, num_classes):
    """Load trained model from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LipReadingGNN(
        num_node_features=3,
        num_classes=num_classes
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, device

def evaluate(model, test_loader, device, classes):
    """Evaluate model and return predictions and true labels."""
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.max(1)[1]
            
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(batch.y.cpu().numpy())
    
    return predictions, true_labels

def plot_confusion_matrix(true_labels, predictions, classes, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Setup paths
    checkpoint_path = 'checkpoints/best_model.pt'
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Load test dataset
    test_dataset = LipReadingDataset(root='data/processed', split='test')
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load model
    model, device = load_model(checkpoint_path, len(test_dataset.classes))
    
    # Evaluate
    predictions, true_labels = evaluate(model, test_loader, device, test_dataset.classes)
    
    # Generate and save confusion matrix
    plot_confusion_matrix(
        true_labels,
        predictions,
        test_dataset.classes,
        results_dir / 'confusion_matrix.png'
    )
    
    # Generate classification report
    report = classification_report(
        true_labels,
        predictions,
        target_names=test_dataset.classes,
        digits=3
    )
    
    # Save classification report
    with open(results_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    
    print('Evaluation completed. Results saved in results directory.')

if __name__ == '__main__':
    main()