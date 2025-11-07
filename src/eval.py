import torch
from sklearn.metrics import accuracy_score

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(data.y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return acc
