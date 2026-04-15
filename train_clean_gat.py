import os

import torch
import torch.optim as optim
import torch.nn.functional as F

from models import GAT

import preprocess


def _prepare_clean_training_data(data):
    data = data.clone()
    data.x = data.x.float()

    if hasattr(preprocess, "normalize_features"):
        data = preprocess.normalize_features(data)

    if hasattr(preprocess, "sanitize_edge_index_binary"):
        data.edge_index = preprocess.sanitize_edge_index_binary(
            data.edge_index, data.x.size(0)
        )

    if hasattr(preprocess, "validate_data_consistency"):
        preprocess.validate_data_consistency(data, name="clean_train_data")

    return data

def train_gat(model, data, optimizer, epochs, model_save_path=None, device='cpu'):
    print(f"开始在 {device} 上训练干净的GAT模型...")
    model.to(device)
    data = data.to(device)

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            preds = val_out[data.val_mask].argmax(dim=1)
            correct = (preds == data.y[data.val_mask]).sum().item()
            val_acc = correct / max(1, data.val_mask.sum().item())

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if model_save_path:
                try:
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                    torch.save(best_model_state, model_save_path)
                except Exception:
                    pass
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[Clean GAT] epoch {epoch}/{epochs} "
                f"loss={loss.item():.4f} val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print("训练完成。")
    return model, best_val_acc


def test_gat(model, data, device='cpu'):
    model.to(device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out[data.test_mask].argmax(dim=1)
        correct = (preds == data.y[data.test_mask]).sum().item()
        acc = correct / max(1, data.test_mask.sum().item())
    print(f"测试集准确率: {acc:.4f}")
    return acc


def train_clean_model(data, dataset_name='dataset', device='cpu', epochs=200, hidden_channels=8, heads=8, dropout=0.6, lr=0.005, weight_decay=5e-4, output_dir='./clean_models'):
    data = _prepare_clean_training_data(data)

    num_node_features = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1

    model = GAT(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        heads=heads,
        dropout=dropout,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model_filename = f"gat_{str(dataset_name).lower()}_clean.pth"
    model_save_path = os.path.join(output_dir, model_filename)
    trained_model, best_val_accuracy = train_gat(model, data, optimizer, int(epochs), model_save_path, device)
    test_accuracy = test_gat(trained_model, data, device)

    return {
        'model': trained_model,
        'best_val_acc': float(best_val_accuracy),
        'test_acc': float(test_accuracy),
        'model_path': model_save_path,
    }

train_clean_gat = train_clean_model
train_clean_GAT = train_clean_model
