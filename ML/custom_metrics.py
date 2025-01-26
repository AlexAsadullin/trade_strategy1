import torch

def directional_accuracy_score(y_test, y_pred):
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    sign_agreement = torch.sign(y_pred - 1) * torch.sign(y_test - 1)
    confidence_weight = torch.abs(y_pred - 1)
    
    return torch.mean(sign_agreement * confidence_weight)