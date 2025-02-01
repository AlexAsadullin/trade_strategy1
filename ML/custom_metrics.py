import torch
import numpy as np

def directional_accuracy_score(y_test, y_pred):
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    sign_agreement = torch.sign(y_pred - 1) * torch.sign(y_test - 1)
    confidence_weight = torch.abs(y_pred - 1)
    
    return torch.mean(sign_agreement * confidence_weight)

def wise_match_score(y_test, y_pred):
    if not isinstance(y_test, np.ndarray):
        y_test = np.array(y_test)
    if not isinstance(y_test, np.ndarray):
        y_pred = np.array(y_pred)

    condition = (y_test > 1) & (y_pred > 1) | (y_test < 1) & (y_pred < 1)
    row_means = np.mean(condition.astype(int), axis=1)  # Среднее по строкам

    return row_means
