import torch
import numpy as np

def directional_accuracy_score(actuals, predictions):
    if not isinstance(actuals, torch.Tensor):
        actuals = torch.tensor(actuals, dtype=torch.float32)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)

    sign_agreement = torch.sign(predictions - 1) * torch.sign(actuals - 1)
    confidence_weight = torch.abs(predictions - 1)
    
    return torch.mean(sign_agreement * confidence_weight)

def wise_match_score(actuals, predictions):
    if not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    condition = (actuals > 1) & (predictions > 1) | (actuals < 1) & (predictions < 1)
    row_means = np.mean(condition.astype(int))  # Среднее по строкам

    return row_means
