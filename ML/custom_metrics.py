import torch
import numpy as np

def directional_accuracy_score(actuals, predictions):
    if not isinstance(actuals, torch.Tensor):
        actuals = torch.tensor(actuals, dtype=torch.float32)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)

    """sign_agreement = torch.sign(predictions - 1) * torch.sign(actuals - 1)
    confidence_weight = torch.abs(predictions - 1)
    return torch.mean(sign_agreement * confidence_weight)"""
    
    scores = []
    for i in range(actuals.shape[1]):  # Проходим по каждому столбцу
        sign_agreement = torch.sign(predictions[:, i] - 1) * torch.sign(actuals[:, i] - 1)
        confidence_weight = torch.abs(predictions[:, i] - 1)
        scores.append(torch.mean(sign_agreement * confidence_weight).item())

    return np.mean(scores)

    
def wise_match_score(actuals, predictions):
    if not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    scores = []
    for i in range(actuals.shape[1]):  # Проходим по каждому столбцу
        condition = ((actuals[:, i] > 1) & (predictions[:, i] > 1)) | ((actuals[:, i] < 1) & (predictions[:, i] < 1))
        scores.append(np.mean(condition.astype(int)))  # Среднее по строкам для i-го столбца

    return np.mean(scores)

    """condition = (actuals > 1) & (predictions > 1) | (actuals < 1) & (predictions < 1)
    row_means = np.mean(condition.astype(int))  # Среднее по строкам

    return row_means"""
