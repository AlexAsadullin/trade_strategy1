import torch
import numpy as np

def das_metric_solo(actuals, predictions):
    if not isinstance(actuals, torch.Tensor):
        actuals = torch.tensor(actuals, dtype=torch.float32)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)

    sign_agreement = torch.sign(predictions - 1) * torch.sign(actuals - 1)
    confidence_weight = torch.abs(predictions - 1)
    return torch.mean(sign_agreement * confidence_weight)
    
def wms_metric_solo(actuals, predictions):
    if not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    
    condition = (actuals > 1) & (predictions > 1) | (actuals < 1) & (predictions < 1)
    row_means = np.mean(condition.astype(int))  # Среднее по строкам

    return row_means

def das_metric_multi(actuals, predictions):
    if not isinstance(actuals, torch.Tensor):
        actuals = torch.tensor(actuals, dtype=torch.float32)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)

    # Преобразуем 1D массивы в 2D (нужно для обработки по столбцам)
    if actuals.ndim == 1:
        actuals = actuals.unsqueeze(1)  # [samples] -> [samples, 1]
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)  # [samples] -> [samples, 1]

    scores = []
    for i in range(actuals.shape[1]):  # Теперь точно 2D
        sign_agreement = torch.sign(predictions[:, i] - 1) * torch.sign(actuals[:, i] - 1)
        confidence_weight = torch.abs(predictions[:, i] - 1)
        scores.append(torch.mean(sign_agreement * confidence_weight).item())

    return np.mean(scores)
    
def wms_metric_multi(actuals, predictions):
    if not isinstance(actuals, np.ndarray):
        actuals = np.array(actuals)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    # Преобразуем 1D массивы в 2D
    if actuals.ndim == 1:
        actuals = actuals.reshape(-1, 1)  # [samples] -> [samples, 1]
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)  # [samples] -> [samples, 1]

    scores = []
    for i in range(actuals.shape[1]):  # Теперь точно 2D
        condition = ((actuals[:, i] > 1) & (predictions[:, i] > 1)) | ((actuals[:, i] < 1) & (predictions[:, i] < 1))
        scores.append(np.mean(condition.astype(int)))  # Среднее по строкам для i-го столбца

    return np.mean(scores)