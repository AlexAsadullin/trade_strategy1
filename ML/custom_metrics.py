import torch.nn as nn
import torch.nn.functional as F
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

def das_metric_single(actuals, predictions):
    if not isinstance(actuals, torch.Tensor):
        actuals = torch.tensor(actuals, dtype=torch.float32)
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)

    # Убедимся, что форма [N, 1]
    if actuals.ndim == 1:
        actuals = actuals.unsqueeze(1)
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)

    sign_agreement = torch.sign(predictions - 1) * torch.sign(actuals - 1)
    confidence_weight = torch.abs(predictions - 1)
    return torch.mean(sign_agreement * confidence_weight).item()

class DirectionalLoss(nn.Module):
    def __init__(self, margin=1.0, bonus_weight=0.5):
        super().__init__()
        self.margin = margin
        self.bonus_weight = bonus_weight

    def forward(self, y_pred, y_true):
        base_loss = F.mse_loss(y_pred, y_true, reduction='none')  # (batch_size, 1)

        correct_direction = ((y_true > self.margin) & (y_pred > self.margin)).float()
        adjusted_loss = base_loss - (correct_direction * self.bonus_weight * base_loss)

        return adjusted_loss.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, under_weight=1.5, over_weight=0.5):
        super().__init__()
        self.under_weight = under_weight
        self.over_weight = over_weight

    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        loss = torch.where(
            diff < 0,  # Under-prediction
            torch.pow(diff, 2) * self.under_weight,
            torch.pow(diff, 2) * self.over_weight
        )
        return loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, loss_fn1, loss_fn2, alpha=0.5):
        """
        alpha — вес первой функции, (1 - alpha) — вес второй
        """
        super().__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        loss1 = self.loss_fn1(y_pred, y_true)
        loss2 = self.loss_fn2(y_pred, y_true)
        return self.alpha * loss1 + (1 - self.alpha) * loss2
