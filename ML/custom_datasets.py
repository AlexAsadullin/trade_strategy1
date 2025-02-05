from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size):
        self.X = X
        self.y = y
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        X_window = self.X[idx:idx + self.window_size]
        y_target = self.y[idx + self.window_size]  # Predict the next value after the window
        return np.array(X_window), y_target