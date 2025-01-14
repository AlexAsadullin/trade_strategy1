import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from copy import deepcopy as dc
import plotly.graph_objects as go

# Optimized: Combined normalization and reshaping to reduce redundant calculations.
def normalize_split_data(df: pd.DataFrame, train_part: float):
    train_size = int(len(df) * train_part)  # Size of training set
    df['Close'] /= 100000  # Normalize 'Close' column
    X = df.drop('next_ratio', axis=1).to_numpy()  # Directly convert to NumPy
    y = df['next_ratio'].to_numpy()

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# Optimized: Removed unnecessary `unsqueeze(1)` calls for reshaping.
def create_tensors(X_train, X_test, y_train, y_test):
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device, loss_function):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.device = device
        self.loss_function = loss_function

    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure that x has 3 dimensions (batch_size, sequence_length, input_size)
        if x.dim() == 2:  # If input is 2D, it lacks the sequence dimension
            x = x.unsqueeze(1)  # Adding a dummy sequence dimension

        # Now x should have shape (batch_size, sequence_length, input_size)
        # Example: (16, 31, 32) where batch_size=16, sequence_length=31, input_size=32

        # Initialize the hidden and cell states
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=self.device)  # 3D tensor
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size, device=self.device)  # 3D tensor
        
        # Pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        
        # Fully connected layer output
        out = self.fc(out[:, -1, :])  # Take the output from the last time step
        return out

    # Optimized: Simplified data loader iteration, reduced print calls for efficiency.
    def train_validate_one_epoch(self, train_loader, optimizer, epoch):
        self.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            output = self.forward(x_batch)
            loss = self.loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
        return total_loss / len(train_loader)

def _n_ratio(df, n):
    for i in range(1, n + 1):
        df[f'{i}_prev_ratio'] = df['Close'].shift(i) / df['Close'].shift(i + 1)
    return df

def calculate_metrics(df: pd.DataFrame, n_ratio: int, window_size: int):
    df['next_ratio'] = df['Close'].shift(-1) / df['Close']
    df = _n_ratio(df, n_ratio)
    df['prev_ratio_mean'] = df['1_prev_ratio'].rolling(window=window_size).mean()
    return df.dropna()

def main(data_path:str, model_save_path:str, chart_path:str):
    # Load and preprocess data
    data = pd.read_csv(data_path, index_col=0).drop('Date', axis=1)
    df = calculate_metrics(dc(data), n_ratio=15, window_size=50)
    df = df.dropna()
    df.to_csv('/home/alex/BitcoinScalper/dataframes/TSLA_RSI_LSTM.csv')

    print(df.head())
    print(df.columns)

    X_train, X_test, y_train, y_test = normalize_split_data(df, 0.8)
    X_train, X_test, y_train, y_test = create_tensors(X_train, X_test, y_train, y_test)

    # Prepare data loaders
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTM(input_size=X_train.shape[1], hidden_size=64, num_stacked_layers=2,
                 device=device, loss_function=nn.L1Loss()
                 ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    num_epochs = 40
    for epoch in range(num_epochs):
        avg_loss = model.train_validate_one_epoch(train_loader, optimizer, epoch)

    # Save model
    torch.save(model.state_dict(), model_save_path)

    # Inference and visualization
    model.eval()
    with torch.no_grad():
        predicted = model(X_test.to(device)).cpu().numpy()
    difference = predicted - y_test.numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(difference))), y=difference, mode='markers', marker=dict(size=5, color='blue'), name='Difference'))
    fig.add_trace(go.Scatter(x=list(range(len(predicted))), y=predicted, mode='markers', marker=dict(size=5, color='red'), name='Predicted'))
    fig.write_html(chart_path)

if __name__ == '__main__':
    main(data_path = '/home/alex/BitcoinScalper/dataframes/TSLA_RSI.csv',
        model_save_path = '/home/alex/BitcoinScalper/ML/models/lstm_tsla_model_state.pth',
        chart_path = '/home/alex/BitcoinScalper/html_charts/lstm_tsla_predict.html')