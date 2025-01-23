import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from copy import deepcopy as dc
import plotly.graph_objects as go
from torch.utils.data import Dataset

from data_manipulations import split_data, create_tensors, prepare_data_ratio

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

def main(data_read_path:str, data_write_path:str, model_save_path:str, chart_path:str):
    # Load and preprocess data
    data = pd.read_csv(data_read_path, index_col=0)
    
    df = prepare_data_ratio(data, data_write_path=data_write_path, n_ratio=7, window_size=30)

    X_train, X_test, y_train, y_test = split_data(df, 0.8)
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

    """fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(difference))), y=difference, mode='markers', marker=dict(size=5, color='blue'), name='Difference'))
    fig.add_trace(go.Scatter(x=list(range(len(predicted))), y=predicted, mode='markers', marker=dict(size=5, color='red'), name='Predicted'))
    fig.write_html(chart_path)"""

if __name__ == '__main__':
    data = [
            [
            r"/home/alex/BitcoinScalper/dataframes/TSLA_cycles.csv",
            r"/home/alex/BitcoinScalper/dataframes/TSLA_cycles_TEST.csv" , 
            r"/home/alex/BitcoinScalper/ML/models/LSTM_TSLA_cycles.pkl"
                ],
            [
            r"/home/alex/BitcoinScalper/dataframes/TSLA_momentum.csv",
            r"/home/alex/BitcoinScalper/dataframes/TSLA_momentum_TEST.csv",
            r"/home/alex/BitcoinScalper/ML/models/LSTM_TSLA_momentum.pkl"
                ],
            [
            r"/home/alex/BitcoinScalper/dataframes/TSLA_overlap.csv", 
            r"/home/alex/BitcoinScalper/dataframes/TSLA_overlap_TEST.csv",
            r"/home/alex/BitcoinScalper/ML/models/LSTM_TSLA_overlap.pkl"
                ],
            [
            r"/home/alex/BitcoinScalper/dataframes/TSLA_performance.csv",
            r"/home/alex/BitcoinScalper/dataframes/TSLA_performance_TEST.csv",
            r"/home/alex/BitcoinScalper/ML/models/LSTM_TSLA_performance.pkl"
                ],
            [
            r"/home/alex/BitcoinScalper/dataframes/TSLA_trend.csv",
            r"/home/alex/BitcoinScalper/dataframes/TSLA_trend_TEST.csv",
            r"/home/alex/BitcoinScalper/ML/models/LSTM_TSLA_trend.pkl"
                ], # no
            [
            r"/home/alex/BitcoinScalper/dataframes/TSLA_volume.csv",
            r"/home/alex/BitcoinScalper/dataframes/TSLA_volume_TEST.csv",
            r"/home/alex/BitcoinScalper/ML/models/LSTM_TSLA_volume.pkl"
                ], # no     
        ]

    for data_read_path, data_write_path, model_save_path in data:
        main(data_read_path = data_read_path,
             data_write_path=data_write_path,
            model_save_path = model_save_path,
            chart_path = ''
            )
        print(model_save_path.split('/')[-1])