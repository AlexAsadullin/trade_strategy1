import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from copy import deepcopy as dc
import plotly.graph_objects as go

def normalize_split_data(df: pd.DataFrame, train_part: float):
    train_size = int(len(df) * train_part)  # Определяем размер обучающей выборки
    df['price'] = df['price'] / 100000
    X = df.drop('next_ratio', axis=1) # axis="columns"
    y = df['next_ratio']

    X_train = X[:train_size].to_numpy()
    X_test = X[train_size:].to_numpy()
    y_train = y[:train_size].to_numpy()
    y_test = y[train_size:].to_numpy()

    part = len(X_train[0])
    X_train = X_train.reshape((-1, part, 1))
    X_test = X_test.reshape((-1, part, 1))
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))
    return X_train, X_test, y_train, y_test

def create_trensors(X_train, X_test, y_train, y_test):
    Xtrain = torch.tensor(X_train, dtype=torch.float32)
    ytrain = torch.tensor(y_train, dtype=torch.float32)
    Xtest = torch.tensor(X_test, dtype=torch.float32)
    ytest = torch.tensor(y_test, dtype=torch.float32)
    return Xtrain, Xtest, ytrain, ytest

from torch.utils.data import Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        self.device = device

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def train_validate_one_epoch(self, model, optimizer, loss_function, epoch):
        model.train(True)
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0
        epoch_loss = 0.0
        n_full_epochs = 0
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(model.device), batch[1].to(model.device)

            output = model.forward(x_batch)
            print(output)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            print(running_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 1000 == 999:  # print every 1000 batches
                avg_loss_across_batches = running_loss / 1000
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                epoch_loss += avg_loss_across_batches
                n_full_epochs += 1000
                running_loss = 0.0

        avg_loss_across_batches = epoch_loss / n_full_epochs
        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('*' * 25, end='\n\n')
        return model, optimizer, loss

def n_ratio(df, n):
    for i in range(n + 1):
        df[f'{i + 1}_prev_ratio'] = df['price'].shift(i) / df['price'].shift(i + 1)
    return df

def calculate_metrics(df, n):
    df['next_ratio'] = df['price'].shift(-1) / df['price']
    df = n_ratio(df, n)
    df['prev_ratio_mean'] = df['1_prev_ratio'].rolling(window=100).mean()
    df = df[df.columns[::-1]]
    return df

def main(data_path, model_save_path, chart_path):
    data = pd.read_csv(data_path, index_col=0)
    df = dc(data)
    df = calculate_metrics(df, 15)
    #df.to_csv('/home/alex/BitcoinScalper/dataframes/bullish_trend_metrics.csv')
    lookback = 7
    
    X_train, X_test, y_train, y_test = normalize_split_data(df, 0.8)
    X_train, X_test, y_train, y_test = create_trensors(X_train, X_test, y_train, y_test)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    from torch.utils.data import DataLoader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = 'cpu'
    for i, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        print(x_batch.shape, y_batch.shape)
        print(x_batch.shape, y_batch.shape)
        break

    model = LSTM(1, 4, 1, device=device)
    model.to(model.device)

    learning_rate = 0.001
    num_epochs = 10
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model, optimizer, loss_function = model.train_validate_one_epoch(model, optimizer, loss_function, epoch)
    torch.save(model.state_dict(), model_save_path)
    #torch.save(model, 'ML/models/lstm_model_pure.pth')
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save('lstm_model_ts.pt')

    with torch.no_grad():
        predicted = model(X_test.to(model.device)).to('cpu').numpy()

    difference = predicted - y_test.numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(difference))),
        y=difference,
        mode='markers',
        marker=dict(size=5, color='blue'),  # Use color for easy distinguishing, adjusted marker size.
        name='Difference'  # Added name
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(predicted))),
        y=predicted,
        mode='markers',
        marker=dict(size=5, color='red'),  # Use color for easy distinguishing, adjusted marker size.
        name='Predicted'  # Added name
    ))
    fig.write_html(chart_path)

if __name__ == '__main__':
    main(data_path=r'/home/alex/BitcoinScalper/dataframes/bullish_trend.csv',
         model_save_path=r'/home/alex/BitcoinScalper/ML/models/lstm_model_state.pth',
         chart_path=r'/homw/alex/BitcoinScalper/charts/lstm_predict.html')