from lstm_train import split_data, create_tensors, TimeSeriesDataset, LSTM
import pandas as pd
from copy import deepcopy as dc
import plotly.graph_objects as go
import torch
import numpy as np
import torch.nn as nn

from custom_metrics import directional_accuracy_score

def test_lstm(data_read_path, model_read_path, chart_path):
    device = 'cpu'
    df = pd.read_csv(data_read_path, index_col=0)

    X_train, X_test, y_train, y_test = split_data(df, 0.8)
    X_train, X_test, y_train, y_test = create_tensors(X_train.astype(np.float32),
                                                      X_test.astype(np.float32),
                                                       y_train.astype(np.float32),
                                                       y_test.astype(np.float32))

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTM(input_size=X_train.shape[1], hidden_size=64, num_stacked_layers=2,
                 device=device, loss_function=nn.L1Loss()
                 ).to(device)
    
    model.load_state_dict(torch.load(model_read_path))
    model.eval()
    # model = torch.load('models/lstm_model.pth')
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        predicted = model(X_test_tensor).detach().numpy()
        #predicted = model(X_test_tensor).to('cpu').numpy()

    print(predicted, len(predicted))

    difference = y_test - predicted
    print(difference, len(difference))

    print('directional accuracy:', directional_accuracy_score(y_test=y_test, y_pred=predicted))
    # plot this

    # plot real vs predicted only on y_test
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(difference))),
        y=difference,
        mode='markers',
        marker=dict(size=5, color='blue'), #Use color for easy distinguishing, adjusted marker size.
        name='Difference' #Added name
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(predicted))),
        y=predicted,
        mode='markers',
        marker=dict(size=5, color='red'),   #Use color for easy distinguishing, adjusted marker size.
        name='Predicted' #Added name
    ))
    fig.write_html(chart_path)

if __name__ == '__main__':
    test_lstm(data_read_path=r"/home/alex/BitcoinScalper/dataframes/TSLA_momentum_TEST.csv",
               model_read_path=r"/home/alex/BitcoinScalper/ML/models/LSTM_TSLA_momentum.pkl",
               chart_path=r'/home/alex/BitcoinScalper/html_charts/LSTM_TSLA_momentum.html')