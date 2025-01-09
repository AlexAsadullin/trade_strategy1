from lstm_train import normalize_split_data, create_trensors, TimeSeriesDataset, LSTM
import pandas as pd
from copy import deepcopy as dc
import plotly.graph_objects as go
import torch

device = 'cpu'
data = pd.read_csv('/home/alex/BitcoinScalper/dataframes/bullish_trend_metrics.csv')
df = dc(data)
df = df.dropna(axis='rows')

X_train, X_test, y_train, y_test = normalize_split_data(df, 0.8)
X_train, X_test, y_train, y_test = create_trensors(X_train, X_test, y_train, y_test)

train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

model = LSTM(1, 4, 1, device='cpu')
model.to(device)
model.load_state_dict(torch.load('ML/models/lstm_model_state.pth'))
model.eval()
# model = torch.load('models/lstm_model.pth')
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    predicted = model(X_test_tensor).detach().numpy()
    #predicted = model(X_test_tensor).to('cpu').numpy()

print(predicted, len(predicted))

difference = y_test - predicted
print(difference, len(difference))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(len(difference))),
    y=difference,
    mode='markers',
    marker=dict(size=5, color='blue'), #Use color for easy distinguishing, adjusted marker size.
    name='Difference' #Added name
))
fig.write_html('diff.html')

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(len(predicted))),
    y=predicted,
    mode='markers',
    marker=dict(size=5, color='red'),   #Use color for easy distinguishing, adjusted marker size.
    name='Predicted' #Added name
))
fig.write_html('pred.html')